import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ["PATH"] = os.environ["PATH"] + ':/home/jiangn/anaconda3/envs/nerrf/bin/ninja'
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
sys.path.append("dataloader")


import trainlib
from model import make_model, loss
from render import NeRFRenderer
import random
import util
import numpy as np
import torch
from dotmap import DotMap
from dataset.dataloader import Dataset
import cv2
import trimesh
#torch.autograd.set_detect_anomaly(True)
def extra_args(parser):
    parser.add_argument(
        "--batch_size", "-B", type=int, default=4, help="Object batch size ('SB')"
    )
    parser.add_argument(
        "--freeze_enc",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--no_bbox_step",
        type=int,
        default=100000,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument(
        "--fixed_test",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--enable_refr",
        action="store_true",
        default=False,
        help="Whether to enable refraction",
    )
    parser.add_argument(
        "--enable_refl",
        action="store_true",
        default=False,
        help="Whether to enable reflection",
    )
    parser.add_argument(
        "--use_cone",
        action="store_true",
        default=False,
        help="Whether to use cone sampling",
    )
    parser.add_argument(
        "--use_grid",
        action="store_true",
        default=False,
        help="Use grid param or MLP to predict sdf and deform",
    )
    parser.add_argument(
        "--use_sdf",
        action="store_true",
        default=False,
        help="Use sdf based intersection, aka sphere tracing or ray marching",
    )
    parser.add_argument(
        "--use_progressive_encoder",
        action="store_true",
        default=True,
        help="Whether to use progressive encoder",
    )
    parser.add_argument(
        "--stage",
        "-S",
        type=int,
        default=1,
        help="Stage of training, 1: optimize geometry, 2: optimize envmap",
    )
    parser.add_argument("--tet_scale", type=float, default=4.2, help="Scale of the tet")
    parser.add_argument(
        "--sphere_radius", type=float, default=2.30, help="Radius of the bounding sphere"
    )
    parser.add_argument("--ior", type=float, default=1.1, help="index of refraction")
    return parser


# parse args
args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size= 640)
print('args:',args)
print('conf:',conf)
#
device = util.get_cuda(args.gpu_id[0])
print('device:',device)
if args.stage != 1 and args.stage != 2:
    raise NotImplementedError()

# partition dataset
train_dset = Dataset(args.datadir, stage="train",dataset_type=args.name[0:7] if args.name[0:7] =='eikonal'else 'blender',stage_=args.stage,no_ref=False)
test_dset = Dataset(args.datadir, stage="test",dataset_type=args.name[0:7]if args.name[0:7] =='eikonal'else 'blender',stage_=args.stage,no_ref=False)

# network
net = make_model(conf["model"]).to(device=device)

# renderer
renderer = NeRFRenderer.from_conf(
    conf["renderer"],
    enable_refr=args.enable_refr,
    enable_refl=args.enable_refl,
    stage=args.stage,
    tet_scale=args.tet_scale,
    sphere_radius=args.sphere_radius,
    ior=args.ior,
    use_cone=args.use_cone,
    use_grid=args.use_grid,
    use_sdf=args.use_sdf,
    use_progressive_encoder=True,
    need_trace=args.need_trace,
).to(device=device)

# parallize
render_par = renderer.bind_parallel(net, args.gpu_id).eval()


class RRFTrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(net, train_dset, test_dset, args, conf["train"], device=device)
        self.renderer_state_path = "%s/%s/_renderer" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        print(
            "lambda coarse {} and fine {}".format(self.lambda_coarse, self.lambda_fine)
        )
        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True) #MSEloss
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        if args.stage == 1:
            net.eval()
            self.optim = torch.optim.Adam(
                [
                    {
                        "params": [
                            p
                            for n, p in renderer.named_parameters()
                            if ("sdf" in n) or ("deform" in n) and p.requires_grad
                        ],
                        "lr": 0.001,
                    },
                ],
            )
        elif args.stage == 2:
            self.optim = torch.optim.Adam(
                [
                    {
                        "params": [p for n, p in net.named_parameters()],
                        "lr": 0.01,  # 0.01 for ngp, 5e-4 for nerf
                    },
                ]
            )
        else:
            raise NotImplementedError()

        # load renderer paramters
        if os.path.exists(self.renderer_state_path):
            renderer.load_state_dict(
                torch.load(self.renderer_state_path, map_location=device), False
            )

        # load mesh rendered in stage 1
        if args.stage == 2:
            if not args.use_sdf:
                renderer.init_tet(#慢
                    mesh_path="data/learned_geo/" + args.name.split("_")[0] + ".obj"
                )
        elif args.stage != 1:
            raise NotImplementedError()

        self.z_near = train_dset.z_near
        self.z_far = train_dset.z_far

        self.use_bbox = args.no_bbox_step > 0
        

    def post_batch(self, epoch, batch):
        renderer.sched_step(args.batch_size)

    def extra_save_state(self, global_step):
        torch.save(renderer.state_dict(), self.renderer_state_path)
        mesh = renderer.export_mesh(global_step=global_step)
        geo_path = "data/learned_geo/"
        if not os.path.exists(geo_path):
            os.makedirs(geo_path)
        mesh.export(geo_path + args.name + str(global_step) + ".obj")

    def calc_losses(self, data, is_train=True, global_step=0):
        stage = args.stage
        image = data["images"][0].to(device=device)  # (3, H, W)
        
        pose = data["poses"][0].to(device=device)  # (4, 4)
        focal = data["focal"][0].to(device=device)  # (2)
        mvp = data["mvp"][0].to(device=device)  # (4, 4)
        mask = data["mask"][0].to(device=device).float()  # (1,H, W)
        center = data["center"][0].to(device=device)  # (2)
        _, H, W = image.shape  # (3, H, W)

        cam_rays = util.gen_rays(
            pose, W, H, focal, self.z_near, self.z_far, c=center
        )  # (H, W, 8)
        
        apply_mask = False
        rgbs_gt = image * 0.5 + 0.5  # (3, H, W)
        rgbs_gt = rgbs_gt.permute(1, 2, 0).contiguous().reshape(-1, 3)  # (H * W, 3)
        if not self.rrf_stage=='None' and stage==2:
            mask=mask.permute(1,2,0)
            cam_rays=torch.cat([cam_rays,mask],dim=-1)
        
        if not self.rrf_stage=='all' and stage==2:
            mask_flatten = mask[:, :H, :].reshape(-1)
            
            if self.rrf_stage=='background':
                hit_idx = torch.where(mask_flatten < 0.5)[0]
                perm = torch.randperm(hit_idx.numel())
                selected_indices = perm[: args.ray_batch_size]
            elif self.rrf_stage=='foreground':
                hit_idx = torch.where(mask_flatten > 0.5)[0]
                perm = torch.randperm(hit_idx.numel())
                ray_batch_size = int(torch.sum(mask_flatten)/torch.max(mask_flatten))-1
                selected_indices = perm[: ray_batch_size]
            pix_inds = hit_idx[selected_indices]
            #print('mask',self.rrf_stage)
        else:
            #print('no mask')
            pix_inds = torch.randint(0, H * W, (args.ray_batch_size,))
        samp_rgbs_gt = rgbs_gt[pix_inds].unsqueeze(0)  # (1, ray_batch_size, 3)
        samp_rays = (
            cam_rays.view(-1, cam_rays.shape[-1])[pix_inds]
            .to(device=device)
            .unsqueeze(0)
        )  # (1, ray_batch_size, 9)
        loss_dict = {}
        
        if stage == 1:
            mask_loss = 0.0
            ek_loss = 0.0
            #tmp_mesh=renderer.export_mesh(global_step=global_step)
            #tmp_mesh.export('tmp.obj')
            mask_, ek_loss = renderer.render_mask(
                samp_rays, mvp, h=H, w=W, global_step=global_step
            )
            mask_loss = torch.nn.functional.mse_loss(mask, mask_)
            #print('sum:',mask_.sum(),'loss:',mask_loss.item())
            step=random.randint(0,2)
            
            #cv2.imwrite(f'mask{step}_.bmp',mask.permute(1,2,0).detach().cpu().numpy()*255)
            #cv2.imwrite(f'mask{step}__.bmp',mask_.detach().permute(1,2,0).cpu().numpy()*255)
            
            loss_dict["mask"] = mask_loss.item()
            loss_dict["eikonal"] = ek_loss
            loss = mask_loss + ek_loss
            if is_train:
                loss.backward()
            loss_dict["t"] = loss.item()
            return loss_dict
        elif stage == 2:
            
            render_dict = DotMap(render_par(samp_rays, want_weights=True))
            coarse = render_dict.coarse
            fine = render_dict.fine
            using_fine = len(fine) > 0
            rgb_loss = self.rgb_coarse_crit(coarse.rgb, samp_rgbs_gt)
            loss_dict["rc"] = rgb_loss.item() * self.lambda_coarse
            if using_fine:
                if apply_mask:
                    raise NotImplementedError()
                fine_loss = self.rgb_fine_crit(fine.rgb, samp_rgbs_gt)
                rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
                loss_dict["rf"] = fine_loss.item() * self.lambda_fine
            loss = rgb_loss

            if is_train:
                loss.backward()

            loss_dict["t"] = loss.item()
            return loss_dict

    def train_step(self, data, global_step):
        return self.calc_losses(data, is_train=True, global_step=global_step)

    def eval_step(self, data, global_step):
        renderer.eval()
        losses = self.calc_losses(data, is_train=False, global_step=global_step)
        renderer.train()
        return losses

    def vis_step(self, data, global_step, idx=None):
        if renderer.need_trace and renderer.enable_refl:
            print(('enable reflation'))
            renderer.enable_refl = True
        image = data["images"][0].to(device=device)  # (3, H, W)
        pose = data["poses"][0].to(device=device)  # (4, 4)
        focal = data["focal"][0].to(device=device)  # (2)
        _, H, W = image.shape  # (3, H, W)
        center = data["center"][0].to(device=device)
        cam_rays = util.gen_rays(
            pose, W, H, focal, self.z_near, self.z_far, c=center
        )  # (H, W, 8)
        rgbs_gt = image * 0.5 + 0.5  # ( 3, H, W)
        renderer.eval()
        gt = rgbs_gt.permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
        with torch.no_grad():
            test_rays = cam_rays  # (H, W, 8)
            test_rays = test_rays.reshape(1, H * W, -1)

            num_split = 1  # to avoid OOM
            test_rays_list = torch.chunk(test_rays, num_split, dim=1)
            alpha_coarse_np_list, rgb_coarse_np_list = [], []
            alpha_fine_np_list, rgb_fine_np_list = [], []
            for rays in test_rays_list:
                render_dict = DotMap(render_par(rays, want_weights=True))
                coarse = render_dict.coarse
                fine = render_dict.fine
                using_fine = len(fine) > 0
                alpha_coarse_np = coarse.weights[0].sum(dim=-1)
                rgb_coarse_np = coarse.rgb[0]
                alpha_coarse_np_list.append(alpha_coarse_np.cpu().numpy())
                rgb_coarse_np_list.append(rgb_coarse_np.cpu().numpy())

                if using_fine:
                    alpha_fine_np = fine.weights[0].sum(dim=1)
                    rgb_fine_np = fine.rgb[0]
                    alpha_fine_np_list.append(alpha_fine_np.cpu().numpy())
                    rgb_fine_np_list.append(rgb_fine_np.cpu().numpy())

            alpha_coarse_np = np.concatenate(alpha_coarse_np_list, axis=0).reshape(H, W)
            rgb_coarse_np = np.concatenate(rgb_coarse_np_list, axis=0).reshape(H, W, 3)

            if len(alpha_fine_np_list) > 0:
                alpha_fine_np = np.concatenate(alpha_fine_np_list, axis=0).reshape(H, W)
                rgb_fine_np = np.concatenate(rgb_fine_np_list, axis=0).reshape(H, W, 3)

        print("c rgb min {} max {}".format(rgb_coarse_np.min(), rgb_coarse_np.max()))
        print(
            "c alpha min {}, max {}".format(
                alpha_coarse_np.min(), alpha_coarse_np.max()
            )
        )

        vis_list = [
            gt,
            rgb_coarse_np,
            # alpha_coarse_cmap,
        ]

        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        if using_fine:
            print("f rgb min {} max {}".format(rgb_fine_np.min(), rgb_fine_np.max()))
            print(
                "f alpha min {}, max {}".format(
                    alpha_fine_np.min(), alpha_fine_np.max()
                )
            )
            vis_list = [
                gt,
                rgb_fine_np,
                # alpha_fine_cmap,
            ]

            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))
            rgb_psnr = rgb_fine_np
        else:
            rgb_psnr = rgb_coarse_np

        psnr = util.psnr(rgb_psnr, gt)
        vals = {"psnr": psnr}
        print("psnr", psnr)

        # set the renderer network back to train mode
        renderer.train()
        renderer.enable_refl = False
        return vis, vals

    def test_step(self, data, global_step, idx=None):
        return self.eval_step(data, global_step)
    
    def get_ior(self, num_step):
        if renderer.need_trace ==False:
            return []
        assert renderer.ior_range[0]< renderer.ior_range[1] , "ior_range should be a tuple with two elements"
        ior_list = np.linspace(renderer.ior_range[0], renderer.ior_range[1], num_step)
        min_loss=100000
        best_ior=renderer.ior
        renderer.eval()
        losses_list=[]
        with torch.no_grad():
            for i in range(num_step):
                renderer.ior = ior_list[i]
                total_loss=0
                data_list = self.test_data_loader.dataset
                random_pick_num=10
                random_idx = np.random.randint(0, len(data_list), random_pick_num)
                datas = [data_list.get_item(idx) for idx in random_idx]
                
                for i in range(random_pick_num):
                    data = datas[i]
                    self.rrf_stage='foreground'
                    losses = self.test_step(data, global_step=0)
                    total_loss+= losses["rf"]
                losses_list.append(total_loss)
                if total_loss<min_loss:
                    min_loss=total_loss
                    best_ior=renderer.ior
        renderer.ior=best_ior
        print('best ior:',best_ior, 'ior range:',renderer.ior_range)
        with open(f'tet_visual/{self.exp_name}/ior_list.txt','a') as f:
            f.write('ior list:'+str(ior_list)+'\n')
            f.write('loss list:'+str(losses_list)+'\n')
            f.write('best ior:'+str(best_ior)+' ior range:'+str(renderer.ior_range)+'\n')
        new_range=(renderer.ior_range[1]-renderer.ior_range[0])*0.4
        self.rrf_stage='all'
        renderer.ior_range=(best_ior-new_range/2,best_ior+new_range/2)
        renderer.train()
        return ior_list

#1min
trainer = RRFTrainer()
trainer.start()
