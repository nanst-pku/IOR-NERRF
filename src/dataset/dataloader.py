import imageio
import numpy as np
import cv2
import json
import math
import os
import sys
from os.path import join
from PIL import Image

sys.path.append("..")
import dataset.utils as api_utils
import torch
from torchvision import transforms
from PIL import Image
import random


def get_image_to_tensor_balanced(image_size):
    ops = []
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    ops.append(transforms.Resize(image_size))
    return transforms.Compose(ops)


def get_focal(meta_path, image_size, dataset_type):
    W, H = image_size
    camera_pose = np.eye(4, dtype=np.float32)
    with open(meta_path) as f:
        meta_data = json.load(f)
        #{"cam_world_pose": [[0.9320324063301086, -0.3605630695819855, 0.0361921451985836, 0.36192163825035095], 
        # [0.3623749315738678, 0.9273722171783447, -0.09308665245771408, -0.9308666586875916],
        # [1.9395058004079146e-08, 0.09987490624189377, 0.9950000047683716, 9.949999809265137], 
        # [0.0, 0.0, 0.0, 1.0]]}
        camera_pose = np.array(meta_data["cam_world_pose"])
        if dataset_type == "blender":
            #根据视场角计算焦距
            dx = math.radians(60)
            fx = (W / 2) / math.tan(dx / 2)
            fy = fx
        elif dataset_type[0:7]  == "eikonal":
            fx = fy = meta_data["f"]
        else:
            raise NotImplementedError
    camera_pose[..., 3] = camera_pose[..., 3]
    return fx, fy, camera_pose


def get_mvp(focal, pose, image_size, far=50, near=0.001):
    W, H = image_size
    projection = np.array(
        [
            [2 * focal / W, 0, 0, 0],
            [0, -2 * focal / H, 0, 0],
            [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )
    return projection @ np.linalg.inv(pose)  # [4, 4]


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        stage="train",
        dataset_type="blender",  # if use eikonal dataset, change this to "eikonal"
        stage_=1,
        no_ref=False
    ):
        super().__init__()
        #按照
        self.split = stage
        self.type = dataset_type
        self.stage=stage_

        print('data_dir:',data_dir)
        print('type:',self.type)
        if self.type == "blender":
            self.image_size = (480, 270)
            #远近平面
            self.z_near, self.z_far = 0.02, 80
            self.image_dir = data_dir
            #深度图
            self.depth_dir = data_dir + "/../../depth/" + os.path.basename(data_dir)
            #相机参数
            self.meta_dir = data_dir + "/../../meta"
            #划分训练集和测试集
            if self.split == "train":
                self.image_list = [str(3 * d) for d in range(33)]+ [str(3 * d + 1) for d in range(33)]+['99']
            else:
                self.image_list = [str(3 * d + 2) for d in range(33)]
            self.center=torch.tensor([[240,135]])
        elif self.type[0:7]  == "eikonal":
            self.image_size = (672, 504)
            self.z_near, self.z_far = 0.05, 8
            self.image_dir = data_dir + "/images"
            self.mask_dir = data_dir + "/mask"
            self.meta_dir = data_dir + "/meta"
            self.center=torch.tensor([[331.01,248.55]])
            file_names = os.listdir(self.mask_dir)
            import random
            random.seed(0)
            train_list=random.sample(file_names,75)
            test_list=[name for name in file_names if name not in train_list]
            if self.split == "train":
                self.image_list = [name[5:-4] for name in train_list]
            else:
                self.image_list = [name[5:-4] for name in test_list]
        else:
            raise NotImplementedError

        self.image2tensor = get_image_to_tensor_balanced(
            #首先将图片转换为tensor，然后进行归一化
            (self.image_size[1], self.image_size[0])
        )
        name, focal, poses, images, mvps, masks, centers = (
            [],
            [],
            [],
            [],
            [],
            [],
            []
        )

        for i in range(len(self.image_list)):
            result = self.__get_single_item__(i)
            name.append(result["name"])
            focal.append(result["focal"])
            poses.append(result["poses"])
            images.append(result["images"])
            mvps.append(result["mvp"])
            masks.append(result["mask"])
            centers.append(self.center)

        results = {
            "name": name,
            "focal": torch.stack(focal),
            "poses": torch.stack(poses),
            "images": torch.stack(images),
            "mvp": torch.stack(mvps),
            "mask": torch.stack(masks),
            "center": torch.stack(centers),
        }
        self.results = results

    def __len__(self):
        return len(self.image_list)

    def __get_single_item__(self, index):
        name = self.image_list[index]
        if self.type == "blender":
            img_name = name + "_0001.png"
            meta_name = name + "_meta_0000.json"
            #深度图
            depth_name = name + "_depth_0001.exr"
            depth_path = join(self.depth_dir, depth_name)
            depth = api_utils.exr_loader(depth_path, ndim=1)
            dd = depth
            if self.stage==1:
                dd = cv2.resize(depth, dsize=(480, 272))
            #深度图小于100的像素为mask,采用mask引导光线追踪
            #if not self.no_ref:
            mask = dd < 100
            mask = torch.tensor(mask).unsqueeze(0)
        elif self.type[0:7] == "eikonal":
            img_name = name + ".JPG"
            meta_name = name + ".json"
            mask_path = join(
                self.mask_dir,
                "mask_" + name + ".png",
            )
            

            image = Image.open(mask_path)
            density = np.array(image)[:, :]
            mask = density > 0.5
            image = Image.fromarray(mask.astype("uint8") * 255)
            #image.save("test_mask.png")
            #if not self.no_ref:
            mask = torch.tensor(mask).unsqueeze(0)
            
        else:
            raise NotImplementedError
        #读取图片
        image_path = join(self.image_dir, img_name)
        #取出图片的RGB通道
        img = imageio.imread(image_path)[..., :3]
        #将图片转换为tensor
        img = self.image2tensor(img)
        #读取相机参数
        meta_path = join(self.meta_dir, meta_name)
        fx, fy, camera_pose = get_focal(meta_path, self.image_size, self.type)
        
        
        center=self.center
        #计算mvp矩阵
        mvp = get_mvp(fx, camera_pose, self.image_size)
        result = {
            "name": name,
            "focal": torch.tensor((fx, fy), dtype=torch.float32),
            "poses": torch.tensor(camera_pose, dtype=torch.float32),
            "images": img,
            "mvp": torch.tensor(mvp, dtype=torch.float32),
            "mask": mask,
            "center":torch.tensor(center)
        }
        return result

    def __getitem__(self, index):
        result = {
            "name": self.results["name"][index],
            "focal": self.results["focal"][index],
            "poses": self.results["poses"][index],
            "images": self.results["images"][index],
            "mask": self.results["mask"][index],
            "mvp": self.results["mvp"][index],
            "center":self.results["center"][index]
        }
        return result
    
    def get_item(self,index):
        tmp_result= self.__getitem__(index)
        for key,value in tmp_result.items():
            tmp_result[key]=[value]
        return tmp_result
