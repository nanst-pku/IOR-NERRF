import cv2
import numpy as np
import os
from lpips import LPIPS
import torch
compare_lpips = LPIPS(net="squeeze").cpu()
def crop(mask,img1,img2):
    white_pts=np.where(mask>0)#[0]
    topleft_x=np.min(white_pts[0])
    topleft_y=np.min(white_pts[1])
    bottomright_x=np.max(white_pts[0])
    bottomright_y=np.max(white_pts[1])
    new_img1=img1[topleft_x:bottomright_x+1,topleft_y:bottomright_y+1]
    new_img2=img2[topleft_x:bottomright_x+1,topleft_y:bottomright_y+1]
    return new_img1,new_img2,np.concatenate([img1,img2],axis=1)

exp_name='eikonal_wrongior'
print('exp:',exp_name)
root=f'eval_result/{exp_name}'
output_path=f'{root}/output'
os.makedirs(output_path,exist_ok=True)
imgs=os.listdir(root)
sum_lp=0
for i in range(1,100):
    masked_nerf=f'{root}/mask_rgbs_{i}.png'
    nerf=f'{root}/rgbs_gt{i}.png'
    gt=f'{root}/rgbs_{i}.png'
    masked_gt=f'{root}/mask_rgbs_gt{i}.png'
    if not os.path.exists(masked_nerf):
        break
    masked_nerf=cv2.imread(masked_nerf)
    nerf=cv2.imread(nerf)
    gt=cv2.imread(gt)
    masked_gt=cv2.imread(masked_gt)
    mask=cv2.cvtColor(masked_nerf,cv2.COLOR_BGR2GRAY)
    masked_nerf,masked_gt,total_mask=crop(mask,masked_nerf,masked_gt)
    lp=compare_lpips(torch.tensor(masked_nerf).permute(2,0,1),torch.tensor(masked_gt).permute(2,0,1)).sum().item()
    #print(lp)
    sum_lp+=lp
    cv2.imwrite(f'{output_path}/masked_rgbs{i}.png',masked_nerf)
    cv2.imwrite(f'{output_path}/masked_gt{i}.png',masked_gt)
    cv2.imwrite(f'{output_path}/masked_total{i}.png',total_mask)
print('average lpips:', sum_lp/(i-1))