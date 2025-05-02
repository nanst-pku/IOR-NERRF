import cv2
import numpy as np
import os


def crop(mask,img1):
    white_pts=np.where(mask>0)#[0]
    h,w,_=img1.shape
    img1=img1[h//2:,w//2:]
    topleft_x=np.min(white_pts[0])
    topleft_y=np.min(white_pts[1])
    bottomright_x=np.max(white_pts[0])
    bottomright_y=np.max(white_pts[1])
    mask[mask>0]=1
    new_img1=img1[topleft_x:bottomright_x+1,topleft_y:bottomright_y+1]*mask[topleft_x:bottomright_x+1,topleft_y:bottomright_y+1,np.newaxis]
    #new_img1[mask[topleft_x:bottomright_x+1,topleft_y:bottomright_y+1,np.newaxis]==0]=255
    new_img1[new_img1==0]=255
    
    return new_img1

img_name='tet_visual/ball_notrace_test/2_0000_5999_vis.png'
mask_name='eval_result/ball_norfl/mask_rgbs_gt14.png'

nerf=cv2.imread(img_name)

masked_nerf=cv2.imread(mask_name)
mask=cv2.cvtColor(masked_nerf,cv2.COLOR_BGR2GRAY)
masked_nerf=crop(mask,nerf)
cv2.imwrite('masked.jpg',masked_nerf)
