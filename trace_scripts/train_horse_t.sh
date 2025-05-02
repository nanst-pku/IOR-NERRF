CUDA_LAUNCH_BLOCKING=5 \
python train/train.py \
-n horse_ior_optim_trace \
-c NeRRFsngp.conf \
-D data/blender/transparent/horse \
--gpu_id=0 \
--visual_path tet_visual \
--stage 2 \
--tet_scale 3.8 \
--sphere_radius 2.40 \
--resume \
--enable_refr \
--need_trace \
--enable_refl \
--ior 1.2
# --use_cone
# --use_sdf
# --use_progressive_encoder