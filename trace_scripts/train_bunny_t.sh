CUDA_LAUNCH_BLOCKING=5 \
python train/train.py \
-n bunny_random_trace \
-c NeRRFsngp.conf \
-D data/blender/transparent/bunny \
--gpu_id=0 \
--visual_path tet_visual \
--stage 2 \
--tet_scale 3.8 \
--sphere_radius 2.43 \
--enable_refr \
--ior 1.3 \
--enable_refr \
--need_trace \
--enable_refl \
#--resume \

# --use_cone \
# --use_sdf \
# --use_progressive_encoder