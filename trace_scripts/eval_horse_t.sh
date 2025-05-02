CUDA_LAUNCH_BLOCKING=5 \
python eval/eval_approx.py \
-n horse_ior_optim_orinerf \
-c NeRRFsngp.conf \
-D data/blender/transparent/horse \
--exp_name=horse_orinerf \
--gpu_id=0 \
--stage 2 \
--tet_scale 3.8 \
--sphere_radius 2.40 \
--enable_refr \
--enable_refl \
--ior 1.5
# --use_cone
# --use_progressive_encoder