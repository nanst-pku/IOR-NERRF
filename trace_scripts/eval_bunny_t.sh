CUDA_LAUNCH_BLOCKING=5 \
python eval/eval_approx.py  \
-n bunny_random_trace \
-c NeRRFsngp.conf \
-D data/blender/transparent/bunny \
--exp_name=bunny_noise \
--gpu_id=0 \
--stage 2 \
--tet_scale 3.8 \
--sphere_radius 2.43 \
--enable_refr \
--enable_refl \
--ior 1.2 \
--need_trace \
# --use_cone
# --use_sdf
# --use_progressive_encoder