CUDA_LAUNCH_BLOCKING=5 \
python eval/eval_approx.py  \
-n cow_ior_optim_trace3 \
-c NeRRFsngp.conf \
-D data/blender/transparent/cow \
--exp_name=cow_final \
--gpu_id=0 \
--stage 2 \
--tet_scale 4.2 \
--sphere_radius 2.30 \
--enable_refr \
--enable_refl \
--ior 1.19 \
--need_trace \
# --use_cone
# --use_sdf 
# --use_progressive_encoder