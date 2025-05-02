CUDA_LAUNCH_BLOCKING=5 \
python eval/eval_approx.py \
-n ball_ior_optim_trace_ior11 \
-c NeRRFsngp.conf \
-D data/blender/transparent/ball \
--exp_name=ball_withF \
--gpu_id=0 \
--stage 2 \
--tet_scale 4.2 \
--sphere_radius 2.30 \
--enable_refr \
--ior 1.3 \
--need_trace \
# --use_cone
# --use_sdf
# --use_progressive_encoder