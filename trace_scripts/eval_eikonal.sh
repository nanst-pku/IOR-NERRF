CUDA_LAUNCH_BLOCKING=5 \
python eval/eval_approx.py \
-n eikonal_wrongior \
-c NeRRFngp.conf \
-D data/glass_ball \
--exp_name=eikonal_wrongior \
--gpu_id=0 \
--stage 2 \
--tet_scale 2.0 \
--sphere_radius 2.40 \
--enable_refr \
--enable_refl \
--need_trace \
--ior 1.3 \
# --use_cone
# --use_progressive_encoder

