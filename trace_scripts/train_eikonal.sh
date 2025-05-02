CUDA_LAUNCH_BLOCKING=5 \
python train/train.py \
-n eikonal_wrongior \
-c NeRRFngp.conf \
-D data/glass_ball \
--gpu_id=0 \
--visual_path tet_visual \
--stage 2 \
--tet_scale 2.7 \
--sphere_radius 2.40 \
--enable_refr \
--need_trace \
--enable_refl \
--ior 1.3 \
# --use_progressive_encoder
# --use_grid \
# --use_cone
