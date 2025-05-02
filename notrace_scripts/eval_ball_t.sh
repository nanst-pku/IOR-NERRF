CUDA_LAUNCH_BLOCKING=5 \
python eval/eval_approx.py \
-n ball_orinerf \
-c NeRRFsngp.conf \
-D data/blender/transparent/ball \
--exp_name ball_orinerf \
--gpu_id=0 \
--stage 2 \
--tet_scale 5.2 \
--sphere_radius 2.30 \
--ior 1.3 \
# --use_cone
# --use_sdf
# --use_progressive_encoder