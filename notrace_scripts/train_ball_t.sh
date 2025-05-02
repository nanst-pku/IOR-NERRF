CUDA_LAUNCH_BLOCKING=5 \
python train/train.py \
-n ball_orinerf \
-c NeRRFsngp.conf \
-D data/blender/transparent/ball \
--gpu_id=0 \
--visual_path tet_visual \
--stage 2 \
--tet_scale 4.2 \
--sphere_radius 2.30 \
--ior 1.29

