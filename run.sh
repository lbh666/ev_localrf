#!/bin/bash

OUTDIR='/mnt/share/lbh/'
DATADIR='/mnt/datasets'

mkdir -p /run/determined/workdir/.cache/torch/hub/checkpoints
cp ${OUTDIR}/ckpts/alexnet-owt-7be5be79.pth /run/determined/workdir/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard imageio easydict \
 matplotlib scipy plyfile joblib timm -i https://pypi.mirrors.ustc.edu.cn/simple


python localTensoRF/ev_train.py --datadir  ${OUTDIR}/TNT/Church_3000_upsampled_x2_2_fps/ \
 --fov 71 --loss_flow_weight_inital 1.0 --loss_depth_weight_inital 0.1 \
 --eventdir ${OUTDIR}/TNT/Church_3000_upsampled_x2_2_fps/acc_events \
 --add_frames_every 400 --events_in_imgs 2 --batch_size 4096 \
 --logdir ${OUTDIR}/localrf_results/final_v0_speed_up \
 --loss_warp_weight 0.5
#   --n_iters_per_frame 1000 --upsamp_list 100 200 300 400 500 \
#   --update_AlphaMask_list 200 400 600