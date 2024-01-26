#!/bin/bash

OUTDIR='/mnt/share/lbh/'
DATADIR='/mnt/datasets'

mkdir -p /run/determined/workdir/.cache/torch/hub/checkpoints
cp ${OUTDIR}/ckpts/alexnet-owt-7be5be79.pth /run/determined/workdir/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard imageio easydict \
 matplotlib scipy plyfile joblib timm -i https://pypi.mirrors.ustc.edu.cn/simple


python localTensoRF/ev_train.py --datadir  ${OUTDIR}/TNT/Church_3000_upsampled_x2_2_fps/ --logdir ${OUTDIR}/localrf_results/events_2fps_200add_output \
 --fov 71 --loss_flow_weight_inital 0 --eventdir ${OUTDIR}/TNT/Church_3000_upsampled_x2_2_fps/acc_events \
  --add_frames_every 200 --s3im_weight 0