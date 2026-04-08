#!/bin/bash

cd /home/yzhidkova/projects/OceanVP
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:$PYTHONPATH

echo "=== TRAIN SDELIGHT 50ep START ==="
hostname
pwd
python --version

python tools/train.py \
    -d ocean_t0_32_64 \
    -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT.py \
    --ex_name my_baseline_sdelight_oceanvp_50ep \
    --temp_stride 2 \
    --epoch 50
