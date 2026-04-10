#!/bin/bash
set -e

cd /home/yzhidkova/projects/OceanVP
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:$PYTHONPATH

echo "=== TRAIN PROBKAN_V2 START ==="
hostname
pwd
python --version

python tools/train.py \
    -d ocean_t0_32_64 \
    -c configs/ocean/t0_32_64/PROBKAN_V2.py \
    --ex_name probkan_v2_oceanvp_10ep \
    --temp_stride 2 \
    --epoch 50
