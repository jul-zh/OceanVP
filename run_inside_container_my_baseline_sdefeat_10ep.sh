#!/bin/bash

cd /home/yzhidkova/projects/OceanVP
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:$PYTHONPATH

echo "=== TRAIN SDEFEAT START ==="
hostname
pwd
python --version

python tools/train.py \
    -d ocean_t0_32_64 \
    -c configs/ocean/t0_32_64/MY_BASELINE_SDEFEAT.py \
    --ex_name my_baseline_sdefeat_oceanvp_10ep \
    --temp_stride 2 \
    --epoch 10
