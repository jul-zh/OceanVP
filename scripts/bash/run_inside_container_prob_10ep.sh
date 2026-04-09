#!/bin/bash

cd /home/yzhidkova/projects/OceanVP
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:$PYTHONPATH

echo "=== TRAIN PROB START ==="
hostname
pwd
python --version

python tools/train.py \
    -d ocean_t0_32_64 \
    -c configs/ocean/t0_32_64/PROB.py \
    --ex_name prob_oceanvp_10ep \
    --temp_stride 2 \
    --epoch 10
EOF
