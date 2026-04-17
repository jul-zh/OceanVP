#!/bin/bash
set -e

cd /home/yzhidkova/projects/OceanVP

export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:$PYTHONPATH

echo "=== TRAIN KANATTN START ==="
hostname
pwd
python --version

python tools/train.py \
    -d ocean_t0_32_64 \
    -c configs/ocean/t0_32_64/KANATTN.py \
    --ex_name kanattn_oceanvp_10ep \
    --temp_stride 2 \
    --epoch 10
