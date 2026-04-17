#!/bin/bash
set -ex
cd /home/yzhidkova/projects/OceanVP
export PYTHONNOUSERSITE=1
source /home/yzhidkova/oceanvp-venv/bin/activate
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:$PYTHONPATH

python tools/train.py \
    -d ocean_t0_32_64 \
    -c configs/ocean/t0_32_64/MY_BASELINE_STRONGENC.py \
    --ex_name mb_strongenc_fix_50ep \
    --temp_stride 2 \
    --epoch 50
