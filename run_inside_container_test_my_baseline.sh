#!/bin/bash

cd /home/yzhidkova/projects/OceanVP
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:$PYTHONPATH

echo "=== TEST START ==="
hostname
pwd
python --version

python tools/test.py \
    -d ocean_t0_32_64 \
    -c configs/ocean/t0_32_64/MY_BASELINE.py \
    --ex_name my_baseline_oceanvp_50ep \
    --temp_stride 2
