#!/bin/bash
set -e

cd /home/yzhidkova/projects/OceanVP
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:$PYTHONPATH

echo "=== TRAIN RBFKAN PROB START ==="
hostname
pwd
python --version
python -c "from lib.methods import method_maps; print(method_maps.keys())"

python tools/train.py \
  -d ocean_t0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_RBFKAN_PROB.py \
  --ex_name my_baseline_rbfkan_prob_oceanvp_10ep \
  --temp_stride 2 \
  --epoch 10
