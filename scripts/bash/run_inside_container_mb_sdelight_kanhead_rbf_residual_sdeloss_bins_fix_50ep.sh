#!/bin/bash
set -ex

cd /home/yzhidkova/projects/OceanVP
export PYTHONNOUSERSITE=1
source /home/yzhidkova/oceanvp-venv/bin/activate
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:$PYTHONPATH

echo "=== inside container ==="
hostname
pwd
which python
python --version
python -c "import numpy; print(numpy.__version__)"
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"

echo "=== import lib.methods ==="
python -c "from lib.methods import method_maps; print(method_maps.keys())"

echo "=== train start (mb sdelight kanhead rbf residual sdeloss bins fix 50ep) ==="
python tools/train.py \
    -d ocean_t0_32_64 \
    -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS_BINS_50ep_WIDE.py \
    --ex_name mb_sdelight_kanhead_rbf_residual_sdeloss_bins_fix_50ep \
    --temp_stride 2 \
    --epoch 50
