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

echo "=== train start (rbf residual dropout0 10ep) ==="
python tools/train.py \
    -d ocean_t0_32_64 \
    -c configs/ocean/t0_32_64/MY_BASELINE_KANHEAD_RBF_RESIDUAL_DROPOUT0.py \
    --ex_name my_baseline_kanhead_rbf_residual_dropout0_login_10ep \
    --temp_stride 2 \
    --epoch 10
