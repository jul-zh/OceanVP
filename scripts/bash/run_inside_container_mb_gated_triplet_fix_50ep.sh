#!/bin/bash
set -euo pipefail

cd /home/yzhidkova/projects/OceanVP
export PYTHONNOUSERSITE=1
source /home/yzhidkova/oceanvp-venv/bin/activate
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:${PYTHONPATH:-}

LOG_DIR=/home/yzhidkova/logs
mkdir -p "${LOG_DIR}"

JOB_ID="${OUTER_JOB_ID:-manual}"

echo "=== inside container ==="
hostname
pwd
which python
python --version
python -c "import numpy; print(numpy.__version__)"
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "from lib.methods import method_maps; print(method_maps.keys())"

echo "=== run 1: gatedkan_many_c48_g15 ==="
python tools/train.py \
  -d ocean_t0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_GATEDKAN_MANY_C48_G15_50ep.py \
  --ex_name mb_sdelight_kanhead_rbf_residual_gatedkan_many_c48_g15_fix_50ep \
  --temp_stride 2 \
  --epoch 50 \
  > "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_gatedkan_many_c48_g15_fix_50ep-${JOB_ID}.out" \
  2> "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_gatedkan_many_c48_g15_fix_50ep-${JOB_ID}.err"

echo "=== run 2: gatedkan_many_c48_g10 ==="
python tools/train.py \
  -d ocean_t0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_GATEDKAN_MANY_C48_G10_50ep.py \
  --ex_name mb_sdelight_kanhead_rbf_residual_gatedkan_many_c48_g10_fix_50ep \
  --temp_stride 2 \
  --epoch 50 \
  > "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_gatedkan_many_c48_g10_fix_50ep-${JOB_ID}.out" \
  2> "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_gatedkan_many_c48_g10_fix_50ep-${JOB_ID}.err"

echo "=== run 3: gatedkan_many_c64_g10_ldrop ==="
python tools/train.py \
  -d ocean_t0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_GATEDKAN_MANY_C64_G10_LDROP_50ep.py \
  --ex_name mb_sdelight_kanhead_rbf_residual_gatedkan_many_c64_g10_ldrop_fix_50ep \
  --temp_stride 2 \
  --epoch 50 \
  > "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_gatedkan_many_c64_g10_ldrop_fix_50ep-${JOB_ID}.out" \
  2> "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_gatedkan_many_c64_g10_ldrop_fix_50ep-${JOB_ID}.err"

echo "=== all 3 runs finished ==="
