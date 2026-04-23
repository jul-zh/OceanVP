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

echo "=== test 1: many_centers ==="
python tools/test.py \
  -d ocean_t0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS_BINS_MANY_CENTERS_50ep.py \
  --ex_name mb_sdelight_kanhead_rbf_residual_sdeloss_bins_many_centers_fix_50ep \
  --temp_stride 2 \
  > "${LOG_DIR}/test_mb_sdelight_kanhead_rbf_residual_sdeloss_bins_many_centers_fix_50ep-${JOB_ID}.out" \
  2> "${LOG_DIR}/test_mb_sdelight_kanhead_rbf_residual_sdeloss_bins_many_centers_fix_50ep-${JOB_ID}.err"

echo "=== test 2: gamma2_c32 ==="
python tools/test.py \
  -d ocean_t0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS_BINS_GAMMA2_C32_50ep.py \
  --ex_name mb_sdelight_kanhead_rbf_residual_sdeloss_bins_gamma2_c32_fix_50ep \
  --temp_stride 2 \
  > "${LOG_DIR}/test_mb_sdelight_kanhead_rbf_residual_sdeloss_bins_gamma2_c32_fix_50ep-${JOB_ID}.out" \
  2> "${LOG_DIR}/test_mb_sdelight_kanhead_rbf_residual_sdeloss_bins_gamma2_c32_fix_50ep-${JOB_ID}.err"

echo "=== test 3: gamma15 ==="
python tools/test.py \
  -d ocean_t0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS_BINS_GAMMA15_50ep.py \
  --ex_name mb_sdelight_kanhead_rbf_residual_sdeloss_bins_gamma15_fix_50ep \
  --temp_stride 2 \
  > "${LOG_DIR}/test_mb_sdelight_kanhead_rbf_residual_sdeloss_bins_gamma15_fix_50ep-${JOB_ID}.out" \
  2> "${LOG_DIR}/test_mb_sdelight_kanhead_rbf_residual_sdeloss_bins_gamma15_fix_50ep-${JOB_ID}.err"

echo "=== all 3 tests finished ==="
