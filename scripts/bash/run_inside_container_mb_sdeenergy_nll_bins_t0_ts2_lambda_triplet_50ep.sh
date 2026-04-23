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

echo "=== run 1: lambda_sde=1e-5 ==="
python tools/train.py \
  -d ocean_t0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDEENERGY_NLL_BINS_L1e5_50ep.py \
  --ex_name mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_l1e5_fix_50ep \
  --temp_stride 2 \
  --epoch 50 \
  > "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_l1e5_fix_50ep-${JOB_ID}.out" \
  2> "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_l1e5_fix_50ep-${JOB_ID}.err"

echo "=== run 2: lambda_sde=2e-5 ==="
python tools/train.py \
  -d ocean_t0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDEENERGY_NLL_BINS_L2e5_50ep.py \
  --ex_name mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_l2e5_fix_50ep \
  --temp_stride 2 \
  --epoch 50 \
  > "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_l2e5_fix_50ep-${JOB_ID}.out" \
  2> "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_l2e5_fix_50ep-${JOB_ID}.err"

echo "=== run 3: lambda_sde=1e-4 ==="
python tools/train.py \
  -d ocean_t0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDEENERGY_NLL_BINS_L1e4_50ep.py \
  --ex_name mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_l1e4_fix_50ep \
  --temp_stride 2 \
  --epoch 50 \
  > "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_l1e4_fix_50ep-${JOB_ID}.out" \
  2> "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_l1e4_fix_50ep-${JOB_ID}.err"

echo "=== all 3 runs finished ==="
