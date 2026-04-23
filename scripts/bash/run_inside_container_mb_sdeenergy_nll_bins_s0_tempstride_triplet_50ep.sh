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

echo "=== run 1: sdeenergy_nll_bins s0 temp_stride=1 ==="
python tools/train.py \
  -d ocean_s0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDEENERGY_NLL_BINS_S0_50ep.py \
  --ex_name mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_s0_ts1_fix_50ep \
  --temp_stride 1 \
  --epoch 50 \
  > "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_s0_ts1_fix_50ep-${JOB_ID}.out" \
  2> "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_s0_ts1_fix_50ep-${JOB_ID}.err"

echo "=== run 2: sdeenergy_nll_bins s0 temp_stride=2 ==="
python tools/train.py \
  -d ocean_s0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDEENERGY_NLL_BINS_S0_50ep.py \
  --ex_name mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_s0_ts2_fix_50ep \
  --temp_stride 2 \
  --epoch 50 \
  > "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_s0_ts2_fix_50ep-${JOB_ID}.out" \
  2> "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_s0_ts2_fix_50ep-${JOB_ID}.err"

echo "=== run 3: sdeenergy_nll_bins s0 temp_stride=4 ==="
python tools/train.py \
  -d ocean_s0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDEENERGY_NLL_BINS_S0_50ep.py \
  --ex_name mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_s0_ts4_fix_50ep \
  --temp_stride 4 \
  --epoch 50 \
  > "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_s0_ts4_fix_50ep-${JOB_ID}.out" \
  2> "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_s0_ts4_fix_50ep-${JOB_ID}.err"

echo "=== all 3 runs finished ==="
