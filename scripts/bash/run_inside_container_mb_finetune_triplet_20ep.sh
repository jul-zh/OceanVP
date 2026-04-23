#!/bin/bash
set -euo pipefail

cd /home/yzhidkova/projects/OceanVP
export PYTHONNOUSERSITE=1
source /home/yzhidkova/oceanvp-venv/bin/activate
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:${PYTHONPATH:-}

LOG_DIR=/home/yzhidkova/logs
mkdir -p "${LOG_DIR}"

JOB_ID="${OUTER_JOB_ID:-manual}"

# === ВСТАВЬТЕ РЕАЛЬНЫЕ ПУТИ ===
CKPT_MANY="/home/yzhidkova/projects/OceanVP/work_dirs/mb_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fix_50ep/checkpoints/latest.pth"
CKPT_STEM="/home/yzhidkova/projects/OceanVP/work_dirs/mb_sdelight_kanhead_rbf_residual_stemplus_sdeloss_bins_fix_50ep/checkpoints/latest.pth"
CKPT_GATED="/home/yzhidkova/projects/OceanVP/work_dirs/mb_sdelight_kanhead_rbf_residual_gatedkan_many_c64_g10_ldrop_fix_50ep/checkpoints/latest.pth"

echo "=== inside container ==="
hostname
pwd
which python
python --version
python -c "import numpy; print(numpy.__version__)"
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"

echo "=== finetune 1: many_c48_g15 ==="
python tools/train.py \
  -d ocean_t0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS_BINS_MANY_C48_G15_FINETUNE_20ep.py \
  --ex_name mb_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_finetune20_fix \
  --temp_stride 2 \
  --epoch 20 \
  --resume_from "${CKPT_MANY}" \
  > "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_finetune20_fix-${JOB_ID}.out" \
  2> "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_finetune20_fix-${JOB_ID}.err"

echo "=== finetune 2: stemplus ==="
python tools/train.py \
  -d ocean_t0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_STEMPLUS_SDELOSS_BINS_FINETUNE_20ep.py \
  --ex_name mb_sdelight_kanhead_rbf_residual_stemplus_sdeloss_bins_finetune20_fix \
  --temp_stride 2 \
  --epoch 20 \
  --resume_from "${CKPT_STEM}" \
  > "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_stemplus_sdeloss_bins_finetune20_fix-${JOB_ID}.out" \
  2> "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_stemplus_sdeloss_bins_finetune20_fix-${JOB_ID}.err"

echo "=== finetune 3: gatedkan_many_c64_g10_ldrop ==="
python tools/train.py \
  -d ocean_t0_32_64 \
  -c configs/ocean/t0_32_64/MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_GATEDKAN_MANY_C64_G10_LDROP_FINETUNE_20ep.py \
  --ex_name mb_sdelight_kanhead_rbf_residual_gatedkan_many_c64_g10_ldrop_finetune20_fix \
  --temp_stride 2 \
  --epoch 20 \
  --resume_from "${CKPT_GATED}" \
  > "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_gatedkan_many_c64_g10_ldrop_finetune20_fix-${JOB_ID}.out" \
  2> "${LOG_DIR}/mb_sdelight_kanhead_rbf_residual_gatedkan_many_c64_g10_ldrop_finetune20_fix-${JOB_ID}.err"

echo "=== all 3 finetunes finished ==="
