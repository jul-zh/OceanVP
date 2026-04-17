#!/bin/bash
set -ex

cd /home/yzhidkova/projects/OceanVP
export PYTHONNOUSERSITE=1
source /home/yzhidkova/oceanvp-venv/bin/activate
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:$PYTHONPATH

python /home/yzhidkova/projects/OceanVP/scripts/export_oceanvp_train_t0_to_npy.py
python /home/yzhidkova/projects/OceanVP/scripts/check_npy_stats.py

python /home/yzhidkova/projects/OceanVP/scripts/estimate_sde_ab.py \
  --input /home/yzhidkova/projects/OceanVP/data/ocean/ocean_t0_train_1994_2013_norm.npy \
  --output /home/yzhidkova/projects/OceanVP/logs/sde_alpha_beta_t0_train.json \
  --sample_step_t 2 \
  --sample_step_xy 2

python /home/yzhidkova/projects/OceanVP/scripts/estimate_sde_bins.py \
  --input /home/yzhidkova/projects/OceanVP/data/ocean/ocean_t0_train_1994_2013_norm.npy \
  --output /home/yzhidkova/projects/OceanVP/logs/sde_bins_t0_train_100.json \
  --n_bins 100 \
  --sample_step_t 2 \
  --sample_step_xy 2

python /home/yzhidkova/projects/OceanVP/scripts/show_sde_coeffs.py
