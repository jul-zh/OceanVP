#!/bin/bash
#SBATCH --job-name=my_baseline_kanhead_rbf_wide_login_10ep
#SBATCH --partition=intel-a100-pci5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/home/yzhidkova/logs/%x-%j.out
#SBATCH --error=/home/yzhidkova/logs/%x-%j.err

set -e
mkdir -p /home/yzhidkova/logs

docker-id \
  -v /home/yzhidkova:/home/yzhidkova \
  -w /home/yzhidkova/projects/OceanVP \
  nvcr.io/nvidia/pytorch:25.02-py3 \
  bash /home/yzhidkova/projects/OceanVP/scripts/bash/run_inside_container_my_baseline_kanhead_rbf_wide_login_10ep.sh
