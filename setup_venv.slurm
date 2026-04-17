#!/bin/bash
#SBATCH --job-name=oceanvp_setup
#SBATCH --partition=intel-a100-pci5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/home/yzhidkova/projects/OceanVP/logs/%x-%j.out
#SBATCH --error=/home/yzhidkova/projects/OceanVP/logs/%x-%j.err

set -e
mkdir -p /home/yzhidkova/projects/OceanVP/logs

docker-id \
  -v /home/yzhidkova:/home/yzhidkova \
  -w /home/yzhidkova/projects/OceanVP \
  nvcr.io/nvidia/pytorch:25.02-py3 \
  bash /home/yzhidkova/projects/OceanVP/setup_venv.sh
