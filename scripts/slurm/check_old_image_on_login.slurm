#!/bin/bash
#SBATCH --job-name=check_old_image
#SBATCH --partition=intel-a100-pci5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/home/yzhidkova/logs/%x-%j.out
#SBATCH --error=/home/yzhidkova/logs/%x-%j.err

docker-id \
  -v /home/yzhidkova:/home/yzhidkova \
  -w /home/yzhidkova/projects/OceanVP \
  nvidia/pytorch:22.12-py3 \
  bash -lc '
    python --version
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
    nvidia-smi
  '
