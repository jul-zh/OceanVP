#!/bin/bash
#SBATCH --job-name=check_imports_login
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
  nvcr.io/nvidia/pytorch:25.02-py3 \
  bash -lc '
    export PYTHONNOUSERSITE=1
    source /home/yzhidkova/oceanvp-venv/bin/activate
    export PYTHONPATH=/home/yzhidkova/projects/OceanVP:$PYTHONPATH
    python -c "from lib.methods import method_maps; print(method_maps.keys())"
  '
