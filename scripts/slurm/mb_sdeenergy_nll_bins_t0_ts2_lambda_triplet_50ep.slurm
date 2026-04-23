#!/bin/bash
#SBATCH --job-name=mb_sdeenergy_nll_bins_t0_ts2_lambda_triplet_50ep
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
  -e OUTER_JOB_ID=${SLURM_JOB_ID} \
  -v /home/yzhidkova:/home/yzhidkova \
  -w /home/yzhidkova/projects/OceanVP \
  nvcr.io/nvidia/pytorch:25.02-py3 \
  bash /home/yzhidkova/projects/OceanVP/scripts/bash/run_inside_container_mb_sdeenergy_nll_bins_t0_ts2_lambda_triplet_50ep.sh
