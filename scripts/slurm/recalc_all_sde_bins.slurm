#!/bin/bash
#SBATCH --job-name=recalc_all_sde_bins
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/yzhidkova/logs/%x-%j.out
#SBATCH --error=/home/yzhidkova/logs/%x-%j.err

set -e
mkdir -p /home/yzhidkova/logs

bash /home/yzhidkova/projects/OceanVP/scripts/bash/run_all_sde_bins_recalc.sh
