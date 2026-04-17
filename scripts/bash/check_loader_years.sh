#!/bin/bash
set -ex

cd /home/yzhidkova/projects/OceanVP
export PYTHONNOUSERSITE=1
source /home/yzhidkova/oceanvp-venv/bin/activate
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:$PYTHONPATH

python /home/yzhidkova/projects/OceanVP/scripts/check_loader_years.py
