#!/bin/bash

cd /home/yzhidkova/projects/OceanVP
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:$PYTHONPATH

echo "=== EVAL STEPS START ==="
hostname
pwd
python --version

python evaluate_steps.py
