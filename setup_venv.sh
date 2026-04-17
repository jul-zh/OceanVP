#!/bin/bash
set -e

cd /home/yzhidkova/projects/OceanVP

echo "=== setup start ==="
hostname
pwd
python -V
nvidia-smi || true

export PYTHONNOUSERSITE=1

if [ ! -d /home/yzhidkova/oceanvp-venv ]; then
  python -m venv --system-site-packages /home/yzhidkova/oceanvp-venv
fi

source /home/yzhidkova/oceanvp-venv/bin/activate
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:$PYTHONPATH

python -m pip install --upgrade pip setuptools wheel

python -c "import torch; print('base torch:', torch.__version__)"

python -m pip install \
  "numpy<2" \
  "xarray==0.19.0" \
  netcdf4 dask decord future fvcore hickle lpips matplotlib nni \
  pandas scikit-learn tqdm yacs iopath six \
  "opencv-python-headless" \
  "timm==1.0.15" \
  "scikit-image>=0.21"

python -m pip install --force-reinstall "numpy<2"

python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import torch; print('torch:', torch.__version__)"
python -c "import decord, fvcore, hickle, lpips, nni, netCDF4, cv2, timm, xarray, skimage; print('SETUP_OK')"
