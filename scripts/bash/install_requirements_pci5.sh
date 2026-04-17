#!/bin/bash
set -e

PROJECT_DIR=/home/yzhidkova/projects/OceanVP
REQ_FILE=$PROJECT_DIR/requirements_pci5.txt

cd $PROJECT_DIR

echo "=== INSTALL PCI5 REQUIREMENTS START ==="
python --version
echo "Using requirements file: $REQ_FILE"

pip install --user -r $REQ_FILE

echo "=== VERIFY IMPORTS ==="
python - << 'PY'
mods = [
    "fvcore",
    "timm",
    "einops",
    "future",
    "hickle",
    "lpips",
    "netCDF4",
    "cv2",
    "xarray",
]
for m in mods:
    __import__(m)
    print("OK:", m)
print("=== ALL GOOD ===")
PY
