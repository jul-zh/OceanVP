#!/bin/bash
set -e

echo "=== INSTALL REQUIREMENTS START ==="

PROJECT_DIR=/home/yzhidkova/projects/OceanVP
REQ_FILE=$PROJECT_DIR/requirements.txt

cd $PROJECT_DIR

echo "Python version:"
python --version

echo "Using requirements file:"
echo $REQ_FILE

if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: requirements.txt not found!"
    exit 1
fi

echo "=== Installing packages (user mode) ==="
pip install --user -r $REQ_FILE

echo "=== Verifying installation ==="
python - << 'PY'
import sys
print("Python path:", sys.path)

try:
    import fvcore
    import timm
    import einops
    print("✅ Core packages OK")
except Exception as e:
    print("❌ Import error:", e)
    exit(1)

print("=== ALL GOOD ===")
PY

echo "=== INSTALL DONE ==="
