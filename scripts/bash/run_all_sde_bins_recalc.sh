#!/bin/bash
set -euo pipefail

cd /home/yzhidkova/projects/OceanVP
export PYTHONNOUSERSITE=1
source /home/yzhidkova/oceanvp-venv/bin/activate
export PYTHONPATH=/home/yzhidkova/projects/OceanVP:${PYTHONPATH:-}

echo "=== patch estimate_sde_bins.py to variance ==="
python - <<'PY'
from pathlib import Path
p = Path('/home/yzhidkova/projects/OceanVP/scripts/estimate_sde_bins.py')
txt = p.read_text()
old = """        centers.append(float(np.mean(xb)))
        counts.append(int(xb.size))
        drift.append(float(np.mean(dxb)))
        diffusion2.append(float(np.mean(dxb ** 2)))
"""
new = """        mu = float(np.mean(dxb))
        var = float(np.mean((dxb - mu) ** 2))

        centers.append(float(np.mean(xb)))
        counts.append(int(xb.size))
        drift.append(mu)
        diffusion2.append(var)
"""
if old in txt:
    txt = txt.replace(old, new)
    p.write_text(txt)
    print("patched", p)
else:
    print("pattern not found, maybe already patched:", p)
PY

echo
echo "=== recalc t0 bins ==="
python /home/yzhidkova/projects/OceanVP/scripts/estimate_sde_bins.py \
  --input /home/yzhidkova/projects/OceanVP/data/ocean/ocean_t0_train_1994_2013_norm.npy \
  --output /home/yzhidkova/projects/OceanVP/logs/sde_bins_t0_train_1000.json \
  --n_bins 1000 \
  --sample_step_t 1 \
  --sample_step_xy 1

echo
echo "=== export s0 train npy ==="
python /home/yzhidkova/projects/OceanVP/scripts/export_oceanvp_train_s0_to_npy.py

echo
echo "=== recalc s0 bins ==="
python /home/yzhidkova/projects/OceanVP/scripts/estimate_sde_bins.py \
  --input /home/yzhidkova/projects/OceanVP/data/ocean/ocean_s0_train_1994_2013_norm.npy \
  --output /home/yzhidkova/projects/OceanVP/logs/sde_bins_s0_train_1000.json \
  --n_bins 1000 \
  --sample_step_t 1 \
  --sample_step_xy 1

echo
echo "=== export u0/v0 train npy ==="
python /home/yzhidkova/projects/OceanVP/scripts/export_oceanvp_train_uv0_to_npy.py

echo
echo "=== recalc u0 bins ==="
python /home/yzhidkova/projects/OceanVP/scripts/estimate_sde_bins.py \
  --input /home/yzhidkova/projects/OceanVP/data/ocean/ocean_u0_train_1994_2013_norm.npy \
  --output /home/yzhidkova/projects/OceanVP/logs/sde_bins_u0_train_1000.json \
  --n_bins 1000 \
  --sample_step_t 1 \
  --sample_step_xy 1

echo
echo "=== recalc v0 bins ==="
python /home/yzhidkova/projects/OceanVP/scripts/estimate_sde_bins.py \
  --input /home/yzhidkova/projects/OceanVP/data/ocean/ocean_v0_train_1994_2013_norm.npy \
  --output /home/yzhidkova/projects/OceanVP/logs/sde_bins_v0_train_1000.json \
  --n_bins 1000 \
  --sample_step_t 1 \
  --sample_step_xy 1

echo
echo "=== done ==="
ls -lh /home/yzhidkova/projects/OceanVP/logs/sde_bins_*_train_1000.json
