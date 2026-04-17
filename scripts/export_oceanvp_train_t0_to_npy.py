import json
from pathlib import Path

import numpy as np
import xarray as xr

DATA_ROOT = "/home/yzhidkova/datasets/oceanvp_raw/OceanVP_HYCOM_32_64"
VAR_DIR = "water_temp_depth_0m"

TRAIN_TIME = ["1994", "2013"]
STEP = 1

OUT_DIR = Path("/home/yzhidkova/projects/OceanVP/data/ocean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPY = OUT_DIR / "ocean_t0_train_1994_2013_norm.npy"
OUT_STATS = OUT_DIR / "ocean_t0_train_1994_2013_stats.json"


def pick_var(ds):
    candidates = list(ds.data_vars)
    if not candidates:
        raise RuntimeError("No data_vars found in dataset")
    if len(candidates) == 1:
        return candidates[0]
    for name in candidates:
        if "temp" in name.lower():
            return name
    return candidates[0]


def main():
    pattern = f"{DATA_ROOT}/{VAR_DIR}/{VAR_DIR}_*.nc"
    ds = xr.open_mfdataset(pattern, combine="by_coords")

    ds = ds.sel(time=slice(*TRAIN_TIME))
    ds = ds.isel(time=slice(None, -1, STEP))

    var_name = pick_var(ds)
    data = ds[var_name].values.astype(np.float32)

    # squeeze depth dim if present and singleton
    if data.ndim == 4 and data.shape[1] == 1:
        data = data[:, 0, :, :]

    if data.ndim != 3:
        raise RuntimeError(f"Expected [T,H,W] after squeeze, got shape {data.shape} for var {var_name}")

    mean = data.mean(axis=(0, 1, 2), keepdims=False)
    std = data.std(axis=(0, 1, 2), keepdims=False)

    data_norm = (data - mean) / std
    np.save(OUT_NPY, data_norm)

    stats = {
        "input_pattern": pattern,
        "train_time": TRAIN_TIME,
        "step": STEP,
        "picked_var": var_name,
        "shape": list(data_norm.shape),
        "dtype": str(data_norm.dtype),
        "mean_scalar": float(mean),
        "std_scalar": float(std),
        "output_npy": str(OUT_NPY),
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
