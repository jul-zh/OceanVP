import json
from pathlib import Path

import numpy as np
import xarray as xr

DATA_ROOT = "/home/yzhidkova/datasets/oceanvp_raw/OceanVP_HYCOM_32_64"
U_DIR = "water_u_depth_0m"
V_DIR = "water_v_depth_0m"

TRAIN_TIME = ["1994", "2013"]
STEP = 1

OUT_DIR = Path("/home/yzhidkova/projects/OceanVP/data/ocean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_U_NPY = OUT_DIR / "ocean_u0_train_1994_2013_norm.npy"
OUT_V_NPY = OUT_DIR / "ocean_v0_train_1994_2013_norm.npy"
OUT_STATS = OUT_DIR / "ocean_uv0_train_1994_2013_stats.json"


def pick_var(ds, kind: str):
    candidates = list(ds.data_vars)
    if not candidates:
        raise RuntimeError(f"No data_vars found for {kind}")

    if len(candidates) == 1:
        return candidates[0]

    for name in candidates:
        lname = name.lower()
        if kind == "u" and ("water_u" in lname or lname.endswith("_u") or lname == "u"):
            return name
        if kind == "v" and ("water_v" in lname or lname.endswith("_v") or lname == "v"):
            return name

    return candidates[0]


def squeeze_if_needed(data):
    if data.ndim == 4 and data.shape[1] == 1:
        data = data[:, 0, :, :]
    return data


def main():
    pattern_u = f"{DATA_ROOT}/{U_DIR}/{U_DIR}_*.nc"
    pattern_v = f"{DATA_ROOT}/{V_DIR}/{V_DIR}_*.nc"

    ds_u = xr.open_mfdataset(pattern_u, combine="by_coords")
    ds_v = xr.open_mfdataset(pattern_v, combine="by_coords")

    ds_u = ds_u.sel(time=slice(*TRAIN_TIME))
    ds_v = ds_v.sel(time=slice(*TRAIN_TIME))

    ds_u = ds_u.isel(time=slice(None, -1, STEP))
    ds_v = ds_v.isel(time=slice(None, -1, STEP))

    u_name = pick_var(ds_u, "u")
    v_name = pick_var(ds_v, "v")

    u = ds_u[u_name].values.astype(np.float32)
    v = ds_v[v_name].values.astype(np.float32)

    u = squeeze_if_needed(u)
    v = squeeze_if_needed(v)

    if u.ndim != 3 or v.ndim != 3:
        raise RuntimeError(f"Expected [T,H,W], got u={u.shape}, v={v.shape}")

    mean_u = u.mean(axis=(0, 1, 2), keepdims=False)
    std_u = u.std(axis=(0, 1, 2), keepdims=False)
    mean_v = v.mean(axis=(0, 1, 2), keepdims=False)
    std_v = v.std(axis=(0, 1, 2), keepdims=False)

    u_norm = (u - mean_u) / std_u
    v_norm = (v - mean_v) / std_v

    np.save(OUT_U_NPY, u_norm)
    np.save(OUT_V_NPY, v_norm)

    stats = {
        "input_pattern_u": pattern_u,
        "input_pattern_v": pattern_v,
        "train_time": TRAIN_TIME,
        "step": STEP,
        "u_var": u_name,
        "v_var": v_name,
        "u_shape": list(u_norm.shape),
        "v_shape": list(v_norm.shape),
        "dtype": str(u_norm.dtype),
        "mean_u": float(mean_u),
        "std_u": float(std_u),
        "mean_v": float(mean_v),
        "std_v": float(std_v),
        "output_u_npy": str(OUT_U_NPY),
        "output_v_npy": str(OUT_V_NPY),
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
