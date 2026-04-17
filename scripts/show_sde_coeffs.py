import json
from pathlib import Path

ab_path = Path("/home/yzhidkova/projects/OceanVP/logs/sde_alpha_beta_t0_train.json")
bins_path = Path("/home/yzhidkova/projects/OceanVP/logs/sde_bins_t0_train_100.json")

if ab_path.exists():
    data = json.loads(ab_path.read_text(encoding="utf-8"))
    print("=== alpha/beta ===")
    for k, v in data.items():
        print(f"{k}: {v}")
    print()

if bins_path.exists():
    data = json.loads(bins_path.read_text(encoding="utf-8"))
    print("=== binned sde ===")
    print("n_bins:", data["n_bins"])
    print("first 10 centers:", data["centers"][:10])
    print("first 10 counts:", data["counts"][:10])
    print("first 10 drift:", data["drift"][:10])
    print("first 10 diffusion2:", data["diffusion2"][:10])
