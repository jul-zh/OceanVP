import math
import torch

from lib.models import MY_OCEAN_BASELINE
from lib.datasets.dataloader_ocean import load_data as load_ocean_data

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

ckpt = "/home/yzhidkova/projects/OceanVP/work_dirs/my_baseline_oceanvp_50ep/checkpoint.pth"

model = MY_OCEAN_BASELINE(
    in_shape=[16, 1, 32, 64],
    hid_S=32,
    aft_seq_length=16
).to(device)

state = torch.load(ckpt, map_location=device)
model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
model.eval()
print("model loaded")

_, _, test_loader = load_ocean_data(
    batch_size=8,
    val_batch_size=8,
    data_root="./data",
    num_workers=1,
    data_split="32_64",
    data_name="ocean_t0",
    train_time=["1994", "2013"],
    val_time=["2014", "2014"],
    test_time=["2015", "2015"],
    idx_in=list(range(-15, 1)),
    idx_out=list(range(1, 17)),
    step=1,
    level=0,
    distributed=False,
    use_augment=False,
    use_prefetcher=False,
    drop_last=False,
    temp_stride=2,
)

mean = torch.tensor(test_loader.dataset.mean, dtype=torch.float32, device=device)
std = torch.tensor(test_loader.dataset.std, dtype=torch.float32, device=device)

step_map = {
    1: 0,
    2: 1,
    4: 3,
}

sq_err = {k: 0.0 for k in step_map}
abs_err = {k: 0.0 for k in step_map}
count = {k: 0 for k in step_map}

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        pred = pred * std + mean
        y = y * std + mean

        for step_ts, idx in step_map.items():
            pred_s = pred[:, idx]
            y_s = y[:, idx]

            diff = pred_s - y_s
            sq_err[step_ts] += (diff ** 2).sum().item()
            abs_err[step_ts] += diff.abs().sum().item()
            count[step_ts] += diff.numel()

print("\n=== STEP-WISE RESULTS (DENORMALIZED) ===")
print(f"{'Step(ts)':>8} | {'RMSE':>10} | {'MAE':>10}")
print("-" * 34)

for step_ts in [1, 2, 4]:
    rmse = math.sqrt(sq_err[step_ts] / count[step_ts])
    mae = abs_err[step_ts] / count[step_ts]
    print(f"{step_ts:>8} | {rmse:>10.6f} | {mae:>10.6f}")
