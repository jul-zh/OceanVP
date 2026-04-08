import numpy as np
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

train_loader, val_loader, test_loader = load_ocean_data(
    batch_size=4,
    val_batch_size=4,
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

loader = test_loader

def grad(x):
    dx = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dy = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    return dx.abs().mean() + dy.abs().mean()

vars_gt, vars_pred = [], []
grads_gt, grads_pred = [], []

with torch.no_grad():
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        vars_gt.append(y.var().item())
        vars_pred.append(pred.var().item())

        grads_gt.append(grad(y).item())
        grads_pred.append(grad(pred).item())

        if i >= 20:
            break

print("GT variance:", np.mean(vars_gt))
print("Pred variance:", np.mean(vars_pred))
print("VAR ratio pred/gt:", np.mean(vars_pred) / np.mean(vars_gt))

print("GT gradient:", np.mean(grads_gt))
print("Pred gradient:", np.mean(grads_pred))
print("GRAD ratio pred/gt:", np.mean(grads_pred) / np.mean(grads_gt))
