import argparse
import json
from pathlib import Path

import numpy as np


def estimate_binned_sde(arr: np.ndarray, n_bins: int = 100, sample_step_t: int = 1, sample_step_xy: int = 1):
    if arr.ndim == 3:
        arr = arr[None, ...]
    assert arr.ndim == 4, f"Expected [N,T,H,W] or [T,H,W], got {arr.shape}"

    xs = []
    dxs = []

    N, T, H, W = arr.shape
    for n in range(N):
        for t in range(0, T - 1, sample_step_t):
            x_t = arr[n, t]
            x_next = arr[n, t + 1]

            x_flat = x_t[::sample_step_xy, ::sample_step_xy].reshape(-1)
            dx_flat = (x_next - x_t)[::sample_step_xy, ::sample_step_xy].reshape(-1)

            mask = np.isfinite(x_flat) & np.isfinite(dx_flat)
            if mask.sum() == 0:
                continue

            xs.append(x_flat[mask])
            dxs.append(dx_flat[mask])

    if not xs:
        raise RuntimeError("No valid samples collected.")

    x = np.concatenate(xs, axis=0)
    dx = np.concatenate(dxs, axis=0)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, quantiles)

    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12

    bin_ids = np.digitize(x, edges[1:-1], right=False)

    centers = []
    counts = []
    drift = []
    diffusion2 = []

    for b in range(n_bins):
        mask = bin_ids == b
        xb = x[mask]
        dxb = dx[mask]

        if xb.size == 0:
            centers.append(float((edges[b] + edges[b + 1]) * 0.5))
            counts.append(0)
            drift.append(None)
            diffusion2.append(None)
            continue

        centers.append(float(np.mean(xb)))
        counts.append(int(xb.size))
        drift.append(float(np.mean(dxb)))
        diffusion2.append(float(np.mean(dxb ** 2)))

    return {
        "n_bins": int(n_bins),
        "edges": [float(v) for v in edges.tolist()],
        "centers": centers,
        "counts": counts,
        "drift": drift,
        "diffusion2": diffusion2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_bins", type=int, default=100)
    parser.add_argument("--sample_step_t", type=int, default=1)
    parser.add_argument("--sample_step_xy", type=int, default=1)
    args = parser.parse_args()

    arr = np.load(args.input)
    result = estimate_binned_sde(
        arr,
        n_bins=args.n_bins,
        sample_step_t=args.sample_step_t,
        sample_step_xy=args.sample_step_xy,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Saved to {out_path}")
    print(f"Bins: {args.n_bins}")


if __name__ == "__main__":
    main()
