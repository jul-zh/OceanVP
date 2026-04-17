import argparse
import json
from pathlib import Path

import numpy as np


def laplacian_2d(x: np.ndarray) -> np.ndarray:
    lap = np.zeros_like(x, dtype=np.float64)
    lap[1:-1, 1:-1] = (
        x[2:, 1:-1]
        + x[:-2, 1:-1]
        + x[1:-1, 2:]
        + x[1:-1, :-2]
        - 4.0 * x[1:-1, 1:-1]
    )
    return lap


def solve_alpha_beta_from_array(arr: np.ndarray, sample_step_t: int = 1, sample_step_xy: int = 1):
    if arr.ndim == 3:
        arr = arr[None, ...]
    assert arr.ndim == 4, f"Expected [N,T,H,W] or [T,H,W], got {arr.shape}"

    X_rows = []
    y_rows = []

    N, T, H, W = arr.shape
    for n in range(N):
        for t in range(1, T - 1, sample_step_t):
            x_prev = arr[n, t - 1]
            x_t = arr[n, t]
            x_next = arr[n, t + 1]

            drift = x_t - x_prev
            lap = laplacian_2d(x_t)
            delta = x_next - x_t

            drift_s = drift[::sample_step_xy, ::sample_step_xy].reshape(-1)
            lap_s = lap[::sample_step_xy, ::sample_step_xy].reshape(-1)
            delta_s = delta[::sample_step_xy, ::sample_step_xy].reshape(-1)

            mask = np.isfinite(drift_s) & np.isfinite(lap_s) & np.isfinite(delta_s)
            if mask.sum() == 0:
                continue

            A = np.stack([drift_s[mask], lap_s[mask]], axis=1)
            b = delta_s[mask]

            X_rows.append(A)
            y_rows.append(b)

    if not X_rows:
        raise RuntimeError("No valid samples collected.")

    X = np.concatenate(X_rows, axis=0)
    y = np.concatenate(y_rows, axis=0)

    coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coef.tolist()

    y_pred = X @ coef
    mse = float(np.mean((y - y_pred) ** 2))

    result = {
        "alpha": float(alpha),
        "beta": float(beta),
        "mse_fit": mse,
        "num_samples": int(X.shape[0]),
        "rank": int(rank),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sample_step_t", type=int, default=1)
    parser.add_argument("--sample_step_xy", type=int, default=1)
    args = parser.parse_args()

    arr = np.load(args.input)
    result = solve_alpha_beta_from_array(
        arr,
        sample_step_t=args.sample_step_t,
        sample_step_xy=args.sample_step_xy,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
