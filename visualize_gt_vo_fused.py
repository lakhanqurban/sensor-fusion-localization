"""
visualize_gt_vo_fused.py

Quick trajectory visualization for Sensor-Fusion Localization replay CSV.
Overlays:
- Ground truth trajectory
- VO-only camera trajectory (camera_x_m, camera_y_m)
- Fused observation trajectory (from gnss + camera + confidence)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _fuse_position(gnss_xy: np.ndarray, camera_xy: np.ndarray, confidence: np.ndarray, gnss_std: float, camera_std: float) -> np.ndarray:
    gnss_var = max(gnss_std, 1e-6) ** 2
    camera_var = max(camera_std, 1e-6) ** 2

    fused = np.zeros_like(gnss_xy)
    for i in range(len(gnss_xy)):
        gx, gy = gnss_xy[i]
        cx, cy = camera_xy[i]
        conf = float(np.clip(confidence[i], 0.0, 1.0))

        gnss_valid = not (np.isnan(gx) or np.isnan(gy))
        cam_valid = not (np.isnan(cx) or np.isnan(cy))

        if gnss_valid and not cam_valid:
            fused[i] = [gx, gy]
            continue
        if cam_valid and not gnss_valid:
            fused[i] = [cx, cy]
            continue
        if not gnss_valid and not cam_valid:
            fused[i] = fused[i - 1] if i > 0 else [0.0, 0.0]
            continue

        inv_var_g = 1.0 / gnss_var
        inv_var_c = conf / camera_var
        denom = inv_var_g + inv_var_c
        if denom <= 1e-12:
            fused[i] = [gx, gy]
        else:
            fused[i, 0] = (inv_var_g * gx + inv_var_c * cx) / denom
            fused[i, 1] = (inv_var_g * gy + inv_var_c * cy) / denom

    return fused


def _moving_average_xy(xy: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return xy.copy()

    w = int(max(1, window))
    kernel = np.ones(w, dtype=float) / float(w)
    x_smooth = np.convolve(xy[:, 0], kernel, mode="same")
    y_smooth = np.convolve(xy[:, 1], kernel, mode="same")
    return np.column_stack([x_smooth, y_smooth])


def _save_overlay_plot(
    output_path: Path,
    gt_xy: np.ndarray,
    vo_xy: np.ndarray,
    vo_xy_smooth: np.ndarray,
    fused_xy: np.ndarray,
    fused_xy_smooth: np.ndarray,
    with_raw: bool,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", linewidth=2.0, label="Ground Truth")

    if with_raw:
        ax.plot(vo_xy[:, 0], vo_xy[:, 1], color="tab:green", linewidth=1.2, alpha=0.8, label="VO-only (camera)")
        ax.scatter(fused_xy[:, 0], fused_xy[:, 1], s=8, color="tab:blue", alpha=0.18, label="Fused obs (raw)")
        ax.plot(fused_xy_smooth[:, 0], fused_xy_smooth[:, 1], color="tab:blue", linewidth=2.0, label="Fused trajectory (smoothed)")
    else:
        ax.plot(vo_xy_smooth[:, 0], vo_xy_smooth[:, 1], color="tab:green", linewidth=2.0, label="VO-only trajectory (smoothed)")
        ax.plot(fused_xy_smooth[:, 0], fused_xy_smooth[:, 1], color="tab:blue", linewidth=2.4, label="Fused trajectory (smoothed)")

    ax.scatter(gt_xy[0, 0], gt_xy[0, 1], color="tab:orange", s=60, label="Start")
    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)


def _load_csv(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError("CSV has no rows")
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlay GT vs VO-only vs fused trajectory from replay CSV")
    parser.add_argument("--dataset", required=True, help="Replay CSV path")
    parser.add_argument("--output", default="", help="Output image path (default: output/gt_vo_fused_overlay.png)")
    parser.add_argument("--portfolio-output", default="", help="Clean supplementary output path (default: output/gt_vo_fused_overlay_portfolio.png)")
    parser.add_argument("--gnss-std", type=float, default=1.0, help="GNSS std [m] for fusion")
    parser.add_argument("--camera-std", type=float, default=0.7, help="Camera std [m] for fusion")
    parser.add_argument("--smooth-window", type=int, default=21, help="Moving-average window for fused trajectory")
    parser.add_argument("--show", action="store_true", help="Show figure interactively")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if not args.show:
        matplotlib.use("Agg")

    csv_path = Path(args.dataset)
    data = _load_csv(csv_path)

    timestamps = np.asarray(data["timestamp_s"], dtype=float)
    sort_idx = np.argsort(timestamps)

    gt_xy = np.column_stack([data["gt_x_m"], data["gt_y_m"]]).astype(float)[sort_idx]
    vo_xy = np.column_stack([data["camera_x_m"], data["camera_y_m"]]).astype(float)[sort_idx]
    gnss_xy = np.column_stack([data["gnss_x_m"], data["gnss_y_m"]]).astype(float)[sort_idx]
    cam_conf = np.asarray(data["camera_confidence"], dtype=float)[sort_idx]

    fused_xy = _fuse_position(
        gnss_xy=gnss_xy,
        camera_xy=vo_xy,
        confidence=cam_conf,
        gnss_std=float(args.gnss_std),
        camera_std=float(args.camera_std),
    )
    fused_xy_smooth = _moving_average_xy(fused_xy, int(args.smooth_window))
    vo_xy_smooth = _moving_average_xy(vo_xy, int(args.smooth_window))

    output_path = Path(args.output) if args.output else csv_path.parent.parent / "output" / "gt_vo_fused_overlay.png"
    portfolio_output_path = Path(args.portfolio_output) if args.portfolio_output else csv_path.parent.parent / "output" / "gt_vo_fused_overlay_portfolio.png"

    _save_overlay_plot(
        output_path=output_path,
        gt_xy=gt_xy,
        vo_xy=vo_xy,
        vo_xy_smooth=vo_xy_smooth,
        fused_xy=fused_xy,
        fused_xy_smooth=fused_xy_smooth,
        with_raw=True,
        title="Trajectory Overlay: GT vs VO-only vs Fused",
    )

    _save_overlay_plot(
        output_path=portfolio_output_path,
        gt_xy=gt_xy,
        vo_xy=vo_xy,
        vo_xy_smooth=vo_xy_smooth,
        fused_xy=fused_xy,
        fused_xy_smooth=fused_xy_smooth,
        with_raw=False,
        title="Portfolio Overlay: GT vs VO-only vs Fused",
    )

    print(f"Saved overlay figure to: {output_path}")
    print(f"Saved portfolio figure to: {portfolio_output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
