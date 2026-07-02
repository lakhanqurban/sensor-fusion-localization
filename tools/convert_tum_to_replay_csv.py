"""
convert_tum_to_replay_csv.py

Convert TUM-style groundtruth trajectory text into replay CSV schema used by
Sensor-Fusion Localization benchmark.

Input format (TUM groundtruth):
  timestamp tx ty tz qx qy qz qw

Output CSV columns:
  timestamp_s,dt_s,accel_mps2,yaw_rate_rps,gt_x_m,gt_y_m,gt_yaw_rad,gt_speed_mps,
  gnss_x_m,gnss_y_m,camera_x_m,camera_y_m,camera_confidence

Notes:
- Controls are approximated by finite differences from trajectory.
- GNSS/camera fields are left as NaN placeholders for later sensor projection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _yaw_from_quaternion_z(qx: float, qy: float, qz: float, qw: float) -> float:
    # Standard yaw extraction from quaternion (Z-up convention).
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def convert_tum_groundtruth_to_replay_csv(input_path: Path, output_path: Path) -> None:
    try:
        data = np.loadtxt(input_path, comments="#")
    except ValueError as exc:
        raise ValueError(
            "Failed to parse input as TUM groundtruth numeric text. "
            "Expected rows like: timestamp tx ty tz qx qy qz qw"
        ) from exc
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 8:
        raise ValueError("Expected at least 8 columns per row: t tx ty tz qx qy qz qw")

    t = data[:, 0].astype(float)
    x = data[:, 1].astype(float)
    y = data[:, 2].astype(float)
    qx = data[:, 4].astype(float)
    qy = data[:, 5].astype(float)
    qz = data[:, 6].astype(float)
    qw = data[:, 7].astype(float)

    yaw = np.array([_yaw_from_quaternion_z(ix, iy, iz, iw) for ix, iy, iz, iw in zip(qx, qy, qz, qw)], dtype=float)

    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    dt = np.clip(dt, 1e-3, None)

    vx = np.gradient(x, t, edge_order=1)
    vy = np.gradient(y, t, edge_order=1)
    speed = np.hypot(vx, vy)

    accel = np.gradient(speed, t, edge_order=1)
    yaw_rate = np.gradient(_wrap_pi(yaw), t, edge_order=1)

    nan_col = np.full_like(t, np.nan, dtype=float)
    conf_col = np.full_like(t, np.nan, dtype=float)

    out = np.column_stack(
        [
            t,
            dt,
            accel,
            yaw_rate,
            x,
            y,
            yaw,
            speed,
            nan_col,
            nan_col,
            nan_col,
            nan_col,
            conf_col,
        ]
    )

    header = (
        "timestamp_s,dt_s,accel_mps2,yaw_rate_rps,"
        "gt_x_m,gt_y_m,gt_yaw_rad,gt_speed_mps,"
        "gnss_x_m,gnss_y_m,camera_x_m,camera_y_m,camera_confidence"
    )
    np.savetxt(output_path, out, delimiter=",", header=header, comments="", fmt="%.9f")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert TUM groundtruth to replay CSV schema")
    parser.add_argument("--input", required=True, help="Path to TUM groundtruth text file")
    parser.add_argument("--output", required=True, help="Path to output replay CSV")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    convert_tum_groundtruth_to_replay_csv(input_path, output_path)
    print(f"Saved replay CSV to: {output_path}")


if __name__ == "__main__":
    main()
