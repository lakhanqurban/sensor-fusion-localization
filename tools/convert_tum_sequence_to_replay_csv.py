"""
convert_tum_sequence_to_replay_csv.py

Convert a full extracted TUM RGB-D sequence directory into replay CSV schema.

Expected sequence directory contents:
- groundtruth.txt  (timestamp tx ty tz qx qy qz qw)
- rgb.txt          (timestamp rgb/<file>.png)
- accelerometer.txt (optional)

This converter synchronizes frames to RGB timestamps, associates nearest ground
truth pose, and writes replay CSV rows for this project's benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


REPLAY_HEADER = (
    "timestamp_s,dt_s,accel_mps2,yaw_rate_rps,"
    "gt_x_m,gt_y_m,gt_yaw_rad,gt_speed_mps,"
    "gnss_x_m,gnss_y_m,camera_x_m,camera_y_m,camera_confidence"
)


def _read_numeric_rows(path: Path, min_cols: int) -> np.ndarray:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < min_cols:
            continue
        values.append([float(parts[i]) for i in range(min_cols)])

    if not values:
        raise ValueError(f"No valid numeric rows parsed from {path}")

    return np.array(values, dtype=float)


def _read_rgb_timestamps(rgb_txt: Path) -> np.ndarray:
    ts = []
    for line in rgb_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        ts.append(float(parts[0]))

    if not ts:
        raise ValueError(f"No RGB timestamps found in {rgb_txt}")

    return np.array(ts, dtype=float)


def _read_timestamps_and_files(list_path: Path) -> tuple[np.ndarray, list[str]]:
    timestamps = []
    files: list[str] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        timestamps.append(float(parts[0]))
        files.append(parts[1])

    if not timestamps:
        raise ValueError(f"No timestamp/file rows found in {list_path}")

    return np.array(timestamps, dtype=float), files


def _yaw_from_quaternion(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray) -> np.ndarray:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _nearest_indices(src_t: np.ndarray, query_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(src_t, query_t)
    idx_right = np.clip(idx, 0, len(src_t) - 1)
    idx_left = np.clip(idx - 1, 0, len(src_t) - 1)

    diff_left = np.abs(query_t - src_t[idx_left])
    diff_right = np.abs(query_t - src_t[idx_right])
    choose_left = diff_left <= diff_right
    nearest = np.where(choose_left, idx_left, idx_right)
    min_diff = np.where(choose_left, diff_left, diff_right)
    return nearest, min_diff


def _compute_vo_camera_xy(
    sequence_dir: Path,
    rgb_t: np.ndarray,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    min_pnp_points: int,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute camera XY from RGB-D visual odometry over RGB timestamps.

    Returns:
    - cam_xy: N x 2 camera trajectory aligned to GT start pose/scale
    - confidence: N confidence scores in [0,1]
    """

    rgb_ts_all, rgb_files_all = _read_timestamps_and_files(sequence_dir / "rgb.txt")
    depth_ts_all, depth_files_all = _read_timestamps_and_files(sequence_dir / "depth.txt")

    # Associate each selected RGB timestamp to the nearest RGB list row.
    rgb_idx, rgb_dt = _nearest_indices(rgb_ts_all, rgb_t)
    if np.max(rgb_dt) > 0.03:
        raise ValueError("Some requested RGB timestamps could not be associated in rgb.txt")

    # Associate selected RGB timestamps to depth timestamps.
    depth_idx, depth_dt = _nearest_indices(depth_ts_all, rgb_t)

    # TUM fr1 camera intrinsics.
    fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    orb = cv2.ORB_create(1500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # World-to-camera pose accumulation.
    R_cw = np.eye(3, dtype=np.float64)
    t_cw = np.zeros((3, 1), dtype=np.float64)

    cam_xy = np.zeros((len(rgb_t), 2), dtype=float)
    confidence = np.zeros(len(rgb_t), dtype=float)

    prev_img = None
    prev_depth = None
    prev_kps = None
    prev_desc = None

    for i in range(len(rgb_t)):
        rgb_file = sequence_dir / rgb_files_all[rgb_idx[i]]
        depth_file = sequence_dir / depth_files_all[depth_idx[i]]

        img = cv2.imread(str(rgb_file), cv2.IMREAD_GRAYSCALE)
        depth_raw = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        if img is None or depth_raw is None:
            if i > 0:
                cam_xy[i] = cam_xy[i - 1]
                confidence[i] = 0.0
            continue

        # TUM depth values are uint16 with scale factor 5000.
        depth_m = depth_raw.astype(np.float32) / 5000.0

        kps, desc = orb.detectAndCompute(img, None)
        if desc is None or len(kps) == 0:
            if i > 0:
                cam_xy[i] = cam_xy[i - 1]
                confidence[i] = 0.0
            prev_img, prev_depth, prev_kps, prev_desc = img, depth_m, kps, desc
            continue

        if i == 0 or prev_desc is None or prev_kps is None or prev_depth is None or len(prev_kps) == 0:
            cam_xy[i] = np.array([0.0, 0.0], dtype=float)
            confidence[i] = 1.0
            prev_img, prev_depth, prev_kps, prev_desc = img, depth_m, kps, desc
            continue

        matches = matcher.match(prev_desc, desc)
        matches = sorted(matches, key=lambda m: m.distance)[:350]

        object_points = []
        image_points = []
        for m in matches:
            u_prev, v_prev = prev_kps[m.queryIdx].pt
            u_curr, v_curr = kps[m.trainIdx].pt

            u_i = int(round(u_prev))
            v_i = int(round(v_prev))
            if u_i < 0 or v_i < 0 or v_i >= prev_depth.shape[0] or u_i >= prev_depth.shape[1]:
                continue

            z = float(prev_depth[v_i, u_i])
            if z <= 0.05 or z > max_depth_m:
                continue

            x = (u_prev - cx) * z / fx
            y = (v_prev - cy) * z / fy
            object_points.append([x, y, z])
            image_points.append([u_curr, v_curr])

        if len(object_points) >= min_pnp_points:
            obj = np.asarray(object_points, dtype=np.float64)
            imgp = np.asarray(image_points, dtype=np.float64)

            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj,
                imgp,
                K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=3.0,
                confidence=0.995,
                iterationsCount=120,
            )

            if ok:
                R_rel, _ = cv2.Rodrigues(rvec)
                t_rel = tvec.reshape(3, 1)

                # Compose world-to-camera: T_cw_k = T_rel * T_cw_{k-1}
                R_cw = R_rel @ R_cw
                t_cw = R_rel @ t_cw + t_rel

                inlier_ratio = 0.0 if inliers is None else float(len(inliers)) / max(1, len(object_points))
                confidence[i] = float(np.clip(inlier_ratio, 0.0, 1.0))
            else:
                confidence[i] = 0.0

        else:
            confidence[i] = 0.0

        # Camera center in world coordinates.
        cam_center_w = -R_cw.T @ t_cw
        cam_xy[i, 0] = float(cam_center_w[0, 0])
        cam_xy[i, 1] = float(cam_center_w[1, 0])

        if i > 0 and confidence[i] <= 1e-6:
            cam_xy[i] = cam_xy[i - 1]

        prev_img, prev_depth, prev_kps, prev_desc = img, depth_m, kps, desc

    # Align VO trajectory to GT in 2D with similarity transform from first motion segment.
    gt_xy = np.column_stack([gt_x, gt_y])
    vo_xy = cam_xy.copy()

    if len(vo_xy) >= 2:
        vo_d = vo_xy[1] - vo_xy[0]
        gt_d = gt_xy[1] - gt_xy[0]

        vo_norm = float(np.linalg.norm(vo_d))
        gt_norm = float(np.linalg.norm(gt_d))
        if vo_norm > 1e-6 and gt_norm > 1e-6:
            scale = gt_norm / vo_norm
            vo_angle = float(np.arctan2(vo_d[1], vo_d[0]))
            gt_angle = float(np.arctan2(gt_d[1], gt_d[0]))
            dtheta = gt_angle - vo_angle

            c = np.cos(dtheta)
            s = np.sin(dtheta)
            R2 = np.array([[c, -s], [s, c]], dtype=float)

            vo_xy = (scale * (R2 @ (vo_xy - vo_xy[0]).T)).T + gt_xy[0]
        else:
            vo_xy = vo_xy - vo_xy[0] + gt_xy[0]

    return vo_xy, confidence


def convert_sequence_dir(
    sequence_dir: Path,
    output_csv: Path,
    max_gt_dt_s: float,
    gnss_noise_std_m: float,
    camera_noise_std_m: float,
    camera_confidence: float,
    camera_source: str,
    min_pnp_points: int,
    max_depth_m: float,
    seed: int,
) -> None:
    gt_path = sequence_dir / "groundtruth.txt"
    rgb_path = sequence_dir / "rgb.txt"

    if not gt_path.exists() or not rgb_path.exists():
        raise FileNotFoundError("Sequence directory must contain groundtruth.txt and rgb.txt")

    gt = _read_numeric_rows(gt_path, min_cols=8)
    rgb_t = _read_rgb_timestamps(rgb_path)

    gt_t = gt[:, 0]
    gt_x = gt[:, 1]
    gt_y = gt[:, 2]
    gt_qx = gt[:, 4]
    gt_qy = gt[:, 5]
    gt_qz = gt[:, 6]
    gt_qw = gt[:, 7]
    gt_yaw = _yaw_from_quaternion(gt_qx, gt_qy, gt_qz, gt_qw)

    nearest_gt_idx, gt_dt = _nearest_indices(gt_t, rgb_t)
    valid = gt_dt <= max_gt_dt_s
    if not np.any(valid):
        raise ValueError("No RGB frames could be associated with ground truth under max_gt_dt_s")

    t = rgb_t[valid]
    x = gt_x[nearest_gt_idx[valid]]
    y = gt_y[nearest_gt_idx[valid]]
    yaw = gt_yaw[nearest_gt_idx[valid]]

    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    dt = np.clip(dt, 1e-3, None)

    vx = np.gradient(x, t, edge_order=1)
    vy = np.gradient(y, t, edge_order=1)
    speed = np.hypot(vx, vy)
    accel = np.gradient(speed, t, edge_order=1)
    yaw_rate = np.gradient(_wrap_pi(yaw), t, edge_order=1)

    rng = np.random.default_rng(seed)

    gnss_x = x + rng.normal(0.0, gnss_noise_std_m, size=len(x))
    gnss_y = y + rng.normal(0.0, gnss_noise_std_m, size=len(y))

    if camera_source == "vo":
        cam_xy, cam_conf = _compute_vo_camera_xy(
            sequence_dir=sequence_dir,
            rgb_t=t,
            gt_x=x,
            gt_y=y,
            min_pnp_points=min_pnp_points,
            max_depth_m=max_depth_m,
        )
        cam_x = cam_xy[:, 0]
        cam_y = cam_xy[:, 1]
    else:
        cam_x = x + rng.normal(0.0, camera_noise_std_m, size=len(x))
        cam_y = y + rng.normal(0.0, camera_noise_std_m, size=len(y))
        cam_conf = np.full(len(x), float(np.clip(camera_confidence, 0.0, 1.0)), dtype=float)

    out = np.column_stack([
        t,
        dt,
        accel,
        yaw_rate,
        x,
        y,
        yaw,
        speed,
        gnss_x,
        gnss_y,
        cam_x,
        cam_y,
        cam_conf,
    ])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_csv, out, delimiter=",", header=REPLAY_HEADER, comments="", fmt="%.9f")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert extracted TUM sequence directory to replay CSV")
    parser.add_argument("--sequence-dir", required=True, help="Path to extracted TUM sequence directory")
    parser.add_argument("--output", required=True, help="Output replay CSV path")
    parser.add_argument("--max-gt-dt", type=float, default=0.03, help="Max allowed RGB-groundtruth timestamp gap [s]")
    parser.add_argument("--gnss-noise-std", type=float, default=0.8, help="Synthetic GNSS noise std [m]")
    parser.add_argument("--camera-noise-std", type=float, default=0.35, help="Synthetic camera position noise std [m]")
    parser.add_argument("--camera-confidence", type=float, default=0.8, help="Camera confidence in [0,1]")
    parser.add_argument("--camera-source", choices=["vo", "synthetic"], default="vo", help="How to populate camera_x/y")
    parser.add_argument("--min-pnp-points", type=int, default=16, help="Minimum valid RGB-D correspondences for PnP")
    parser.add_argument("--max-depth-m", type=float, default=4.0, help="Max depth used for RGB-D feature backprojection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    sequence_dir = Path(args.sequence_dir)
    output_csv = Path(args.output)

    if not sequence_dir.exists():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")

    convert_sequence_dir(
        sequence_dir=sequence_dir,
        output_csv=output_csv,
        max_gt_dt_s=float(args.max_gt_dt),
        gnss_noise_std_m=float(args.gnss_noise_std),
        camera_noise_std_m=float(args.camera_noise_std),
        camera_confidence=float(args.camera_confidence),
        camera_source=str(args.camera_source),
        min_pnp_points=int(args.min_pnp_points),
        max_depth_m=float(args.max_depth_m),
        seed=int(args.seed),
    )
    print(f"Saved replay CSV to: {output_csv}")


if __name__ == "__main__":
    main()
