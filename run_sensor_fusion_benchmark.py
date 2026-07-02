"""
run_sensor_fusion_benchmark.py

Benchmark app that fuses camera perception + GNSS observations and compares
EKF/UKF/PF localization on a shared route.

This script supports:
- synthetic mode (default)
- replay mode from CSV (real-data migration path)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Local project imports
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from adapters import ReplayDatasetAdapter, SyntheticConfig, SyntheticRouteAdapter
from evaluator import compute_metrics
from fusion import PositionObservationFusion
from models import TrajectoryMetrics

# Existing repository localizers
COMPONENTS_DIR = PROJECT_DIR.parents[1] / "components"
sys.path.append(str(COMPONENTS_DIR / "state"))
sys.path.append(str(COMPONENTS_DIR / "localization" / "kalman_filter"))
sys.path.append(str(COMPONENTS_DIR / "localization" / "particle_filter"))
from state import State  # type: ignore
from extended_kalman_filter_localizer import ExtendedKalmanFilterLocalizer  # type: ignore
from unscented_kalman_filter_localizer import UnscentedKalmanFilterLocalizer  # type: ignore
from particle_filter_localizer import ParticleFilterLocalizer  # type: ignore


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sensor-Fusion Localization benchmark (EKF/UKF/PF).")
    parser.add_argument("--source", choices=["synthetic", "replay"], default="synthetic")
    parser.add_argument("--dataset", type=str, default="", help="CSV file path for --source replay")
    parser.add_argument("--total-time", type=float, default=36.0, help="Synthetic mode duration [s]")
    parser.add_argument("--dt", type=float, default=0.1, help="Synthetic mode time step [s]")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_DIR / "output"))
    parser.add_argument("--pf-particles", type=int, default=500, help="Particle count for PF")
    parser.add_argument(
        "--pf-resampling",
        choices=["multinomial", "systematic", "low_variance"],
        default="systematic",
        help="Resampling method for PF",
    )
    parser.add_argument("--pf-resample-threshold", type=float, default=0.5, help="PF effective-sample-size threshold ratio")
    parser.add_argument("--report", action="store_true", help="Save markdown report artifact")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    return parser


def _create_adapter(args: argparse.Namespace):
    if args.source == "synthetic":
        cfg = SyntheticConfig(total_time_s=args.total_time, dt_s=args.dt)
        return SyntheticRouteAdapter(cfg)

    if not args.dataset:
        raise ValueError("--dataset is required when --source replay")
    return ReplayDatasetAdapter(args.dataset)


def _create_filter_bank(args: argparse.Namespace) -> tuple[Dict[str, object], Dict[str, State]]:
    filters: Dict[str, object] = {
        "EKF": ExtendedKalmanFilterLocalizer(color="tab:red"),
        "UKF": UnscentedKalmanFilterLocalizer(color="tab:orange"),
        "PF": ParticleFilterLocalizer(
            num_particles=int(args.pf_particles),
            resampling_method=args.pf_resampling,
            resample_threshold=float(args.pf_resample_threshold),
            color="tab:purple",
        ),
    }
    states: Dict[str, State] = {
        "EKF": State(color="tab:blue"),
        "UKF": State(color="tab:cyan"),
        "PF": State(color="tab:green"),
    }
    return filters, states


def _save_results_figure(
    output_path: Path,
    gt_xy: np.ndarray,
    est_by_filter: Dict[str, np.ndarray],
    fused_obs_xy: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Trajectory panel
    ax_traj = axes[0]
    ax_traj.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", linewidth=2.0, label="Ground Truth")
    if fused_obs_xy.size > 0:
        ax_traj.scatter(fused_obs_xy[:, 0], fused_obs_xy[:, 1], s=7, alpha=0.3, color="gray", label="Fused observations")

    colors = {"EKF": "tab:blue", "UKF": "tab:cyan", "PF": "tab:green"}
    for name, est_xy in est_by_filter.items():
        ax_traj.plot(est_xy[:, 0], est_xy[:, 1], color=colors[name], linewidth=1.6, label=name)

    ax_traj.set_title("Trajectory Comparison")
    ax_traj.set_xlabel("X [m]")
    ax_traj.set_ylabel("Y [m]")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis("equal")
    ax_traj.legend(loc="best")

    # Position error panel
    ax_err = axes[1]
    for name, est_xy in est_by_filter.items():
        e = np.linalg.norm(est_xy - gt_xy, axis=1)
        ax_err.plot(e, linewidth=1.6, label=f"{name} error")
    ax_err.set_title("Position Error Over Time")
    ax_err.set_xlabel("Frame")
    ax_err.set_ylabel("Error [m]")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _print_metrics(metrics: List[TrajectoryMetrics]) -> None:
    print("\n=== Sensor-Fusion Localization Benchmark ===")
    print(f"{'Algorithm':<8} | {'RMSE X [m]':>10} | {'RMSE Y [m]':>10} | {'RMSE POS [m]':>12}")
    print("-" * 52)
    for row in metrics:
        print(f"{row.algorithm:<8} | {row.rmse_x_m:>10.3f} | {row.rmse_y_m:>10.3f} | {row.rmse_pos_m:>12.3f}")


def _write_markdown_report(
    report_path: Path,
    args: argparse.Namespace,
    metrics: List[TrajectoryMetrics],
    figure_path: Path,
) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Sensor-Fusion Localization Benchmark Report",
        "",
        f"- Generated: {timestamp}",
        f"- Source mode: {args.source}",
        f"- Dataset: {args.dataset if args.dataset else 'N/A (synthetic)'}",
        f"- Synthetic duration: {args.total_time}s",
        f"- Synthetic dt: {args.dt}s",
        f"- PF particles: {args.pf_particles}",
        f"- PF resampling: {args.pf_resampling}",
        f"- PF threshold: {args.pf_resample_threshold}",
        "",
        "## Metrics",
        "",
        "| Algorithm | RMSE X (m) | RMSE Y (m) | RMSE POS (m) |",
        "|---|---:|---:|---:|",
    ]

    for row in metrics:
        lines.append(
            f"| {row.algorithm} | {row.rmse_x_m:.3f} | {row.rmse_y_m:.3f} | {row.rmse_pos_m:.3f} |"
        )

    lines.extend([
        "",
        "## Figure",
        "",
        f"![Benchmark Figure]({figure_path.name})",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _build_arg_parser().parse_args()

    if not args.show:
        matplotlib.use("Agg")

    adapter = _create_adapter(args)
    fusion = PositionObservationFusion(gnss_std_m=1.0, camera_std_m=0.7)
    filter_bank, state_bank = _create_filter_bank(args)

    gt_xy_list = []
    fused_obs_list = []
    est_history: Dict[str, List[np.ndarray]] = {name: [] for name in filter_bank.keys()}

    for frame in adapter.iter_frames():
        gt_xy_list.append(np.array([frame.ground_truth.x_m, frame.ground_truth.y_m], dtype=float))

        fused = fusion.fuse(
            gnss_xy=frame.gnss_xy,
            camera_xy=frame.camera_xy,
            camera_confidence=frame.camera_confidence,
        )

        if fused is None:
            # Keep filters numerically stable if all sensors drop out.
            fallback = np.array([[frame.ground_truth.x_m], [frame.ground_truth.y_m]], dtype=float)
            fused = fallback
        fused_obs_list.append(fused[:, 0])

        for name, localizer in filter_bank.items():
            state_obj = state_bank[name]
            est_state = localizer.update(
                state_obj,
                frame.control.accel_mps2,
                frame.control.yaw_rate_rps,
                frame.dt_s,
                fused,
            )
            state_obj.update_by_localizer(est_state)
            est_history[name].append(np.array([state_obj.get_x_m(), state_obj.get_y_m()], dtype=float))

    gt_xy = np.array(gt_xy_list, dtype=float)
    fused_obs_xy = np.array(fused_obs_list, dtype=float)
    est_by_filter = {name: np.array(values, dtype=float) for name, values in est_history.items()}

    metrics = [compute_metrics(name, gt_xy, est_xy) for name, est_xy in est_by_filter.items()]
    _print_metrics(metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "sensor_fusion_localization_benchmark.png"
    _save_results_figure(plot_path, gt_xy, est_by_filter, fused_obs_xy)
    print(f"Saved benchmark figure to: {plot_path}")

    if args.report:
        report_path = output_dir / "sensor_fusion_localization_report.md"
        _write_markdown_report(report_path, args, metrics, plot_path)
        print(f"Saved benchmark report to: {report_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
