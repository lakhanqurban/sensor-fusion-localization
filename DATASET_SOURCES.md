# Usable Online Datasets for Sensor-Fusion Localization

This shortlist focuses on datasets that are practical for this project's pipeline:
camera/perception + localization + benchmarking.

## 1) TUM RGB-D SLAM Dataset

- Main page: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- Download list: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
- Why it fits:
  - RGB-D frames + ground-truth trajectory
  - Sequence lengths are manageable for rapid iteration
  - Includes ROS bag and file-format docs
- License:
  - CC BY 4.0 (data) and BSD-2-Clause (accompanying code) according to the dataset page
- Best first sequences:
  - `fr1/xyz`, `fr1/desk`, `fr1/room`
- Mapping into this repo:
  - Use `groundtruth.txt` style trajectories with `tools/convert_tum_to_replay_csv.py`
  - Fill GNSS/camera fields later with your visual front-end outputs

## 2) KITTI Raw / Odometry

- Main page: https://www.cvlibs.net/datasets/kitti/
- Raw data details: https://www.cvlibs.net/datasets/kitti/raw_data.php
- Why it fits:
  - Multi-sensor synchronized data at vehicle scale
  - Camera + Velodyne + GPS/IMU + calibrations
  - Strong baseline for autonomous driving localization
- Access constraints:
  - Requires login for raw download
- License/usage:
  - CC BY-NC-SA 3.0 with academic/non-commercial constraints listed on KITTI pages
- Mapping into this repo:
  - Convert OXTS pose/velocity streams into replay CSV schema
  - Camera estimator outputs can populate `camera_x_m`, `camera_y_m`, `camera_confidence`

## 3) Oxford RobotCar

- Main: https://robotcar-dataset.robots.ox.ac.uk/
- Docs: https://robotcar-dataset.robots.ox.ac.uk/documentation/
- Why it fits:
  - Long-term repeated-route localization benchmark
  - Multiple cameras + LiDAR + GPS/INS + VO reference
  - Challenging weather/lighting variation
- Access constraints:
  - Registration needed for downloads
- License/usage:
  - CC BY-NC-SA 4.0 (non-commercial academic use) per site license section
- Mapping into this repo:
  - `ins.csv` can drive GT/control approximation
  - Visual odometry and camera streams can feed camera observation channels

## 4) NCLT (University of Michigan)

- Main: https://robots.engin.umich.edu/nclt/
- Why it fits:
  - Rich long-term campus dataset with many sensors
  - Includes explicit `groundtruth.csv` and sensor files
  - Useful for robustness under seasonal/temporal changes
- License/usage:
  - ODbL / DbCL (share-alike style open database terms)
- Caveat:
  - Very large file sizes (many sessions are tens to hundreds of GB)
- Mapping into this repo:
  - Start from a short date/session and convert `groundtruth.csv` slices to replay CSV

## 5) comma2k19

- Repository: https://github.com/commaai/comma2k19
- Why it fits:
  - Camera + IMU + GNSS + CAN with provided global camera poses
  - Good for fused localization development with commodity sensors
- Access:
  - Public dataset reference + download pointer from repository README
- License:
  - MIT license shown in repository metadata
- Mapping into this repo:
  - Use global pose arrays for GT columns
  - Derive accel/yaw-rate from speed/orientation deltas if needed

## Recommended first real-data migration path

1. Start with TUM `fr1/xyz` (small, clean, quick feedback loop).
2. Use `tools/convert_tum_to_replay_csv.py` to create baseline replay CSV.
3. Add camera-estimated XY from your visual module into `camera_*` columns.
4. Benchmark EKF/UKF/PF with:
   - `run_sensor_fusion_benchmark.py --source replay --dataset <csv> --report`
5. Move to KITTI/Oxford once converter tooling and timing alignment are stable.
