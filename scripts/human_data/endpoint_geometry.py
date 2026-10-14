"""Shared, immutable MuJoCo body-point definitions for endpoint retargeting."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


ENDPOINT_COMPARISON_FRAME = "gravity_aligned_heading_pelvis_xy_lower_foot_z"


def root_tilt_rotations(root_quaternion_wxyz: np.ndarray) -> Rotation:
    """Remove heading only, retaining the measured root's pitch and roll."""
    q = np.asarray(root_quaternion_wxyz, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or not len(q):
        raise ValueError("root quaternions must have non-empty shape (N, 4), wxyz")
    norms = np.linalg.norm(q, axis=1)
    if not np.isfinite(q).all() or np.any(np.abs(norms - 1.0) > 2e-3):
        raise ValueError("root quaternions must be finite and unit length")
    root = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    matrices = root.as_matrix()
    if np.any(np.hypot(matrices[:, 0, 0], matrices[:, 1, 0]) < 1e-6):
        raise ValueError("root heading is undefined for a vertical forward axis")
    yaw = np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0])
    return Rotation.from_euler("z", -yaw) * root


def measured_root_quaternions(control, rows: int) -> np.ndarray:
    """Read same-tick pelvis IMU orientation; never substitute a target pose."""
    if "base_quat_measured" not in control:
        raise ValueError("measured endpoint FK requires base_quat_measured telemetry")
    q = np.asarray(control["base_quat_measured"], dtype=np.float64)
    if q.shape != (rows, 4):
        raise ValueError("measured root quaternion and joint frame counts differ")
    root_tilt_rotations(q)
    return q


G1_LEFT_PALM_ENDPOINT_BODY = "left_wrist_yaw_link"
G1_RIGHT_PALM_ENDPOINT_BODY = "right_wrist_yaw_link"
G1_LEFT_FOOT_ENDPOINT_BODY = "left_ankle_roll_link"
G1_RIGHT_FOOT_ENDPOINT_BODY = "right_ankle_roll_link"

G1_LEFT_PALM_ENDPOINT_OFFSET_M = np.array(
    [0.0415, 0.0030, 0.0], dtype=np.float64
)
G1_RIGHT_PALM_ENDPOINT_OFFSET_M = np.array(
    [0.0415, -0.0030, 0.0], dtype=np.float64
)
G1_LEFT_FOOT_ENDPOINT_OFFSET_M = np.array(
    [0.10, 0.0, -0.025], dtype=np.float64
)
G1_RIGHT_FOOT_ENDPOINT_OFFSET_M = np.array(
    [0.10, 0.0, -0.025], dtype=np.float64
)
