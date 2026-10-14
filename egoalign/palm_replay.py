"""Measured palm poses and source-time sampling for orientation-aware alignment."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from egoalign.adapters.sonic.telemetry_schema import MUJOCO_BODY_JOINT_NAMES
from egoalign.pose_ik import CONFIG, geodesic
from scripts.human_data.endpoint_geometry import root_tilt_rotations
from scripts.human_data.evaluate_endpoint_retarget_replay import (
    LEFT_PALM_OFFSET, RIGHT_PALM_OFFSET, LEFT_FOOT_OFFSET, RIGHT_FOOT_OFFSET,
)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def measured_poses(joints, root_quaternions, joint_names, robot_xml):
    """MuJoCo palm-center XYZ and wrist-frame orientation, retaining root tilt."""
    import mujoco

    names = tuple(str(name) for name in joint_names)
    if len(names) != 29 or set(names) != set(MUJOCO_BODY_JOINT_NAMES):
        raise ValueError("Expected all 29 measured G1 joints with explicit names")
    if np.shape(joints) != (len(root_quaternions), 29) or not np.isfinite(joints).all():
        raise ValueError("Invalid measured joint array")
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    addresses = [model.joint(name).qposadr[0] for name in names]
    bodies = [model.body(name).id for name in (
        "left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link",
        "right_ankle_roll_link", "pelvis")]
    offsets = (LEFT_PALM_OFFSET, RIGHT_PALM_OFFSET, LEFT_FOOT_OFFSET, RIGHT_FOOT_OFFSET)
    tilt = root_tilt_rotations(root_quaternions).as_quat()[:, [3, 0, 1, 2]]
    positions = np.empty((len(joints), 2, 3))
    rotations = np.empty((len(joints), 2, 3, 3))
    for index, row in enumerate(joints):
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = 0
        data.qpos[3:7] = tilt[index]
        data.qpos[addresses] = row
        mujoco.mj_forward(model, data)
        points = [data.xpos[body] + data.xmat[body].reshape(3, 3) @ offset
                  for body, offset in zip(bodies, offsets)]
        anchor = np.array([*data.xpos[bodies[4], :2], min(points[2][2], points[3][2])])
        positions[index] = np.asarray(points[:2]) - anchor
        rotations[index] = data.xmat[bodies[:2]].reshape(2, 3, 3)
    return positions, rotations


def sample_replay(replay_root, query_source, robot_xml, episode=0):
    """Sample real simulation telemetry at source time plus a simulation-clock delay."""
    stem = Path(replay_root) / f"telemetry/episode_{episode:06d}"
    sim = load_npz(str(stem) + ".sim.npz")
    ref = load_npz(str(stem) + ".reference.npz")
    control = load_npz(str(stem) + ".control.npz")
    if not np.array_equal(np.sort(control["replay_frame_index"].reshape(-1)),
                          np.arange(len(ref["source_timestamp_s"]))):
        raise ValueError("Replay must cover every source frame exactly once")
    p, r = measured_poses(sim["body_joint_position"], sim["root_quaternion_wxyz"],
                          sim["joint_names"], robot_xml)
    source = ref["source_timestamp_s"] - ref["source_timestamp_s"][0]
    origin = ref["send_monotonic_ns"][0]
    send = (ref["send_monotonic_ns"] - origin) * 1e-9
    wall = (sim["monotonic_ns"] - origin) * 1e-9
    sim_time = sim["sim_time_s"]
    for name, times in (("source", source), ("send", send), ("wall", wall), ("simulation", sim_time)):
        if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"Nonmonotonic {name} timestamps")
    query_source = np.asarray(query_source)
    qwall = np.interp(query_source, source, send)
    qsim = np.interp(qwall, wall, sim_time) + CONFIG.delay_s
    valid = ((query_source >= source[0]) & (query_source <= source[-1]) &
             (qwall >= wall[0]) & (qwall <= wall[-1]) &
             (qsim >= sim_time[0]) & (qsim <= sim_time[-1]))
    q = np.clip(qsim, sim_time[0], sim_time[-1])
    actual_p = np.stack([np.interp(q, sim_time, col) for col in p.reshape(len(p), -1).T], -1).reshape(-1, 2, 3)
    actual_r = np.stack([Slerp(sim_time, Rotation.from_matrix(r[:, hand]))(q).as_matrix()
                         for hand in range(2)], 1)
    up = Rotation.from_quat(sim["root_quaternion_wxyz"][:, [1, 2, 3, 0]]).apply(
        np.tile([0, 0, 1], (len(p), 1)))
    stability = dict(min_root_height_m=float(sim["root_position"][:, 2].min()),
                     max_root_tilt_deg=float(np.rad2deg(np.arccos(np.clip(up[:, 2], -1, 1))).max()))
    return actual_p, actual_r, valid, stability


def metrics(actual, target_p, target_r, valid):
    if not np.asarray(valid).any():
        raise ValueError("No common valid replay frames")
    pe = np.linalg.norm(actual[0][valid] - target_p[valid], axis=-1) * 100
    drift = np.rad2deg(geodesic(actual[1][valid], target_r[valid]))
    if not np.isfinite(pe).all() or not np.isfinite(drift).all():
        raise ValueError("Nonfinite measured palm error")
    return dict(mean_cm=float(pe.mean()), p95_cm=float(np.percentile(pe, 95)), max_cm=float(pe.max()),
                left_mean_cm=float(pe[:, 0].mean()), right_mean_cm=float(pe[:, 1].mean()),
                orientation_drift_mean_deg=float(drift.mean()),
                orientation_drift_p95_deg=float(np.percentile(drift, 95)),
                joint_score_cm=float(np.sqrt(pe ** 2 + (CONFIG.orientation_length_m * 100 * np.deg2rad(drift)) ** 2).mean()),
                n_valid_frames=int(np.sum(valid)), stability=actual[3])


def select_round(scores):
    """Select a stable whole trajectory with an explicit position-regression guard."""
    stable = [key for key in scores if key.startswith("round_") and
              scores[key]["stability"]["min_root_height_m"] >= .5 and
              scores[key]["stability"]["max_root_tilt_deg"] <= 30]
    if not stable:
        return None, False
    guarded = [key for key in stable if scores[key]["mean_cm"] <= scores["paper"]["mean_cm"] + 1.]
    if guarded:
        return min(guarded, key=lambda key: scores[key]["orientation_drift_mean_deg"]), True
    return min(stable, key=lambda key: scores[key]["joint_score_cm"]), False
