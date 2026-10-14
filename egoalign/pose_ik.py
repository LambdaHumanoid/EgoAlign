"""SMPL palm-position optimization with replay-referenced orientation preservation."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
import torch

from egoalign.geometry.torch_transforms import angle_axis_to_rotation_matrix
from scripts.human_data.smpl_endpoint_retargeting import (
    OUTPUT_FROM_SMPL_INTERNAL, forward_joints_torch, output_frame_joints_torch,
    palm_centers_torch,
)


@dataclass(frozen=True)
class Config:
    orientation_length_m: float = .10
    arm_iterations: int = 350
    spine_iterations: int = 200
    learning_rate: float = .035
    wrist_limit_rad: float = .50
    spine_limit_rad: float = .10
    feedback_gain: float = .65
    rotation_increment_deg: float = 15.
    rotation_total_deg: float = 45.
    delay_s: float = .24


CONFIG = Config()
ARMS = np.array([16, 17, 18, 19, 20, 21]) - 1
SPINE = np.array([3, 6, 9]) - 1


def wrist_rotations(pose, parents):
    """Return wrist material axes in the output body frame, without root tilt."""
    p = pose.reshape(-1, 21, 3)
    r = angle_axis_to_rotation_matrix(p)
    identity = torch.eye(3, dtype=p.dtype, device=p.device).expand(len(p), 3, 3)
    global_r = [identity]
    for j in range(1, 22):
        global_r.append(global_r[parents[j]] @ r[:, j - 1])
    transform = torch.as_tensor(OUTPUT_FROM_SMPL_INTERNAL, dtype=p.dtype, device=p.device)
    return transform @ torch.stack([global_r[20], global_r[21]], 1)


def geodesic(a, b):
    relative = np.asarray(a) @ np.swapaxes(np.asarray(b), -1, -2)
    return Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(relative.shape[:-2])


def clip_norm(values, limit):
    return values * np.minimum(1., limit / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12))


def smooth_valid(values, valid):
    """Smooth residuals without allowing invalid samples to bias valid neighbors."""
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        raise ValueError("Residual smoothing requires valid frames")
    clean = np.asarray(values, dtype=float).copy()
    for column in clean.reshape(len(clean), -1).T:
        column[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), column[valid])
    return gaussian_filter1d(clean, 2, axis=0, mode="nearest")


def rotation_update(current, target, measured, valid, gain=CONFIG.feedback_gain):
    """Apply bounded SO(3) feedback; invalid frames retain their correction."""
    residual = target @ np.swapaxes(measured, -1, -2)
    rv = Rotation.from_matrix(residual.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 2, 3)
    rv = clip_norm(smooth_valid(rv, valid) * gain, np.deg2rad(CONFIG.rotation_increment_deg))
    rv[~valid] = 0
    updated = Rotation.from_rotvec(rv.reshape(-1, 3)).as_matrix().reshape(-1, 2, 3, 3) @ current
    total = Rotation.from_matrix(updated.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 2, 3)
    total = clip_norm(total, np.deg2rad(CONFIG.rotation_total_deg))
    return Rotation.from_rotvec(total.reshape(-1, 3)).as_matrix().reshape(-1, 2, 3, 3)


def release_gate(position_error, orientation_error):
    x = np.maximum((position_error.max(1) - .03) / .03,
                   (orientation_error.max(1) - np.deg2rad(15)) / np.deg2rad(30))
    x = np.clip(x, 0, 1)
    gate = x * x * (3 - 2 * x)
    for i in range(1, len(gate)):
        gate[i] = min(gate[i], gate[i - 1] + .1)
    for i in range(len(gate) - 2, -1, -1):
        gate[i] = min(gate[i], gate[i + 1] + .1)
    return gate.astype(np.float32)


def optimize(base, initial, rest, parents, target_position, target_rotation,
             anchor, valid, device=None, config=CONFIG):
    """Optimize a complete SMPL trajectory against position and orientation targets."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    valid = np.asarray(valid, dtype=bool)
    n = len(base)
    if n == 0 or valid.shape != (n,) or not valid.any():
        raise ValueError("Optimization requires at least one valid frame")
    for name, value, shape in (
        ("base", base, (n, 63)), ("initial", initial, (n, 63)),
        ("position", target_position, (n, 2, 3)),
        ("rotation", target_rotation, (n, 2, 3, 3)), ("anchor", anchor, (n, 2, 3, 3)),
    ):
        if np.shape(value) != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name}: expected finite {shape}")
    if min(config.arm_iterations, config.spine_iterations) < 1:
        raise ValueError("Iteration counts must be positive")

    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    b = tensor(base).reshape(-1, 21, 3)
    init = tensor(initial).reshape_as(b)
    rest_t, target_p, target_r, anchor_t = map(tensor, (rest, target_position, target_rotation, anchor))
    mask = tensor(valid)[:, None]
    limits = tensor([.90, .90, .75, .75, config.wrist_limit_rad, config.wrist_limit_rad])[None, :, None]

    def fk(pose):
        joints = output_frame_joints_torch(forward_joints_torch(pose.reshape(-1, 63), rest_t, parents))
        return torch.stack(palm_centers_torch(joints), 1), wrist_rotations(pose, parents) @ anchor_t

    def errors(pose):
        p, r = fk(pose)
        return (p - target_p).square().sum(-1), (r - target_r).square().sum((-1, -2)) / 2

    losses, gate_np = [], np.zeros(n, np.float32)
    for stage, iterations in enumerate((config.arm_iterations, config.spine_iterations)):
        start = init if stage == 0 else arm_pose
        ap = torch.nn.Parameter(torch.atanh(((start[:, ARMS] - b[:, ARMS]) / limits).clamp(-.999, .999)))
        sp = torch.nn.Parameter(torch.zeros((n, 3, 3), device=device))
        opt = torch.optim.Adam([ap] if stage == 0 else [ap, sp], lr=config.learning_rate)
        gate = tensor(gate_np)[:, None, None]
        best_score = float("inf") if stage == 0 else float(arm_score)
        best = b.clone() if stage == 0 else arm_pose.clone()
        for it in range(iterations):
            opt.zero_grad(set_to_none=True)
            pose = b.clone()
            pose[:, ARMS] = b[:, ARMS] + limits * torch.tanh(ap)
            if stage:
                sd = config.spine_limit_rad * sp / (1 + torch.linalg.vector_norm(sp, dim=-1, keepdim=True)) * gate
                pose[:, SPINE] = b[:, SPINE] + sd
            pe, re = errors(pose)
            score = ((pe + config.orientation_length_m ** 2 * re) * mask).sum() / (mask.sum() * 2)
            delta = pose - b
            vel = (delta[1:] - delta[:-1]).square().mean() if n > 1 else delta.sum() * 0
            acc = (delta[2:] - 2 * delta[1:-1] + delta[:-2]).square().mean() if n > 2 else delta.sum() * 0
            loss = 500 * score + .6 * delta[:, ARMS].square().mean() + 18 * vel + 8 * acc
            if stage:
                # Keep the derivative finite when an endpoint matches exactly.
                distance = pe.clamp_min(1e-12).sqrt()
                arm_distance = arm_pe.clamp_min(1e-12).sqrt()
                loss = loss + 6 * sd.square().mean() + 2000 * torch.relu(distance - arm_distance).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite palm alignment loss")
            with torch.no_grad():
                eligible = not stage or bool(torch.all(pe.sqrt() <= arm_pe.sqrt() + .002))
                if eligible and float(score) < best_score:
                    best_score, best = float(score), pose.detach().clone()
            loss.backward()
            opt.step()
            if it % 50 == 0:
                losses.append(dict(stage=stage + 1, iteration=it, loss=float(loss.detach()), score=float(score.detach())))
        if stage == 0:
            arm_pose, arm_score = best.clone(), best_score
            with torch.no_grad():
                arm_pe, _ = errors(arm_pose)
                _, ar = fk(arm_pose)
                ae = geodesic(ar.cpu().numpy(), target_rotation)
            gate_np = release_gate(arm_pe.sqrt().cpu().numpy(), ae)
            gate_np[~valid] = 0
    with torch.no_grad():
        pe, re = errors(best)
    return best.cpu().numpy().reshape(-1, 63), dict(
        config=asdict(config), losses=losses, arm_score=arm_score, selected_score=best_score,
        arm_pose=arm_pose.cpu().numpy().reshape(-1, 63), gate=gate_np,
        position_error_m=pe.sqrt().cpu().numpy(), rotation_chordal_error=re.sqrt().cpu().numpy())
