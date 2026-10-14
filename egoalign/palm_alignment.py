"""Orientation-aware palm alignment using measured SONIC–MuJoCo feedback."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from egoalign.palm_replay import load_npz, metrics, sample_replay, select_round
from egoalign.io_safety import validate_output_root
from egoalign.runtime import human_skeleton_path
from egoalign.pose_ik import (
    ARMS, SPINE, CONFIG, clip_norm, optimize, rotation_update, smooth_valid, wrist_rotations,
)
from scripts.human_data.build_smpl_endpoint_retarget_candidate import (
    _load_human_joint_model, _invalidate_stale_robot_fields,
)
from scripts.human_data.endpoint_geometry import ENDPOINT_COMPARISON_FRAME, root_tilt_rotations
from scripts.human_data.processing.body_transforms import compute_wrist_joints
from scripts.human_data.smpl_endpoint_retargeting import (
    ENDPOINT_DEFINITION, canonical_output_joints_with_full,
    propagate_endpoint_delta_to_vr3pt, validate_endpoint_candidate_archive,
)

ROOT = Path(__file__).resolve().parents[1]
ALGORITHM = "palm_pose_alignment"


def identity():
    return dict(alignment_method="palm", algorithm=ALGORITHM, candidate_contract_version=1,
                endpoint_parts="hands", hand_endpoint_side="both", endpoint_definition=ENDPOINT_DEFINITION,
                endpoint_target_contract="four-MCP positions and original replay palm orientations",
                comparison_frame=ENDPOINT_COMPARISON_FRAME,
                wrist_local_rotation_policy="optimized and synchronized with SMPL",
                vr3pt_orientation_policy="SMPL wrist delta applied to source hand orientations; head unchanged")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def rot6d_to_matrix(values):
    a = values.reshape(-1, 3, 2, 3)
    xn = np.linalg.norm(a[:, :, 0], axis=-1, keepdims=True)
    if np.any(xn < 1e-8):
        raise ValueError("Degenerate first VR3 orientation axis")
    x = a[:, :, 0] / xn
    y = a[:, :, 1] - (a[:, :, 1] * x).sum(-1, keepdims=True) * x
    yn = np.linalg.norm(y, axis=-1, keepdims=True)
    if np.any(yn < 1e-8):
        raise ValueError("Degenerate second VR3 orientation axis")
    y /= yn
    return np.stack((x, y, np.cross(x, y)), -1)


def matrix_to_rot6d(values):
    return values[..., :2].swapaxes(-1, -2).reshape(len(values), 18).astype(np.float32)


def candidate_frame(frame, seed, pose, rest, parents):
    """Synchronize all encoder pose fields; keep root, time, cameras and fingers."""
    result = frame.copy()
    joints, _ = canonical_output_joints_with_full(
        pose, rest, parents, seed["baseline_wire_smpl_joints"], device="cpu")
    source_pose = np.stack(frame["teleop.smpl_pose"])
    with torch.no_grad():
        source_r = wrist_rotations(torch.tensor(source_pose), parents).numpy()
        new_r = wrist_rotations(torch.tensor(pose), parents).numpy()
    vr_source = np.stack(frame["teleop.vr_3pt_orientation"])
    vr = rot6d_to_matrix(vr_source)
    vr[:, :2] = (new_r @ source_r.swapaxes(-1, -2)) @ vr[:, :2]
    vr6 = matrix_to_rot6d(vr)
    vr6[:, 12:] = vr_source[:, 12:]
    positions = propagate_endpoint_delta_to_vr3pt(
        np.stack(frame["teleop.vr_3pt_position"]), seed["source_canonical_baseline_joints"],
        joints, propagate_head=True)
    result["teleop.smpl_pose"] = list(pose)
    result["teleop.smpl_joints"] = list(joints.reshape(-1, 72))
    result["teleop.vr_3pt_position"] = list(positions)
    result["teleop.vr_3pt_orientation"] = list(vr6)
    wrists = [compute_wrist_joints(row) for row in pose]
    result["teleop.left_wrist_joints"] = [item[0] for item in wrists]
    result["teleop.right_wrist_joints"] = [item[1] for item in wrists]
    _invalidate_stale_robot_fields(result)
    return result


def quality_gate(score, baseline, guarded):
    failures = []
    for name, maximum in (("mean_cm", 3.), ("p95_cm", 4.5), ("max_cm", 6.),
                          ("left_mean_cm", 3.5), ("right_mean_cm", 3.5)):
        if score[name] > maximum:
            failures.append(name)
    if score["mean_cm"] > .25 * baseline["mean_cm"]:
        failures.append("mean_reduction_below_75_percent")
    if any(score[key] > baseline[key] for key in ("left_mean_cm", "right_mean_cm")):
        failures.append("per_hand_regression")
    if score["n_valid_frames"] < 20:
        failures.append("insufficient_frames")
    if score["stability"]["min_root_height_m"] < .5 or score["stability"]["max_root_tilt_deg"] > 30:
        failures.append("unstable_replay")
    if not guarded:
        failures.append("position_guard")
    return dict(accepted=not failures, failures=failures,
                mask="all common valid source frames", delay_simulation_s=CONFIG.delay_s)


def validate_candidate(round_dir, baseline_parquet, staging_parquet):
    """Validate both artifact provenance and the fields actually sent to SONIC."""
    round_dir = Path(round_dir)
    contract = read_json(round_dir / "round_contract.json")
    for key, expected in identity().items():
        if contract.get(key) != expected:
            raise ValueError(f"Incompatible palm candidate {key}")
    for path, expected in contract["artifacts"].items():
        if digest(path) != expected:
            raise ValueError(f"Palm artifact hash mismatch: {path}")
    for path, expected in read_json(round_dir.parent / "inputs.json")["inputs"].items():
        if digest(path) != expected:
            raise ValueError(f"Palm input hash mismatch: {path}")
    if digest(baseline_parquet) != contract["baseline_sha256"]:
        raise ValueError("Palm candidate belongs to another baseline")
    z = load_npz(round_dir / "candidate_smpl.npz")
    seed = load_npz(contract["seed_candidate"])
    frame = pd.read_parquet(baseline_parquet)
    staged = pd.read_parquet(staging_parquet)
    pose = z["candidate_smpl_pose"]
    expected = candidate_frame(frame, seed, pose, z["canonical_rest"], z["parents"].tolist())
    allowed = set(ARMS) | set(SPINE)
    locked = [j for j in range(21) if j not in allowed]
    if not np.array_equal(pose.reshape(-1, 21, 3)[:, locked], seed["baseline_smpl_pose"].reshape(-1, 21, 3)[:, locked]):
        raise ValueError("Candidate changed a constrained source joint")
    delta = (pose - seed["baseline_smpl_pose"]).reshape(-1, 21, 3)
    limits = np.asarray([.90, .90, .75, .75, CONFIG.wrist_limit_rad, CONFIG.wrist_limit_rad])[None, :, None]
    if np.any(np.abs(delta[:, ARMS]) > limits + 1e-6):
        raise ValueError("Candidate exceeds arm rotation bounds")
    if np.any(np.linalg.norm(delta[:, SPINE], axis=-1) > CONFIG.spine_limit_rad * z["gate"][:, None] + 1e-6):
        raise ValueError("Candidate exceeds torso rotation bounds")
    fixed_arm_phase = [j for j in range(21) if j not in ARMS]
    if not np.array_equal(z["arm_pose"].reshape(-1, 21, 3)[:, fixed_arm_phase],
                          seed["baseline_smpl_pose"].reshape(-1, 21, 3)[:, fixed_arm_phase]):
        raise ValueError("Candidate proof changed a constrained reference joint")
    if len(staged) != len(expected) or set(staged) != set(expected):
        raise ValueError("Candidate changed the episode schema or length")
    for column in expected:
        if column in {"index", "episode_index", "frame_index"}:
            continue
        a, b = np.asarray(expected[column].tolist()), np.asarray(staged[column].tolist())
        if np.issubdtype(a.dtype, np.number):
            if not np.isfinite(a).all() or not np.allclose(a, b, rtol=0, atol=1e-6):
                raise ValueError(f"Candidate input-field mismatch: {column}")
        elif not np.array_equal(a, b):
            raise ValueError(f"Candidate metadata mismatch: {column}")
    from scripts.human_data.run_endpoint_retarget_dataset import validate_replay
    validate_replay(round_dir / "replay", Path(staging_parquet))
    return identity()


def refine_episode(*, baseline_dataset, baseline_parquet, baseline_replay, baseline_replay_episode,
                   seed_candidate, seed_replay, output, robot_xml, compact_episode=0, source_episode=0,
                   resume=False):
    """Refine a validated position-alignment seed with three real pose replays."""
    from scripts.human_data.run_endpoint_retarget_dataset import (
        ROLLOUT_CONTRACT, build_isolated_staging, run_replay, validate_replay,
    )

    output = validate_output_root(output, [baseline_dataset, baseline_replay, seed_candidate, seed_replay])
    seed_candidate = Path(seed_candidate)
    seed = load_npz(seed_candidate)
    with np.load(seed_candidate, allow_pickle=False) as archive:
        validate_endpoint_candidate_archive(archive)
    frame = pd.read_parquet(baseline_parquet)
    if not np.array_equal(np.stack(frame["teleop.smpl_pose"]), seed["source_baseline_smpl_pose"]):
        raise ValueError("Position seed does not belong to this source SMPL trajectory")
    if not np.array_equal(np.asarray(frame["timestamp"]), seed["timestamp_s"]):
        raise ValueError("Position seed timestamps differ from the source")
    seed_parquet = Path(seed_replay).parent / "staging/data/chunk-000/episode_000000.parquet"
    validate_replay(Path(seed_replay), seed_parquet)
    if not np.array_equal(np.stack(pd.read_parquet(seed_parquet)["teleop.smpl_pose"]), seed["candidate_smpl_pose"]):
        raise ValueError("Position seed replay used a different SMPL trajectory")
    baseline_contract = Path(baseline_replay) / f"telemetry/episode_{baseline_replay_episode:06d}.rollout.json"
    rollout = read_json(baseline_contract)
    baseline_summary = read_json(Path(baseline_replay) / "summary.json")
    if (rollout.get("source_sha256") != digest(baseline_parquet)
            or rollout.get("source_rows") != len(frame) or rollout.get("replay_rows") != len(frame)
            or rollout.get("protocol") != "pose"
            or rollout.get("sim_episode_lifecycle_contract") != ROLLOUT_CONTRACT
            or rollout.get("pose_wrist_contract", {}).get("zero_fallback") is not False
            or baseline_replay_episode not in baseline_summary.get("successful_episodes", [])):
        raise ValueError("Baseline replay does not match the complete source episode")
    inputs = [Path(baseline_parquet), seed_candidate, Path(robot_xml),
              seed_parquet, baseline_contract, Path(baseline_replay) / "summary.json",
              Path(__file__), ROOT / "egoalign/pose_ik.py", ROOT / "egoalign/palm_replay.py",
              ROOT / "scripts/human_data/endpoint_geometry.py",
              ROOT / "scripts/human_data/smpl_endpoint_retargeting.py", human_skeleton_path()]
    for replay, ep in ((baseline_replay, baseline_replay_episode), (seed_replay, 0)):
        inputs.extend(Path(replay) / f"telemetry/episode_{ep:06d}.{kind}.npz" for kind in ("sim", "reference", "control"))
    provenance = dict(identity(), inputs={str(path): digest(path) for path in inputs}, config=asdict(CONFIG))
    if output.exists():
        if not resume or not (output / "inputs.json").exists() or read_json(output / "inputs.json") != provenance:
            raise ValueError(f"Output exists or has incompatible inputs: {output}")
        if (output / "final_decision.json").exists():
            decision = read_json(output / "final_decision.json")
            for index in range(3):
                directory = output / f"round_{index:02d}"
                validate_candidate(directory, baseline_parquet, directory / "staging/data/chunk-000/episode_000000.parquet")
            return decision
        raise ValueError(f"Incomplete palm run preserved at {output}; use a new output directory")
    output.mkdir(parents=True)
    write_json(output / "inputs.json", provenance)
    times = np.asarray(frame["timestamp"], dtype=float)
    times -= times[0]
    original = sample_replay(baseline_replay, times, robot_xml, baseline_replay_episode)
    paper = sample_replay(seed_replay, times, robot_xml)
    lock = original[1]
    root = root_tilt_rotations(np.stack(frame["teleop.body_quat_w"])).as_matrix()
    root_inv = root.swapaxes(-1, -2)[:, None]
    lock_local = root_inv @ lock
    parents = seed["staged_parents"].astype(int).tolist()
    source_pose = np.stack(frame["teleop.smpl_pose"])
    with torch.no_grad():
        source_rot = wrist_rotations(torch.tensor(source_pose), parents).numpy()
    anchor = source_rot.swapaxes(-1, -2) @ lock_local
    rest, model_parents = _load_human_joint_model()
    if list(model_parents) != parents:
        raise ValueError("Human skeleton parent graph differs from the seed")
    rest = rest * float(seed["canonical_uniform_scale"])
    gravity_correction = np.stack((seed["left_target_correction_m"], seed["right_target_correction_m"]), 1)
    correction = (root_inv @ gravity_correction[..., None])[..., 0]
    target_position = np.stack((seed["target_left_m"], seed["target_right_m"]), 1) - correction
    target_grounded = np.stack((seed["target_left_grounded_m"], seed["target_right_grounded_m"]), 1)
    rotation_correction = np.broadcast_to(np.eye(3), (len(frame), 2, 3, 3)).copy()
    initial = seed["candidate_smpl_pose"].copy()
    np.savez_compressed(output / "targets.npz", position_grounded=target_grounded, position_local=target_position,
                        original_orientation=lock, orientation_anchor=anchor, source_time_s=times,
                        source_root_quaternion_wxyz=seed["source_root_quaternion_wxyz"])
    rows = {"original": original, "paper": paper}
    for index in range(3):
        directory = output / f"round_{index:02d}"
        directory.mkdir()
        print(f"palm: episode {compact_episode}, round {index}: optimize", flush=True)
        pose, proof = optimize(seed["baseline_smpl_pose"], initial, seed["staged_virtual_rest_joints"], parents,
                               target_position + correction, rotation_correction @ lock_local,
                               anchor, original[2])
        candidate = candidate_frame(frame, seed, pose, rest, parents)
        candidate.to_parquet(directory / "candidate.parquet", index=False)
        arrays = {key: proof.pop(key) for key in ("arm_pose", "gate", "position_error_m", "rotation_chordal_error")}
        np.savez_compressed(directory / "candidate_smpl.npz", **arrays,
                            algorithm=np.asarray(ALGORITHM), candidate_smpl_pose=pose,
                            position_correction=correction, rotation_correction=rotation_correction,
                            position_target=target_position + correction,
                            orientation_target=rotation_correction @ lock_local,
                            canonical_rest=rest, parents=parents)
        write_json(directory / "ik_metrics.json", proof)
        staging = directory / "staging"
        staging_parquet = build_isolated_staging(directory / "candidate.parquet", baseline_dataset, compact_episode, staging)
        print(f"palm: episode {compact_episode}, round {index}: replay", flush=True)
        run_replay(staging, directory / "replay")
        replay_paths = validate_replay(directory / "replay", staging_parquet)
        actual = sample_replay(directory / "replay", times, robot_xml)
        rows[directory.name] = actual
        shared = original[2] & paper[2] & actual[2]
        write_json(directory / "replay_metrics.json", metrics(actual, target_grounded, lock, shared))
        np.savez_compressed(directory / "evaluation.npz", actual_position=actual[0], actual_rotation=actual[1], valid=actual[2])
        residual = (root_inv @ (target_grounded - actual[0])[..., None])[..., 0]
        increment = clip_norm(CONFIG.feedback_gain * smooth_valid(residual, shared), .08)
        increment[~shared] = 0
        correction = clip_norm(correction + increment, .12)
        rotation_correction = rotation_update(rotation_correction, lock_local, root_inv @ actual[1], shared)
        initial = pose
        artifacts = [directory / "candidate.parquet", directory / "candidate_smpl.npz", staging_parquet,
                     directory / "ik_metrics.json", directory / "replay_metrics.json", directory / "evaluation.npz",
                     output / "targets.npz", output / "inputs.json", seed_candidate, *replay_paths.values()]
        write_json(directory / "round_contract.json", dict(identity(), baseline_sha256=digest(baseline_parquet),
                   seed_candidate=str(seed_candidate), artifacts={str(path): digest(path) for path in artifacts}))
        validate_candidate(directory, baseline_parquet, staging_parquet)
    common = np.logical_and.reduce([item[2] for item in rows.values()])
    scores = {name: metrics(value, target_grounded, lock, common) for name, value in rows.items()}
    selected, guarded = select_round(scores)
    gate = quality_gate(scores[selected], scores["original"], guarded) if selected else dict(accepted=False, failures=["no_stable_round"])
    result = dict(identity(), compact_episode=compact_episode, source_episode=source_episode,
                  status="optimized_pass" if gate["accepted"] else "optimization_fail",
                  accepted_round=int(selected[-2:]) if selected and gate["accepted"] else None,
                  selected_round=int(selected[-2:]) if selected else None, position_guard_satisfied=guarded,
                  closed_loop_gate=gate, variants=scores, delay_simulation_s=CONFIG.delay_s,
                  orientation_reference="unoptimized robot replay, per source frame",
                  selection_rule="stable; position <= paper + 1 cm; minimum orientation drift; otherwise diagnostic pose score",
                  production_gate=dict(status="pending_dataset_smoke_calibration", accepted=False))
    np.savez_compressed(output / "common_mask.npz", valid=common)
    write_json(output / "final_decision.json", result)
    return result


def run(args):
    """Use the paper pipeline as initialization, then refine palm poses separately."""
    from scripts.human_data.evaluate_endpoint_retarget_replay import DEFAULT_ROBOT_XML
    from scripts.human_data.promote_endpoint_retarget_dataset import format_episode_path

    if args.workers != 1:
        raise ValueError("Palm alignment currently requires --workers 1; independent runs may use isolated runtime ports")
    if args.gate_budget_limit_improvement_only:
        raise ValueError("Palm alignment uses strict quality gates; the budget-relaxation option is paper-only")
    if args.endpoint_parts != "hands" or args.hand_endpoint_side != "both":
        raise ValueError("Palm alignment requires both hand endpoints")
    if args.mode in {"capacity", "promote"}:
        raise ValueError("Use preflight, smoke or full for palm alignment, then the promote command")
    sources = [args.raw_dataset, args.baseline_dataset]
    if args.baseline_replay_root is not None:
        sources.append(args.baseline_replay_root)
    output = validate_output_root(args.output_root, sources)
    if (output / "summary.json").exists() and not args.resume:
        raise ValueError("Alignment output exists; use --resume or a fresh output directory")
    paper_root = output / "paper_alignment"
    command = [os.environ.get("EGOALIGN_DATA_PYTHON", sys.executable),
               str(ROOT / "scripts/human_data/run_endpoint_retarget_dataset.py")]
    forwarded = list(sys.argv[1:])
    for flag, value in (("--alignment-method", "paper"), ("--output-root", str(paper_root))):
        for index, item in enumerate(forwarded):
            if item == flag:
                forwarded[index + 1] = value
                break
            if item.startswith(flag + "="):
                forwarded[index] = flag + "=" + value
                break
        else:
            forwarded.extend([flag, value])
    command.extend(forwarded)
    write_json(output / "commands.json", [command])
    subprocess.run(command, check=True)
    if args.dry_run or args.mode == "preflight":
        write_json(output / "run_plan.json", dict(identity(), initialization=str(paper_root),
                   refinement_rounds=3, dry_run=args.dry_run, config=asdict(CONFIG)))
        return 0
    preflight = read_json(paper_root / "preflight.json")
    summary = read_json(paper_root / "summary.json")
    baseline = args.baseline_dataset.expanduser().resolve()
    info = read_json(baseline / "meta/info.json")
    decisions = []
    for decision in summary["episodes"]:
        compact = int(decision["compact_episode"])
        seed_index = decision.get("accepted_round")
        if seed_index is None:
            seed_index = decision["rounds"][-1]["round"]
        seed = paper_root / f"episode_{compact:06d}/round_{seed_index:02d}"
        decisions.append(refine_episode(
            baseline_dataset=baseline,
            baseline_parquet=format_episode_path(baseline, info["data_path"], compact, int(info.get("chunks_size", 1000))),
            baseline_replay=Path(preflight["baseline_replay_root"]),
            baseline_replay_episode=int(preflight["baseline_compact_to_replay"][compact]),
            seed_candidate=seed / "candidate_smpl.npz", seed_replay=seed / "replay",
            output=output / f"episode_{compact:06d}", robot_xml=DEFAULT_ROBOT_XML,
            compact_episode=compact, source_episode=int(decision["source_episode"]), resume=args.resume))
        write_json(output / "summary.json", dict(identity(), completed=len(decisions), episodes=decisions))
    return 0
