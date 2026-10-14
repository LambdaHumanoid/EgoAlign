from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from egoalign.palm_alignment import matrix_to_rot6d, quality_gate, rot6d_to_matrix
from egoalign.palm_replay import metrics, select_round
from egoalign.pose_ik import (
    ARMS, SPINE, CONFIG, geodesic, optimize, rotation_update, smooth_valid, wrist_rotations,
)
from egoalign.sample_pipeline import plan
from test_staged_ik import endpoints, skeleton


@pytest.mark.parametrize("method", [None, "paper", "palm"])
def test_method_selection_is_explicit_and_defaults_to_paper(method):
    args = ("basket", Path("/raw"), Path("/output"), Path("/robot.xml"), "/python")
    commands = plan(*args) if method is None else plan(*args, alignment_method=method)
    command = commands[3]
    assert command[command.index("--alignment-method") + 1] == (method or "paper")


@pytest.mark.parametrize("method", [None, "paper", "palm"])
def test_endpoint_cli_defaults_to_paper(method):
    from scripts.human_data.run_endpoint_retarget_dataset import build_parser

    parser = build_parser()
    assert parser.get_default("alignment_method") == "paper"
    args = ["--raw-dataset", "/raw", "--baseline-dataset", "/baseline",
            "--output-root", "/output", "--state-json", "/state.json", "--mode", "preflight"]
    if method is not None:
        args += ["--alignment-method", method]
    assert parser.parse_args(args).alignment_method == (method or "paper")


@pytest.mark.parametrize("method", [None, "paper", "palm"])
def test_sample_cli_defaults_to_paper(method, monkeypatch, capsys, tmp_path):
    import json
    from egoalign import sample_pipeline

    args = ["sample_pipeline", "basket", "--raw-root", str(tmp_path / "raw"),
            "--output-root", str(tmp_path / "output"), "--robot-xml", "/robot.xml", "--dry-run"]
    if method is not None:
        args += ["--alignment-method", method]
    monkeypatch.setattr("sys.argv", args)
    assert sample_pipeline.main() == 0
    command = json.loads(capsys.readouterr().out)[3]
    assert command[command.index("--alignment-method") + 1] == (method or "paper")
    assert not (tmp_path / "output").exists()


def test_orientation_encoding_preserves_material_axes():
    rotations = Rotation.random(21, random_state=7).as_matrix().reshape(7, 3, 3, 3)
    np.testing.assert_allclose(rot6d_to_matrix(matrix_to_rot6d(rotations)), rotations, atol=2e-7)
    with pytest.raises(ValueError, match="Degenerate"):
        rot6d_to_matrix(np.zeros((7, 18)))


def test_invalid_residual_frames_do_not_bias_valid_neighbors():
    values = np.ones((7, 2, 3))
    valid = np.array([False, True, True, True, True, True, False])
    values[~valid] = 1e9
    np.testing.assert_allclose(smooth_valid(values, valid), 1)
    with pytest.raises(ValueError, match="valid frames"):
        smooth_valid(values, np.zeros(7, bool))


def test_rotation_feedback_has_increment_total_caps_and_keeps_invalid_frames():
    identity = np.broadcast_to(np.eye(3), (7, 2, 3, 3)).copy()
    target = np.broadcast_to(Rotation.from_euler("x", 90, degrees=True).as_matrix(), identity.shape)
    valid = np.array([False, True, True, True, True, True, False])
    correction = identity.copy()
    for _ in range(6):
        previous = correction.copy()
        correction = rotation_update(correction, target, identity, valid)
        assert np.rad2deg(geodesic(correction, previous)).max() <= 15 + 1e-6
        assert np.rad2deg(geodesic(correction, identity)).max() <= 45 + 1e-6
        np.testing.assert_allclose(correction[~valid], identity[~valid], atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_pose_optimizer_keeps_exact_reachable_reference_and_handles_short_clips(n):
    rest, parents = skeleton()
    base = np.zeros((n, 63), np.float32)
    base.reshape(n, 21, 3)[:, [12, 13], 0] = .1
    target = endpoints(base, rest, parents)
    rotations = wrist_rotations(torch.tensor(base), parents).numpy()
    anchor = np.broadcast_to(np.eye(3), rotations.shape).copy()
    dirty = base.reshape(n, 21, 3).copy()
    dirty[:, [12, 13]] += .4
    dirty[:, SPINE] += .08
    pose, proof = optimize(base, dirty.reshape(n, 63), rest, parents, target, rotations,
                           anchor, np.ones(n, bool), device="cpu",
                           config=replace(CONFIG, arm_iterations=2, spine_iterations=2))
    np.testing.assert_allclose(pose, base, atol=1e-6)
    assert np.isfinite(proof["selected_score"])


def test_pose_optimizer_limits_changes_and_rejects_nonfinite_input():
    rest, parents = skeleton()
    base = np.zeros((4, 63), np.float32)
    target = endpoints(base, rest, parents) + .1
    rotations = wrist_rotations(torch.tensor(base), parents).numpy()
    anchor = np.broadcast_to(np.eye(3), rotations.shape).copy()
    args = (base, base.copy(), rest, parents, target, rotations, anchor, np.ones(4, bool))
    pose, _ = optimize(*args, device="cpu", config=replace(CONFIG, arm_iterations=5, spine_iterations=5))
    delta = (pose - base).reshape(4, 21, 3)
    fixed = [j for j in range(21) if j not in set(ARMS) | set(SPINE)]
    np.testing.assert_array_equal(delta[:, fixed], 0)
    assert np.linalg.norm(delta[:, SPINE], axis=-1).max() <= CONFIG.spine_limit_rad + 1e-6
    assert np.abs(delta[:, [19, 20]]).max() <= CONFIG.wrist_limit_rad + 1e-6
    base[0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid base"):
        optimize(*args, device="cpu")


def test_selection_respects_position_guard_and_never_selects_unstable_replay():
    def score(position, drift, height=.8):
        return dict(mean_cm=position, orientation_drift_mean_deg=drift, joint_score_cm=position,
                    stability=dict(min_root_height_m=height, max_root_tilt_deg=4))
    scores = {"paper": score(2, 20), "round_00": score(2.2, 10),
              "round_01": score(3.1, 1), "round_02": score(2.3, 0, .4)}
    assert select_round(scores) == ("round_00", True)
    scores["round_00"]["mean_cm"] = 4
    assert select_round(scores) == ("round_00", False)
    for key in ("round_00", "round_01"):
        scores[key]["stability"]["min_root_height_m"] = .4
    assert select_round(scores) == (None, False)


def test_metrics_are_bilateral_cm_on_the_same_mask_and_gate_is_strict():
    target = np.zeros((30, 2, 3))
    position = target.copy()
    position[:, 0, 0], position[:, 1, 0] = .01, .03
    rotation = np.broadcast_to(np.eye(3), (30, 2, 3, 3))
    valid = np.ones(30, bool)
    valid[-1] = False
    position[-1] = 100
    actual = (position, rotation, valid, dict(min_root_height_m=.8, max_root_tilt_deg=5))
    score = metrics(actual, target, rotation, valid)
    assert score["mean_cm"] == pytest.approx(2)
    assert score["left_mean_cm"] == pytest.approx(1)
    assert score["right_mean_cm"] == pytest.approx(3)
    assert score["n_valid_frames"] == 29
    assert score["orientation_drift_mean_deg"] == 0
    baseline = dict(mean_cm=12, left_mean_cm=12, right_mean_cm=12)
    assert quality_gate(score, baseline, True)["accepted"]
    assert not quality_gate(score, baseline, False)["accepted"]
    assert not quality_gate(dict(score, max_cm=6.1), baseline, True)["accepted"]


def test_palm_command_rejects_overlapping_output_before_writing(tmp_path):
    from egoalign.palm_alignment import run
    source = tmp_path / "source"
    source.mkdir()
    args = SimpleNamespace(workers=1, gate_budget_limit_improvement_only=False,
                           endpoint_parts="hands", hand_endpoint_side="both", mode="preflight",
                           raw_dataset=source, baseline_dataset=tmp_path / "baseline",
                           baseline_replay_root=None, output_root=source / "output")
    with pytest.raises(ValueError, match="disjoint"):
        run(args)
    assert not args.output_root.exists()
