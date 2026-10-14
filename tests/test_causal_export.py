"""Causal export must use one captured control tick, never inferred labels."""
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from scripts.human_data.processing import apply_sonic_mujoco_feedback as export


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_requires_captured_feedback(monkeypatch, explicit):
    argv = ["stage-c", "--source-root", "input", "--output-root", "output",
            "--telemetry-root", "replay"]
    if explicit:
        argv += ["--feedback-mode", "captured_cpp"]
    monkeypatch.setattr(sys, "argv", argv)
    assert export.parse_args().feedback_mode == "captured_cpp"


def test_cli_rejects_state_only_feedback(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "stage-c", "--source-root", "input", "--output-root", "output",
        "--telemetry-root", "replay", "--feedback-mode", "velocity_gate",
    ])
    with pytest.raises(SystemExit) as error:
        export.parse_args()
    assert error.value.code == 2


def test_unsupported_feedback_preserves_output(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing result")
    args = SimpleNamespace(source_root=tmp_path / "input", output_root=output,
                           telemetry_root=tmp_path / "replay", overwrite=True,
                           feedback_mode="velocity_gate")
    with pytest.raises(ValueError, match="unsupported feedback mode"):
        export.process_dataset(args)
    assert marker.read_text() == "existing result"


@pytest.mark.parametrize("state_schema", export.STATE_SCHEMAS)
@pytest.mark.parametrize("explicit_mode", [False, True])
def test_export_uses_same_tick_state_action_and_token(monkeypatch, state_schema, explicit_mode):
    frames = 4
    rng = np.random.default_rng(42)
    widths = {"observation.state": 43, "observation.eef_state": 14,
              "action.wbc": 43, "action.motion_token": 64,
              "observation.root_orientation": 4, "observation.projected_gravity": 3,
              "observation.init_base_quat": 4, "teleop.g1_joint_pos": 29,
              "teleop.g1_joint_vel": 29, "teleop.left_hand_joints": 6,
              "teleop.right_hand_joints": 6}
    values = {key: rng.normal(size=(frames, width)) for key, width in widths.items()}
    columns = {key: export._fixed_list(value, pa.float64()) for key, value in values.items()}
    columns.update(timestamp=pa.array(np.arange(frames) / 50),
                   camera=pa.array(["same-video.mp4"] * frames))
    source = pa.table(columns)
    feedback = {key: rng.normal(size=(frames, width)) for key, width in
                {"body_q": 29, "body_dq": 29, "left_hand_q": 7,
                 "right_hand_q": 7, "policy_action": 29, "token_state": 64}.items()}
    feedback["base_quat"] = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1))
    if explicit_mode:
        feedback["feedback_mode"] = np.asarray("captured_cpp")
    eef = rng.normal(size=(frames, 14))

    def measured_fk(body, left, right):
        np.testing.assert_array_equal(body, feedback["body_q"])
        np.testing.assert_array_equal(left, feedback["left_hand_q"])
        np.testing.assert_array_equal(right, feedback["right_hand_q"])
        return eef

    monkeypatch.setattr(export, "_compute_eef_state", measured_fk)
    result, replacements = export.rebuild_episode(source, feedback, state_schema=state_schema)
    assert set(replacements) == set(export.CHANGED_COLUMNS)
    physical = values["observation.state"].copy()
    physical[:, export.BODY_STATE_INDICES] = feedback["body_q"]
    physical[:, export.LEFT_HAND_STATE_INDICES] = feedback["left_hand_q"]
    physical[:, export.RIGHT_HAND_STATE_INDICES] = feedback["right_hand_q"]
    expected_state = (export.physical_qpos_to_isaaclab_offset_state(physical)
                      if state_schema == "isaaclab_offsets" else physical)
    np.testing.assert_array_equal(export._stack_fixed_list(result, "observation.state", 43), expected_state)
    np.testing.assert_array_equal(export._stack_fixed_list(result, "action.motion_token", 64), feedback["token_state"])
    action = export._stack_fixed_list(result, "action.wbc", 43)
    np.testing.assert_array_equal(action[:, export.BODY_STATE_INDICES], feedback["policy_action"])
    for indices in (export.LEFT_HAND_STATE_INDICES, export.RIGHT_HAND_STATE_INDICES):
        np.testing.assert_array_equal(action[:, indices], values["action.wbc"][:, indices])
    np.testing.assert_array_equal(export._stack_fixed_list(result, "observation.eef_state", 14), eef)
    for key in ("teleop.left_hand_joints", "teleop.right_hand_joints", "timestamp", "camera"):
        assert result[key].equals(source[key])


def test_rebuild_rejects_state_only_feedback():
    feedback = {"body_q": np.zeros((1, 29)), "body_dq": np.zeros((1, 29)),
                "base_quat": np.array([[1.0, 0.0, 0.0, 0.0]]),
                "feedback_mode": np.asarray("velocity_gate")}
    with pytest.raises(ValueError, match="unsupported feedback mode"):
        export.rebuild_episode(pa.table({"timestamp": [0.0]}), feedback)
