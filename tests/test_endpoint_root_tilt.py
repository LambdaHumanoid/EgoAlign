"""Regression tests for gravity-aligned palm targets and replay measurements."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.human_data.endpoint_geometry import (
    measured_root_quaternions, root_tilt_rotations,
)
from scripts.human_data.smpl_endpoint_retargeting import (
    direct_big_smpl_endpoint_targets, root_centered_grounded_joints,
    validate_endpoint_correction_archive,
)


def wxyz(rotation):
    return np.atleast_2d(rotation.as_quat())[:, [3, 0, 1, 2]]


@pytest.mark.parametrize("axis", ["x", "y"])
def test_heading_removed_but_pitch_roll_retained(axis):
    tilt = Rotation.from_euler(axis, 15, degrees=True)
    q = wxyz(Rotation.from_euler("z", 73, degrees=True) * tilt)
    np.testing.assert_allclose(root_tilt_rotations(q).as_matrix()[0], tilt.as_matrix(), atol=1e-12)
    assert np.linalg.norm(root_tilt_rotations(q).apply([[.4, .1, 1.]]) - [[.4, .1, 1.]]) > .1


def test_target_and_correction_inverse_transform():
    rng = np.random.default_rng(91)
    big = rng.normal(size=(5, 55, 3))
    virtual = big * .8 + [1, 2, 3]
    q = wxyz(Rotation.from_euler("zyx", rng.normal(size=(5, 3)) * .15))
    left, right, left_grounded, right_grounded = direct_big_smpl_endpoint_targets(
        big, virtual, root_quaternion_wxyz=q
    )
    tilt = root_tilt_rotations(q)
    _, anchor = root_centered_grounded_joints(virtual, root_quaternion_wxyz=q)
    correction = np.broadcast_to([.02, -.03, .01], (5, 3))
    for local, grounded in [(left, left_grounded), (right, right_grounded)]:
        actual = tilt.apply(local + tilt.inv().apply(correction)) - anchor
        np.testing.assert_allclose(actual, grounded + correction, atol=1e-12)
    # Global heading and translation must not change the grounded comparison.
    q2 = wxyz(Rotation.from_euler("z", 1.3) * Rotation.from_quat(q[:, [1, 2, 3, 0]]))
    first, _ = root_centered_grounded_joints(big, root_quaternion_wxyz=q)
    second, _ = root_centered_grounded_joints(big + [5, -2, 4], root_quaternion_wxyz=q2)
    np.testing.assert_allclose(first, second, atol=1e-12)


@pytest.mark.parametrize("q", [np.zeros((2, 4)), np.full((2, 4), np.nan), np.ones((2, 3))])
def test_invalid_root_orientation_rejected(q):
    with pytest.raises(ValueError):
        root_tilt_rotations(q)


def test_missing_measured_root_is_not_replaced_with_identity():
    with pytest.raises(ValueError, match="base_quat_measured"):
        measured_root_quaternions({"base_quat_target": [[1, 0, 0, 0]]}, 1)


def test_old_correction_frame_rejected():
    with pytest.raises(ValueError, match="comparison_frame"):
        validate_endpoint_correction_archive({}, expected_parent_candidate_sha256="x", expected_shape=(2, 3))


def test_measured_fk_retains_root_tilt_and_waist_but_not_heading(tmp_path):
    pytest.importorskip("mujoco")
    from egoalign.adapters.sonic.telemetry_schema import MUJOCO_BODY_JOINT_NAMES
    from scripts.human_data.evaluate_endpoint_retarget_replay import measured_endpoints_from_control

    # A portable articulated fixture with the same named measurement anchors.
    names = MUJOCO_BODY_JOINT_NAMES
    bodies = []
    for name in names:
        body = name.replace("_joint", "_link")
        bodies.append(f'<body name="{body}" pos=".01 0 .04"><joint name="{name}" axis="0 1 0"/>'
                      '<geom type="sphere" size=".01" mass=".1"/>')
    xml = tmp_path / "robot.xml"
    xml.write_text('<mujoco><worldbody><body name="pelvis"><freejoint/>'
                   '<geom type="sphere" size=".01" mass="1"/>' + ''.join(bodies)
                   + '</body>' * len(bodies) + '</body></worldbody></mujoco>')
    q = np.zeros((4, 29))
    q[3, names.index("waist_pitch_joint")] = .2
    roots = wxyz(Rotation.from_euler("y", [0, 15, 15, 0], degrees=True))
    roots[2] = wxyz(Rotation.from_euler("z", 70, degrees=True)
                   * Rotation.from_euler("y", 15, degrees=True))[0]
    control = tmp_path / "control.npz"
    np.savez(control, joint_names=names, measured_joint_names=names,
             body_q_measured=q, base_quat_measured=roots, format_version=4)
    endpoints, info = measured_endpoints_from_control(control, xml, grounded=True)
    from egoalign.palm_replay import measured_poses
    positions, rotations = measured_poses(q, roots, names, xml)
    np.testing.assert_allclose(positions[:, 0], endpoints["hand_left"], atol=1e-12)
    np.testing.assert_allclose(positions[:, 1], endpoints["hand_right"], atol=1e-12)
    np.testing.assert_allclose(rotations[1], rotations[2], atol=1e-12)
    assert not np.allclose(rotations[0], rotations[1])
    assert not np.allclose(rotations[0], rotations[3])
    for side in ("hand_left", "hand_right"):
        assert np.linalg.norm(endpoints[side][1] - endpoints[side][0]) > .02
        np.testing.assert_allclose(endpoints[side][1], endpoints[side][2], atol=1e-12)
        assert np.linalg.norm(endpoints[side][3] - endpoints[side][0]) > .02
    assert "base_quat_measured" in info["root_orientation_source"]
