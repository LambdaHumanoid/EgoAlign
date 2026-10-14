import numpy as np
import pytest

from egoalign import palm_replay


@pytest.fixture
def telemetry(monkeypatch):
    # Simulation advances twice as fast as the sender clock in this fixture.
    # A 0.24-second simulation delay must not become a 0.48-second delay.
    sim_time = np.arange(5, dtype=float) * 2
    sim = dict(sim_time_s=sim_time, monotonic_ns=np.arange(5) * 1_000_000_000,
               body_joint_position=np.zeros((5, 29)), joint_names=np.arange(29),
               root_quaternion_wxyz=np.tile([1., 0, 0, 0], (5, 1)),
               root_position=np.tile([0., 0, .8], (5, 1)))
    ref = dict(source_timestamp_s=np.array([10., 11, 12]),
               send_monotonic_ns=np.array([1, 2, 3]) * 1_000_000_000)
    control = dict(replay_frame_index=np.array([2, 0, 1]))
    payloads = dict(sim=sim, reference=ref, control=control)
    monkeypatch.setattr(palm_replay, "load_npz", lambda path: payloads[str(path).split(".")[-2]])
    position = np.broadcast_to(sim_time[:, None, None], (5, 2, 3)).copy()
    rotation = np.broadcast_to(np.eye(3), (5, 2, 3, 3)).copy()
    monkeypatch.setattr(palm_replay, "measured_poses", lambda *args: (position, rotation))
    return payloads


def test_delay_is_applied_in_simulation_time_and_outside_frames_are_invalid(telemetry):
    position, rotation, valid, stability = palm_replay.sample_replay(
        "/replay", [-1., 0, 1, 2, 3], "/robot.xml")
    np.testing.assert_array_equal(valid, [False, True, True, True, False])
    np.testing.assert_allclose(position[valid, 0, 0], [2.24, 4.24, 6.24])
    np.testing.assert_allclose(rotation, np.broadcast_to(np.eye(3), rotation.shape))
    assert stability == dict(min_root_height_m=.8, max_root_tilt_deg=0.)


def test_missing_or_duplicated_control_frame_is_rejected(telemetry):
    telemetry["control"]["replay_frame_index"] = np.array([0, 1, 1])
    with pytest.raises(ValueError, match="exactly once"):
        palm_replay.sample_replay("/replay", [0., 1, 2], "/robot.xml")


def test_nonmonotonic_simulation_clock_is_rejected(telemetry):
    telemetry["sim"]["sim_time_s"][2] = 1.
    with pytest.raises(ValueError, match="Nonmonotonic simulation"):
        palm_replay.sample_replay("/replay", [0., 1, 2], "/robot.xml")
