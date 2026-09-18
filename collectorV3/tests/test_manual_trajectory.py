import json
from pathlib import Path

import numpy as np

from b2d_collector.experts.manual_trajectory import ManualTrajectoryFollower


def _make_plan(tmp_path: Path):
    generated = tmp_path / "path.npz"
    s = np.linspace(0.0, 20.0, 101)
    np.savez_compressed(
        str(generated),
        s=s,
        x=s,
        y=np.zeros_like(s),
        yaw=np.zeros_like(s),
        curvature=np.zeros_like(s),
        target_speed=np.full_like(s, 5.0),
    )
    plan = tmp_path / "case.plan.json"
    plan.write_text(json.dumps({
        "schema": "b2d-manual-plan-v2",
        "generated": {"trajectory_file": generated.name},
        "controller": {
            "lateral_kp": 0.45,
            "lateral_ki": 0.0,
            "lateral_kd": 0.0,
            "heading_kp": 1.0,
            "curvature_feedforward": 0.85,
            "speed_kp": 0.4,
            "speed_ki": 0.0,
            "speed_kd": 0.0,
        },
    }))
    return plan


def test_manual_pid_tracks_straight_path_and_speed(tmp_path):
    follower = ManualTrajectoryFollower(str(_make_plan(tmp_path)))
    throttle, steer, brake, diag = follower.compute(
        x=1.0, y=0.0, yaw_rad=0.0, speed_mps=2.0, world_time=1.0
    )
    assert abs(steer) < 1e-6
    assert throttle > 0.0
    assert brake == 0.0
    assert diag["target_speed_mps"] == 5.0


def test_manual_pid_corrects_lateral_error(tmp_path):
    follower = ManualTrajectoryFollower(str(_make_plan(tmp_path)))
    _, steer, _, _ = follower.compute(
        x=1.0, y=1.0, yaw_rad=0.0, speed_mps=3.0, world_time=1.0
    )
    assert abs(steer) > 0.05
