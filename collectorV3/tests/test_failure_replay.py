import json
from pathlib import Path

from b2d_collector.failure_replay.spec import load_intervention_spec
from b2d_collector.failure_replay.tape import BehaviorTape


def _frame(index, t, x=0.0, throttle=0.0):
    return {
        "schema": "e2e_probe_frame_v1",
        "world": {"frame": 100 + index, "elapsed_seconds": t, "delta_seconds": 0.1},
        "ego": {
            "state": {
                "transform": {
                    "location": {"x": x, "y": 0.0, "z": 0.0},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
                "velocity": {"x": 1.0, "y": 0.0, "z": 0.0},
                "acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            "applied_control": {
                "throttle": throttle,
                "steer": 0.0,
                "brake": 0.0,
                "reverse": False,
                "hand_brake": False,
                "gear": 1,
                "manual_gear_shift": False,
            },
        },
    }


def _make_tape(tmp_path: Path):
    run = tmp_path / "probe"
    run.mkdir()
    with (run / "frames.jsonl").open("w") as handle:
        for i in range(30):
            handle.write(json.dumps(_frame(i, 0.7 + 0.1 * i, x=i * 0.1, throttle=1.0 if i >= 15 else 0.0)) + "\n")
    return BehaviorTape(str(run))


def test_timestamp_indexed_replay_preserves_tape_onset(tmp_path):
    tape = _make_tape(tmp_path)
    assert tape.frame_for_replay_time(0.1) is None
    selected = tape.frame_for_replay_time(2.2)
    assert selected is not None
    assert selected.sample_index == 15
    assert abs(selected.sim_time - 2.2) < 1e-6
    assert selected.control["throttle"] == 1.0


def test_current_intervention_spec(tmp_path):
    tape = _make_tape(tmp_path)
    path = tmp_path / "case.json"
    path.write_text(json.dumps({
        "schema": "b2d-intervention-v1",
        "case_id": "case_a",
        "intervention_type": "slow_decision",
        "record_start": {"sample_index": 5},
        "handoff": {"relative_time_s": 1.0},
        "record_end": {"sim_time_s": 2.7},
    }))
    spec = load_intervention_spec(str(path), tape)
    assert abs(spec.record_start_time - 1.2) < 1e-6
    assert abs(spec.handoff_time - 1.7) < 1e-6
    assert abs(spec.record_end_time - 2.7) < 1e-6


def test_legacy_v13_spec(tmp_path):
    tape = _make_tape(tmp_path)
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "schema": "intervention_spec_v1_3",
        "record_start_step": 3,
        "handoff_step": 7,
        "record_end_step": 12,
    }))
    spec = load_intervention_spec(str(path), tape)
    assert abs(spec.record_start_time - 1.0) < 1e-6
    assert abs(spec.handoff_time - 1.4) < 1e-6
    assert abs(spec.record_end_time - 1.9) < 1e-6


def test_v2_three_point_spatial_end_and_manual_plan(tmp_path):
    tape = _make_tape(tmp_path)
    case_dir = tmp_path / "cases"
    (case_dir / "plans").mkdir(parents=True)
    plan = case_dir / "plans" / "case_manual.plan.json"
    plan.write_text("{}")
    path = case_dir / "case_manual.json"
    path.write_text(json.dumps({
        "schema": "b2d-intervention-v2",
        "case_id": "case_manual",
        "source_run": str(tape.run_dir),
        "record_start": {"sample_index": 5, "x": 0.5, "y": 0.0, "z": 0.0},
        "handoff": {"sample_index": 10, "x": 1.0, "y": 0.0, "z": 0.0},
        "end": {"x": 12.5, "y": -4.0, "radius_m": 1.6, "confirm_frames": 4},
        "replacement": {"mode": "manual_pid", "plan": "plans/case_manual.plan.json"},
        "replay": {"time_offset_seconds": 0.0},
    }))
    spec = load_intervention_spec(str(path), tape)
    assert abs(spec.record_start_time - 1.2) < 1e-6
    assert abs(spec.handoff_time - 1.7) < 1e-6
    assert spec.record_end_time is None
    assert spec.end_xy == (12.5, -4.0)
    assert abs(spec.end_radius_m - 1.6) < 1e-9
    assert spec.end_confirm_frames == 4
    assert spec.replacement_mode == "manual_pid"
    assert Path(spec.manual_plan_path) == plan.resolve()
