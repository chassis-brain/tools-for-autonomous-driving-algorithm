#!/usr/bin/env python3
"""Offline preflight for Behavior Tape -> PDM/manual intervention recovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from b2d_collector.failure_replay.config import ReplayCollectorConfig
from b2d_collector.failure_replay.spec import load_intervention_spec
from b2d_collector.failure_replay.tape import BehaviorTape
from b2d_collector.route_xml import RouteCatalog


def _manual_plan_check(plan_path: Path, spec):
    errors = []
    warnings = []
    info = {"plan": str(plan_path), "trajectory": None, "samples": None, "start_gap_m": None, "end_gap_m": None, "speed_min": None, "speed_max": None}
    if not plan_path.is_file():
        errors.append("manual plan JSON is missing: %s" % plan_path)
        return errors, warnings, info

    try:
        doc = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append("manual plan JSON is unreadable: %s" % exc)
        return errors, warnings, info

    candidates = []
    generated = doc.get("generated") or {}
    for value in (
        generated.get("trajectory_file"),
        doc.get("trajectory_file"),
        doc.get("trajectory_npz"),
    ):
        if value:
            p = Path(str(value)).expanduser()
            if not p.is_absolute():
                p = plan_path.parent / p
            candidates.append(p.resolve())
    stem = plan_path.name[:-len(".plan.json")] if plan_path.name.endswith(".plan.json") else plan_path.stem
    candidates.extend([
        (plan_path.parent / (stem + ".trajectory.npz")).resolve(),
        (plan_path.parent / "generated" / (stem + ".trajectory.npz")).resolve(),
    ])

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_file():
            continue
        info["trajectory"] = str(candidate)
        try:
            with np.load(str(candidate)) as data:
                required = ("x", "y", "target_speed")
                missing = [key for key in required if key not in data]
                if missing:
                    errors.append("manual trajectory missing arrays %s: %s" % (missing, candidate))
                    return errors, warnings, info
                n = len(np.asarray(data["x"]).reshape(-1))
                if n < 2 or len(np.asarray(data["y"]).reshape(-1)) != n:
                    errors.append("manual trajectory has invalid x/y lengths: %s" % candidate)
                speeds = np.asarray(data["target_speed"], dtype=float).reshape(-1)
                if len(speeds) != n:
                    errors.append("manual trajectory target_speed length mismatch: %s" % candidate)
                info["samples"] = n
                if len(speeds):
                    info["speed_min"] = float(np.min(speeds))
                    info["speed_max"] = float(np.max(speeds))

                xs = np.asarray(data["x"], dtype=float).reshape(-1)
                ys = np.asarray(data["y"], dtype=float).reshape(-1)
                if n >= 2:
                    if spec.handoff_x is not None and spec.handoff_y is not None:
                        info["start_gap_m"] = float(np.hypot(xs[0] - spec.handoff_x, ys[0] - spec.handoff_y))
                        if info["start_gap_m"] > 2.5:
                            errors.append(
                                "manual trajectory starts %.3fm from case handoff; re-save the plan"
                                % info["start_gap_m"]
                            )
                    if spec.has_spatial_end:
                        info["end_gap_m"] = float(np.hypot(xs[-1] - spec.end_x, ys[-1] - spec.end_y))
                        if info["end_gap_m"] > 2.5:
                            errors.append(
                                "manual trajectory ends %.3fm from case End; re-save the plan"
                                % info["end_gap_m"]
                            )
        except Exception as exc:
            errors.append("manual trajectory NPZ is unreadable: %s" % exc)
        return errors, warnings, info

    # Runtime can regenerate dense P(s)+v(s) if the edited control points are in JSON.
    path_points = doc.get("path_control_points") or doc.get("anchors")
    speed_points = doc.get("speed_control_points")
    if isinstance(path_points, list) and len(path_points) >= 2 and isinstance(speed_points, list) and len(speed_points) >= 2:
        warnings.append("trajectory NPZ not found; runtime will regenerate it from plan control points")
        def xy(item):
            if isinstance(item, dict):
                return float(item["x"]), float(item["y"])
            return float(item[0]), float(item[1])
        try:
            sx, sy = xy(path_points[0])
            ex, ey = xy(path_points[-1])
            if spec.handoff_x is not None and spec.handoff_y is not None:
                info["start_gap_m"] = float(np.hypot(sx - spec.handoff_x, sy - spec.handoff_y))
                if info["start_gap_m"] > 2.5:
                    errors.append("manual plan starts %.3fm from case handoff" % info["start_gap_m"])
            if spec.has_spatial_end:
                info["end_gap_m"] = float(np.hypot(ex - spec.end_x, ey - spec.end_y))
                if info["end_gap_m"] > 2.5:
                    errors.append("manual plan ends %.3fm from case End" % info["end_gap_m"])
        except Exception as exc:
            errors.append("manual path control points are invalid: %s" % exc)
        return errors, warnings, info

    errors.append(
        "manual plan has no usable trajectory NPZ and lacks path/speed control points: %s" % plan_path
    )
    return errors, warnings, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight a Replay recovery case")
    parser.add_argument("config")
    args = parser.parse_args()

    config = ReplayCollectorConfig.load(args.config)
    tape = BehaviorTape(config.replay.tape_dir)
    spec = load_intervention_spec(config.replay.intervention_spec, tape)
    route = RouteCatalog(config.route_xml).get(config.route_id)

    errors = []
    warnings = []
    complete = (tape.run_dir / "COMPLETE").is_file()
    manifest_path = tape.run_dir / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append("Probe manifest is unreadable: %s" % exc)
    elif config.replay.require_complete_tape:
        errors.append("Probe manifest.json is missing")

    result = manifest.get("result") or {}
    if config.replay.require_complete_tape and not complete:
        errors.append("Probe COMPLETE marker is missing")
    if config.replay.require_complete_tape and result and not bool(result.get("integrity_ok", False)):
        errors.append("Probe manifest result.integrity_ok is not true")

    tolerance = tape.dt * 0.51
    if spec.record_start_time < tape.first_time - tolerance:
        errors.append("record_start is before the first Tape sample")
    if spec.handoff_time > tape.last_time + tolerance:
        errors.append("handoff is after the end of the Tape")
    if spec.record_end_time is not None and spec.record_end_time > tape.last_time + tolerance:
        warnings.append(
            "record_end is after the original Tape; this is allowed because replacement owns control after handoff"
        )

    plan_info = None
    if spec.replacement_mode == "manual_pid":
        if not spec.replacement_plan:
            errors.append("manual_pid selected but replacement.plan is empty")
        else:
            pe, pw, plan_info = _manual_plan_check(Path(spec.replacement_plan), spec)
            errors.extend(pe)
            warnings.extend(pw)

    print("config             =", Path(args.config).expanduser().resolve())
    print("case               =", Path(config.replay.intervention_spec).expanduser().resolve())
    print("route              = id=%s town=%s" % (route.id, route.town))
    print("tape               =", tape.run_dir)
    print("samples            =", len(tape.frames))
    print("sim_time           = %.3f .. %.3f (dt %.3f)" % (tape.first_time, tape.last_time, tape.dt))
    print("COMPLETE           =", complete)
    if result:
        print("probe_integrity     =", result.get("integrity_ok"))
    print("record_start       = %.3f" % spec.record_start_time)
    print("handoff            = %.3f" % spec.handoff_time)
    if spec.has_spatial_end:
        print("end_xy             = (%.3f, %.3f)" % (spec.end_x, spec.end_y))
        print("end_gate           = radius %.2fm x %d frames" % (spec.end_radius_m, spec.end_confirm_frames))
    else:
        print("record_end         = %.3f" % float(spec.record_end_time))
    print("replacement_mode   =", spec.replacement_mode)
    if spec.replacement_plan:
        print("replacement_plan   =", spec.replacement_plan)
    if plan_info:
        print("trajectory_file    =", plan_info.get("trajectory"))
        print("trajectory_samples =", plan_info.get("samples"))
        if plan_info.get("start_gap_m") is not None:
            print("plan_start_gap_m   = %.3f" % plan_info["start_gap_m"])
        if plan_info.get("end_gap_m") is not None:
            print("plan_end_gap_m     = %.3f" % plan_info["end_gap_m"])
        if plan_info.get("speed_min") is not None:
            print("target_speed_mps   = %.3f .. %.3f" % (plan_info["speed_min"], plan_info["speed_max"]))
    print("time_offset        = %+.3f" % spec.time_offset_seconds)
    print("shadow_pdm         =", bool(config.replay.shadow_pdm and spec.replacement_mode == "pdm_expert"))
    print("sensor_profile     =", config.sensor_profile)

    for warning in warnings:
        print("WARNING            =", warning)
    for error in errors:
        print("ERROR              =", error)

    print("RESULT             =", "PASS" if not errors else "FAIL")
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
