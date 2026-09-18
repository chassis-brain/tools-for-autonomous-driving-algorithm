#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from b2d_collector.failure_replay.tape import BehaviorTape


def yaw_error(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Compare reference and replay Probe trajectories by simulation time")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--max-time", type=float, default=None, help="Relative seconds to compare")
    args = parser.parse_args()

    ref = BehaviorTape(args.ref)
    rep = BehaviorTape(args.replay)
    pos = []
    yaw = []
    throttle = []
    steer = []
    brake = []
    samples = 0

    for a in ref.frames:
        if args.max_time is not None and a.sim_time - ref.first_time > args.max_time + 1e-9:
            break
        b = rep.nearest(a.sim_time)
        if abs(b.sim_time - a.sim_time) > max(ref.dt, rep.dt) * 0.51:
            continue
        dx = a.ego_location[0] - b.ego_location[0]
        dy = a.ego_location[1] - b.ego_location[1]
        dz = a.ego_location[2] - b.ego_location[2]
        pos.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        yaw.append(yaw_error(a.ego_rotation[2], b.ego_rotation[2]))
        throttle.append(abs(float(a.control.get("throttle", 0.0)) - float(b.control.get("throttle", 0.0))))
        steer.append(abs(float(a.control.get("steer", 0.0)) - float(b.control.get("steer", 0.0))))
        brake.append(abs(float(a.control.get("brake", 0.0)) - float(b.control.get("brake", 0.0))))
        samples += 1

    print("matched_samples =", samples)
    if not samples:
        raise SystemExit(2)
    print("position_error_m mean=%.6f max=%.6f end=%.6f" % (mean(pos), max(pos), pos[-1]))
    print("yaw_error_deg    mean=%.6f max=%.6f" % (mean(yaw), max(yaw)))
    print("control_mae      throttle=%.6f steer=%.6f brake=%.6f" % (mean(throttle), mean(steer), mean(brake)))

    for t in (2.0, 5.0, 10.0):
        if args.max_time is not None and t > args.max_time:
            continue
        target = ref.first_time + t
        a = ref.nearest(target)
        b = rep.nearest(target)
        dx = a.ego_location[0] - b.ego_location[0]
        dy = a.ego_location[1] - b.ego_location[1]
        dz = a.ego_location[2] - b.ego_location[2]
        print("position_error@%.1fs = %.6f" % (t, math.sqrt(dx * dx + dy * dy + dz * dz)))


if __name__ == "__main__":
    main()
