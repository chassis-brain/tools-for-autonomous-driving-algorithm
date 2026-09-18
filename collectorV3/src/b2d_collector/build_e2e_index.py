from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any, Dict, List


RGB_IDS = ("front", "front_left", "front_right", "back", "back_left", "back_right")


def _load_json_gz(path: Path) -> Dict[str, Any]:
    with gzip.open(str(path), "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _wrap_pi(value: float) -> float:
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def _relative_pose(origin: Dict[str, Any], target: Dict[str, Any]) -> List[float]:
    origin_loc = origin["ego"]["location"]
    target_loc = target["ego"]["location"]
    origin_yaw = math.radians(float(origin["ego"]["rotation"][2]))
    target_yaw = math.radians(float(target["ego"]["rotation"][2]))

    dx = float(target_loc[0]) - float(origin_loc[0])
    dy = float(target_loc[1]) - float(origin_loc[1])
    c = math.cos(origin_yaw)
    s = math.sin(origin_yaw)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    local_heading = _wrap_pi(target_yaw - origin_yaw)
    return [local_x, local_y, local_heading]


def build_index(clip: Path, horizon_s: float, interval_s: float, output: Path) -> int:
    metadata = json.loads((clip / "_collector_meta" / "clip.json").read_text(encoding="utf-8"))
    frequency = float(metadata["frequency_hz"])
    interval_frames = int(round(interval_s * frequency))
    num_poses = int(round(horizon_s / interval_s))
    if interval_frames <= 0 or num_poses <= 0:
        raise ValueError("invalid horizon/interval")

    measurement_paths = sorted((clip / "measurements").glob("*.json.gz"))
    measurements = [_load_json_gz(path) for path in measurement_paths]
    required_future_frames = interval_frames * num_poses

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for index in range(0, max(0, len(measurements) - required_future_frames)):
            current = measurements[index]
            future_indices = [index + interval_frames * step for step in range(1, num_poses + 1)]
            future = [_relative_pose(current, measurements[j]) for j in future_indices]
            stem = "%05d" % index
            sample = {
                "schema": "b2d-base-e2e-index-v1",
                "frame": stem,
                "timestamp": current["timestamp"],
                "rgb": {
                    name: "camera/rgb_%s/%s.jpg" % (name, stem)
                    for name in RGB_IDS
                },
                "lidar": "lidar/%s.laz" % stem if (clip / "lidar" / (stem + ".laz")).exists() else None,
                "measurement": "measurements/%s.json.gz" % stem,
                "annotation": "anno/%s.json.gz" % stem,
                "driving_command": int(current["navigation"]["near_command"]),
                "target_point_ego": current["navigation"]["near_ego_xy"],
                "ego_velocity": current["ego"]["velocity"],
                "ego_acceleration": current["ego"]["acceleration"],
                "future_trajectory": future,
                "trajectory_sampling": {
                    "time_horizon": float(horizon_s),
                    "interval_length": float(interval_s),
                    "num_poses": int(num_poses),
                },
            }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="从连续 Base v1 measurements 构造 E2E future-trajectory index")
    parser.add_argument("clip")
    parser.add_argument("--horizon", type=float, default=4.0, help="future horizon seconds")
    parser.add_argument("--interval", type=float, default=0.5, help="future pose interval seconds")
    parser.add_argument("--output", default="", help="default: <clip>/_collector_meta/e2e_samples.jsonl")
    args = parser.parse_args()

    clip = Path(args.clip).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else clip / "_collector_meta" / "e2e_samples.jsonl"
    count = build_index(clip, args.horizon, args.interval, output)
    print(json.dumps({"clip": str(clip), "output": str(output), "samples": count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
