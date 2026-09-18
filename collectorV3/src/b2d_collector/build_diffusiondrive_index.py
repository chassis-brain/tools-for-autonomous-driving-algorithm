from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .dataset_profiles import DatasetProfile


def _load_json_gz(path: Path) -> Dict[str, Any]:
    with gzip.open(str(path), "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _wrap_pi(value: float) -> float:
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def _world_xy_to_ego(x: float, y: float, measurement: Dict[str, Any]) -> List[float]:
    ego = measurement["ego"]
    loc = ego["location"]
    yaw = math.radians(float(ego["rotation"][2]))
    dx = float(x) - float(loc[0])
    dy = float(y) - float(loc[1])
    c = math.cos(yaw)
    s = math.sin(yaw)
    return [c * dx + s * dy, -s * dx + c * dy]


def _world_vector_to_ego(vector: Sequence[float], measurement: Dict[str, Any]) -> List[float]:
    yaw = math.radians(float(measurement["ego"]["rotation"][2]))
    vx = float(vector[0]) if len(vector) > 0 else 0.0
    vy = float(vector[1]) if len(vector) > 1 else 0.0
    c = math.cos(yaw)
    s = math.sin(yaw)
    return [c * vx + s * vy, -s * vx + c * vy]


def _relative_pose(origin: Dict[str, Any], target: Dict[str, Any]) -> List[float]:
    local_xy = _world_xy_to_ego(
        float(target["ego"]["location"][0]),
        float(target["ego"]["location"][1]),
        origin,
    )
    origin_yaw = math.radians(float(origin["ego"]["rotation"][2]))
    target_yaw = math.radians(float(target["ego"]["rotation"][2]))
    return [local_xy[0], local_xy[1], _wrap_pi(target_yaw - origin_yaw)]


def _command_one_hot(command: int, profile: DatasetProfile) -> List[float]:
    config = profile.driving_command
    classes = list(config.get("classes") or ["left", "straight", "right", "other"])
    mapping = dict(config.get("carla_road_option_map") or {})
    label = mapping.get(str(int(command)), "other")
    if label not in classes:
        label = "other" if "other" in classes else classes[-1]
    return [1.0 if item == label else 0.0 for item in classes]


def _agent_targets(
    annotation: Dict[str, Any],
    measurement: Dict[str, Any],
    profile: DatasetProfile,
) -> Tuple[List[List[float]], List[bool]]:
    preprocess = profile.lidar_preprocess
    min_x = float(preprocess.get("min_x", preprocess.get("lidar_min_x", -32.0)))
    max_x = float(preprocess.get("max_x", preprocess.get("lidar_max_x", 32.0)))
    min_y = float(preprocess.get("min_y", preprocess.get("lidar_min_y", -32.0)))
    max_y = float(preprocess.get("max_y", preprocess.get("lidar_max_y", 32.0)))
    max_agents = int((profile.targets or {}).get("num_bounding_boxes", 30))

    ego_yaw = math.radians(float(measurement["ego"]["rotation"][2]))
    candidates = []

    for box in annotation.get("bounding_boxes", []):
        if box.get("class") != "vehicle":
            continue
        if "world2ego" in box:
            continue

        center = box.get("center") or box.get("location") or [0.0, 0.0, 0.0]
        local_xy = _world_xy_to_ego(float(center[0]), float(center[1]), measurement)
        x, y = local_xy
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            continue

        rotation = box.get("rotation") or [0.0, 0.0, 0.0]
        actor_yaw = math.radians(float(rotation[2]))
        heading = _wrap_pi(actor_yaw - ego_yaw)

        extent = box.get("extent") or [0.0, 0.0, 0.0]
        length = 2.0 * float(extent[0])
        width = 2.0 * float(extent[1])
        state = [x, y, heading, length, width]
        candidates.append((math.hypot(x, y), state))

    candidates.sort(key=lambda item: item[0])
    selected = [item[1] for item in candidates[:max_agents]]

    states = [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(max_agents)]
    labels = [False for _ in range(max_agents)]
    for index, state in enumerate(selected):
        states[index] = state
        labels[index] = True
    return states, labels


def build_index(clip: Path, profile: DatasetProfile, output: Path) -> int:
    metadata = json.loads(
        (clip / "_collector_meta" / "clip.json").read_text(encoding="utf-8")
    )
    frequency = float(metadata["frequency_hz"])
    interval_frames = int(round(profile.interval_s * frequency))
    required_future_frames = interval_frames * profile.num_poses
    if interval_frames <= 0:
        raise ValueError("invalid interval_frames")

    measurement_paths = sorted((clip / "measurements").glob("*.json.gz"))
    anno_paths = sorted((clip / "anno").glob("*.json.gz"))
    if len(measurement_paths) != len(anno_paths):
        raise RuntimeError(
            "measurement/anno frame counts differ: %d vs %d"
            % (len(measurement_paths), len(anno_paths))
        )

    measurements = [_load_json_gz(path) for path in measurement_paths]
    annotations = [_load_json_gz(path) for path in anno_paths]

    for sensor_id in profile.rgb_ids:
        name = sensor_id.replace("CAM_", "").lower()
        directory = clip / "camera" / ("rgb_%s" % name)
        if not directory.exists():
            raise RuntimeError("missing required camera directory: %s" % directory)

    if "LIDAR_TOP" in profile.lidar_ids and not (clip / "lidar").exists():
        raise RuntimeError("missing required LiDAR directory: %s" % (clip / "lidar"))

    output.parent.mkdir(parents=True, exist_ok=True)
    sample_count = 0

    with output.open("w", encoding="utf-8") as handle:
        stop = max(0, len(measurements) - required_future_frames)
        for index in range(stop):
            current = measurements[index]
            annotation = annotations[index]
            future_indices = [
                index + interval_frames * step
                for step in range(1, profile.num_poses + 1)
            ]
            future_trajectory = [
                _relative_pose(current, measurements[j])
                for j in future_indices
            ]

            command_source = str(profile.driving_command.get("source", "near_command"))
            if command_source not in ("near_command", "far_command"):
                raise ValueError(
                    "driving_command.source must be near_command or far_command"
                )
            command = int(current["navigation"][command_source])
            command_one_hot = _command_one_hot(command, profile)

            velocity_ego = _world_vector_to_ego(current["ego"]["velocity"], current)
            acceleration_ego = _world_vector_to_ego(
                current["ego"]["acceleration"], current
            )

            stem = "%05d" % index
            camera_paths = {}
            for sensor_id in profile.rgb_ids:
                name = sensor_id.replace("CAM_", "").lower()
                camera_paths[sensor_id] = "camera/rgb_%s/%s.jpg" % (name, stem)

            agents, agent_labels = _agent_targets(annotation, current, profile)

            sample = {
                "schema": "diffusiondrive-carla-v1",
                "frame": stem,
                "timestamp": float(current["timestamp"]),
                "inputs": {
                    "cameras": camera_paths,
                    "lidar": (
                        "lidar/%s.laz" % stem
                        if "LIDAR_TOP" in profile.lidar_ids
                        else None
                    ),
                    "ego_status": {
                        "driving_command_raw": command,
                        "driving_command": command_one_hot,
                        "ego_velocity": velocity_ego,
                        "ego_acceleration": acceleration_ego,
                    },
                    "measurement": "measurements/%s.json.gz" % stem,
                    "camera_preprocess": profile.camera_preprocess,
                    "lidar_preprocess": profile.lidar_preprocess,
                },
                "targets": {
                    "trajectory": future_trajectory,
                    "agent_states": agents,
                    "agent_labels": agent_labels,
                    "bev_semantic_map": None,
                },
                "trajectory_sampling": {
                    "time_horizon": profile.horizon_s,
                    "interval_length": profile.interval_s,
                    "num_poses": profile.num_poses,
                },
            }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            sample_count += 1

    return sample_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DiffusionDrive-CARLA v1 manifest from Base v1 raw clip"
    )
    parser.add_argument("clip")
    parser.add_argument(
        "--sensor-config",
        default="",
        help="DiffusionDrive dataset-profile JSON; default B2D_SENSOR_CONFIG",
    )
    parser.add_argument(
        "--output",
        default="",
        help="default: <clip>/_collector_meta/diffusiondrive_samples.jsonl",
    )
    args = parser.parse_args()

    clip = Path(args.clip).expanduser().resolve()
    config = args.sensor_config or os.environ.get("B2D_SENSOR_CONFIG", "")
    profile = DatasetProfile.load("diffusiondrive", config)

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else clip / "_collector_meta" / "diffusiondrive_samples.jsonl"
    )
    count = build_index(clip, profile, output)
    print(
        json.dumps(
            {
                "clip": str(clip),
                "profile": profile.metadata(),
                "output": str(output),
                "samples": count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
