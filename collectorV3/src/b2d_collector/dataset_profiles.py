from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_RGB_ALIASES = {
    "FRONT": "CAM_FRONT",
    "FRONT_LEFT": "CAM_FRONT_LEFT",
    "FRONT_RIGHT": "CAM_FRONT_RIGHT",
    "BACK": "CAM_BACK",
    "BACK_LEFT": "CAM_BACK_LEFT",
    "BACK_RIGHT": "CAM_BACK_RIGHT",
    "CAM_FRONT": "CAM_FRONT",
    "CAM_FRONT_LEFT": "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT": "CAM_FRONT_RIGHT",
    "CAM_BACK": "CAM_BACK",
    "CAM_BACK_LEFT": "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT": "CAM_BACK_RIGHT",
}

_LIDAR_ALIASES = {
    "LIDAR": "LIDAR_TOP",
    "LIDAR_TOP": "LIDAR_TOP",
}


def _normalize_rgb(value: str) -> str:
    key = str(value).strip().upper()
    if key not in _RGB_ALIASES:
        raise ValueError("Unsupported RGB sensor in dataset profile: %s" % value)
    return _RGB_ALIASES[key]


def _normalize_lidar(value: str) -> str:
    key = str(value).strip().upper()
    if key not in _LIDAR_ALIASES:
        raise ValueError("Unsupported LiDAR sensor in dataset profile: %s" % value)
    return _LIDAR_ALIASES[key]


@dataclass
class DatasetProfile:
    name: str
    version: int
    source_path: Optional[Path]
    raw: Dict[str, Any]
    rgb_ids: Tuple[str, ...]
    lidar_ids: Tuple[str, ...]
    horizon_s: float
    interval_s: float
    num_poses: int
    camera_preprocess: Dict[str, Any]
    lidar_preprocess: Dict[str, Any]
    features: Dict[str, Any]
    targets: Dict[str, Any]
    driving_command: Dict[str, Any]

    @classmethod
    def base(cls) -> "DatasetProfile":
        return cls(
            name="base",
            version=1,
            source_path=None,
            raw={},
            rgb_ids=(
                "CAM_FRONT",
                "CAM_FRONT_LEFT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ),
            lidar_ids=(),
            horizon_s=4.0,
            interval_s=0.5,
            num_poses=8,
            camera_preprocess={},
            lidar_preprocess={},
            features={},
            targets={},
            driving_command={},
        )

    @classmethod
    def from_environment(cls) -> "DatasetProfile":
        name = os.environ.get("B2D_DATASET_PROFILE", "base").strip().lower() or "base"
        config = os.environ.get("B2D_SENSOR_CONFIG", "").strip()
        return cls.load(name, config)

    @classmethod
    def load(cls, name: str, config_path: str = "") -> "DatasetProfile":
        name = str(name or "base").strip().lower()
        if name == "base":
            return cls.base()
        if name != "diffusiondrive":
            raise ValueError("Unsupported dataset profile: %s" % name)
        if not config_path:
            raise ValueError(
                "diffusiondrive profile requires --sensor-config / B2D_SENSOR_CONFIG"
            )

        source = Path(config_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError("dataset profile JSON not found: %s" % source)

        with source.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        declared_name = str(raw.get("name", "diffusiondrive")).strip().lower()
        if "diffusiondrive" not in declared_name:
            raise ValueError(
                "sensor config name must identify diffusiondrive, got: %s" % declared_name
            )

        sensors = raw.get("sensors") or {}
        rgb_ids = tuple(_normalize_rgb(item) for item in sensors.get("rgb", []))
        lidar_ids = tuple(_normalize_lidar(item) for item in sensors.get("lidar", []))

        required_rgb = {"CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"}
        if not required_rgb.issubset(set(rgb_ids)):
            raise ValueError(
                "DiffusionDrive v1 requires RGB sensors: %s"
                % ", ".join(sorted(required_rgb))
            )
        if "LIDAR_TOP" not in lidar_ids:
            raise ValueError("DiffusionDrive v1 requires LIDAR_TOP")

        trajectory = raw.get("trajectory") or {}
        horizon_s = float(trajectory.get("time_horizon", 4.0))
        interval_s = float(trajectory.get("interval_length", 0.5))
        num_poses = int(
            trajectory.get(
                "num_poses",
                round(horizon_s / interval_s) if interval_s > 0 else 0,
            )
        )
        if horizon_s <= 0 or interval_s <= 0 or num_poses <= 0:
            raise ValueError("invalid diffusiondrive trajectory sampling")
        if abs(num_poses * interval_s - horizon_s) > 1e-6:
            raise ValueError(
                "trajectory num_poses * interval_length must equal time_horizon"
            )

        targets = dict(raw.get("targets") or {})
        if bool(targets.get("bev_semantic_map", False)):
            raise ValueError(
                "DiffusionDrive CARLA profile v1 does not materialize NAVSIM BEV semantic "
                "map yet. Set targets.bev_semantic_map=false for v1."
            )

        driving_command = dict(raw.get("driving_command") or {})
        if not driving_command:
            driving_command = {
                "classes": ["left", "straight", "right", "other"],
                "carla_road_option_map": {
                    "1": "left",
                    "2": "right",
                    "3": "straight",
                    "4": "straight",
                    "5": "other",
                    "6": "other",
                    "-1": "other",
                },
                "source": "near_command",
                "note": (
                    "CARLA RoadOption -> one-hot mapping. Keep this explicit in JSON "
                    "when reproducing experiments."
                ),
            }

        return cls(
            name="diffusiondrive",
            version=int(raw.get("version", 1)),
            source_path=source,
            raw=raw,
            rgb_ids=rgb_ids,
            lidar_ids=lidar_ids,
            horizon_s=horizon_s,
            interval_s=interval_s,
            num_poses=num_poses,
            camera_preprocess=dict(raw.get("camera_preprocess") or {}),
            lidar_preprocess=dict(raw.get("lidar_preprocess") or {}),
            features=dict(raw.get("features") or {}),
            targets=targets,
            driving_command=driving_command,
        )

    def validate_collection_sensor_profile(self, sensor_profile: str) -> None:
        if self.name == "diffusiondrive" and sensor_profile != "full":
            raise ValueError(
                "DiffusionDrive v1 requires Collector sensor_profile: full "
                "(LiDAR is required)."
            )

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "source_path": str(self.source_path) if self.source_path is not None else None,
            "sensors": {
                "rgb": list(self.rgb_ids),
                "lidar": list(self.lidar_ids),
            },
            "trajectory": {
                "time_horizon": self.horizon_s,
                "interval_length": self.interval_s,
                "num_poses": self.num_poses,
            },
            "camera_preprocess": self.camera_preprocess,
            "lidar_preprocess": self.lidar_preprocess,
            "features": self.features,
            "targets": self.targets,
            "driving_command": self.driving_command,
            "raw_config": self.raw,
            "collection_strategy": (
                "raw_superset_profile_manifest_v1"
                if self.name != "base"
                else "base"
            ),
        }
