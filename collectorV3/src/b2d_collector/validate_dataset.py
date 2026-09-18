from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, List, Tuple


RGB_IDS = ("front", "front_left", "front_right", "back", "back_left", "back_right")

PRIMARY_RGB_CALIBRATION_IDS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

REQUIRED_CAMERA_CALIBRATION = {
    "location",
    "rotation",
    "intrinsic",
    "world2cam",
    "cam2ego",
    "fov",
    "image_size_x",
    "image_size_y",
}

FORBIDDEN_CALIBRATION_IDS = {
    "front",
    "default",
}
REQUIRED_ANNO = {
    "x", "y", "throttle", "steer", "brake", "reverse", "theta", "speed",
    "x_command_far", "y_command_far", "command_far", "x_command_near",
    "y_command_near", "command_near", "weather", "acceleration",
    "angular_velocity", "sensors", "bounding_boxes",
}
REQUIRED_MEASUREMENT = {
    "schema", "frame_index", "timestamp", "sensor_frames", "ego", "control",
    "navigation", "collector",
}
FORBIDDEN_ANNO = {"control_source", "failure_segment", "timestamp"}


def _stems(directory: Path, pattern: str) -> List[str]:
    return [item.name.split(".")[0] for item in sorted(directory.glob(pattern))]


def _compare_stems(
    errors: List[str], counts: Dict[str, int], clip: Path, relative: str,
    pattern: str, expected: List[str],
) -> None:
    directory = clip / relative
    actual = _stems(directory, pattern) if directory.exists() else []
    counts[relative] = len(actual)
    if actual != expected:
        errors.append("%s 帧集合与 anno 不一致: %d vs %d" % (relative, len(actual), len(expected)))


def validate_clip(path: Path) -> Tuple[List[str], Dict[str, int]]:
    errors: List[str] = []
    counts: Dict[str, int] = {}

    metadata_path = path / "_collector_meta" / "clip.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append("clip metadata 无法读取: %s" % exc)
    else:
        errors.append("缺少 _collector_meta/clip.json")

    anno_files = sorted((path / "anno").glob("*.json.gz")) if (path / "anno").exists() else []
    counts["anno"] = len(anno_files)
    if not anno_files:
        return ["缺少 anno/*.json.gz"] + errors, counts

    expected = ["%05d" % index for index in range(len(anno_files))]
    actual = [item.name.split(".")[0] for item in anno_files]
    if actual != expected:
        errors.append("anno 帧号不连续")

    missing_primary_rgb = set()
    missing_primary_rgb_fields = set()

    for item in anno_files:
        try:
            with gzip.open(str(item), "rt", encoding="utf-8") as handle:
                data = json.load(handle)

            missing = REQUIRED_ANNO - set(data)
            forbidden = FORBIDDEN_ANNO & set(data)

            if missing:
                errors.append(
                    "%s 缺字段: %s"
                    % (item.name, sorted(missing))
                )

            if forbidden:
                errors.append(
                    "%s anno 混入 collector 私有字段: %s"
                    % (item.name, sorted(forbidden))
                )

            sensors = data.get("sensors", {})

            if not isinstance(sensors, dict):
                errors.append(
                    "%s sensors 不是字典"
                    % item.name
                )
                continue

            forbidden_calibration = (
                FORBIDDEN_CALIBRATION_IDS
                & set(sensors)
            )

            if forbidden_calibration:
                errors.append(
                    "%s calibration 含非 canonical alias: %s"
                    % (
                        item.name,
                        sorted(forbidden_calibration),
                    )
                )

            for sensor_id in PRIMARY_RGB_CALIBRATION_IDS:
                calibration = sensors.get(sensor_id)

                if not isinstance(calibration, dict):
                    missing_primary_rgb.add(sensor_id)
                    continue

                missing_fields = (
                    REQUIRED_CAMERA_CALIBRATION
                    - set(calibration)
                )

                for field in missing_fields:
                    missing_primary_rgb_fields.add(
                        (sensor_id, field)
                    )

        except Exception as exc:
            errors.append(
                "%s 无法读取: %s"
                % (item.name, exc)
            )

    if missing_primary_rgb:
        errors.append(
            "主 RGB calibration 缺少 canonical sensors: %s"
            % sorted(missing_primary_rgb)
        )

    if missing_primary_rgb_fields:
        errors.append(
            "主 RGB calibration 缺字段: %s"
            % [
                "%s.%s" % pair
                for pair
                in sorted(missing_primary_rgb_fields)
            ]
        )

    measurement_files = sorted((path / "measurements").glob("*.json.gz")) if (path / "measurements").exists() else []
    measurement_stems = [item.name.split(".")[0] for item in measurement_files]
    counts["measurements"] = len(measurement_files)
    if measurement_stems != expected:
        errors.append("measurements 帧集合与 anno 不一致")
    for item in measurement_files:
        try:
            with gzip.open(str(item), "rt", encoding="utf-8") as handle:
                data = json.load(handle)
            missing = REQUIRED_MEASUREMENT - set(data)
            if missing:
                errors.append("%s measurement 缺字段: %s" % (item.name, sorted(missing)))
            for key in ("location", "rotation", "velocity", "acceleration", "speed"):
                if key not in data.get("ego", {}):
                    errors.append("%s ego 缺字段 %s" % (item.name, key))
        except Exception as exc:
            errors.append("%s measurement 无法读取: %s" % (item.name, exc))

    for name in RGB_IDS:
        _compare_stems(errors, counts, path, "camera/rgb_%s" % name, "*.jpg", expected)

    profile = metadata.get("sensor_profile")
    if profile == "full":
        for name in RGB_IDS:
            _compare_stems(errors, counts, path, "camera/depth_%s" % name, "*.npz", expected)
            _compare_stems(errors, counts, path, "camera/semantic_%s" % name, "*.png", expected)
            _compare_stems(errors, counts, path, "camera/instance_%s" % name, "*.png", expected)
        _compare_stems(errors, counts, path, "camera/rgb_top_down", "*.jpg", expected)
        _compare_stems(errors, counts, path, "lidar", "*.laz", expected)
        _compare_stems(errors, counts, path, "radar", "*.h5", expected)

    return errors, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 b2d-base-e2e-v1 单个 clip")
    parser.add_argument("clip")
    args = parser.parse_args()
    clip = Path(args.clip).expanduser().resolve()
    errors, counts = validate_clip(clip)
    print(json.dumps({"clip": str(clip), "counts": counts, "errors": errors}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
