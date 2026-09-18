from __future__ import annotations

from typing import Dict, List


CAMERAS = (
    ("FRONT", 0.80, 0.0, 1.60, 0.0, 70),
    ("FRONT_LEFT", 0.27, -0.55, 1.60, -55.0, 70),
    ("FRONT_RIGHT", 0.27, 0.55, 1.60, 55.0, 70),
    ("BACK", -2.0, 0.0, 1.60, 180.0, 110),
    ("BACK_LEFT", -0.32, -0.55, 1.60, -110.0, 70),
    ("BACK_RIGHT", -0.32, 0.55, 1.60, 110.0, 70),
)


def _camera(kind: str, name: str, x: float, y: float, z: float, yaw: float, fov: int) -> Dict:
    suffix = {"rgb": "", "depth": "_DEPTH", "semantic_segmentation": "_SEM_SEG", "instance_segmentation": "_INS_SEG"}[kind]
    return {
        "type": "sensor.camera.%s" % kind,
        "x": x,
        "y": y,
        "z": z,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": yaw,
        "width": 1600,
        "height": 900,
        "fov": fov,
        "id": "CAM_%s%s" % (name, suffix),
    }


def collection_sensors(profile: str = "full") -> List[Dict]:
    sensors: List[Dict] = [_camera("rgb", *camera) for camera in CAMERAS]
    sensors += [
        {"type": "sensor.other.gnss", "x": -1.4, "y": 0.0, "z": 0.0, "id": "GPS"},
        {"type": "sensor.other.imu", "x": -1.4, "y": 0.0, "z": 0.0, "roll": 0.0,"pitch": 0.0,"yaw": 0.0,"sensor_tick": 0.05, "id": "IMU"},
        {"type": "sensor.speedometer", "reading_frequency": 20, "id": "SPEED"},
    ]
    if profile == "camera_only":
        return sensors
    # Rich training sensors are spawned by AuxiliarySensorRig after Leaderboard validation.
    return sensors


def auxiliary_sensor_specs(profile: str = "full") -> List[Dict]:
    """Privileged training sensors spawned after the Leaderboard sensor check."""
    if profile != "full":
        return []
    sensors: List[Dict] = []
    for kind in ("depth", "semantic_segmentation", "instance_segmentation"):
        sensors += [_camera(kind, *camera) for camera in CAMERAS]
    sensors += [
        {
            "type": "sensor.lidar.ray_cast", "x": -0.39, "y": 0.0, "z": 1.84,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "range": 85,
            "rotation_frequency": 10, "channels": 64, "points_per_second": 600000,
            "dropoff_general_rate": 0.0, "dropoff_intensity_limit": 0.0,
            "dropoff_zero_intensity": 0.0, "id": "LIDAR_TOP",
        },
        {"type": "sensor.other.radar", "x": 2.27, "y": 0.0, "z": 0.48, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "range": 100, "horizontal_fov": 30, "vertical_fov": 30, "id": "RADAR_FRONT"},
        {"type": "sensor.other.radar", "x": 1.21, "y": -0.85, "z": 0.74, "roll": 0.0, "pitch": 0.0, "yaw": -90.0, "range": 100, "horizontal_fov": 30, "vertical_fov": 30, "id": "RADAR_FRONT_LEFT"},
        {"type": "sensor.other.radar", "x": 1.21, "y": 0.85, "z": 0.74, "roll": 0.0, "pitch": 0.0, "yaw": 90.0, "range": 100, "horizontal_fov": 30, "vertical_fov": 30, "id": "RADAR_FRONT_RIGHT"},
        {"type": "sensor.other.radar", "x": -2.0, "y": -0.67, "z": 0.51, "roll": 0.0, "pitch": 0.0, "yaw": -90.0, "range": 100, "horizontal_fov": 30, "vertical_fov": 30, "id": "RADAR_BACK_LEFT"},
        # Official 0.0.3 used -90 for both rear radars; +90 fixes the known right-radar orientation bug.
        {"type": "sensor.other.radar", "x": -2.0, "y": 0.67, "z": 0.51, "roll": 0.0,
         "pitch": 0.0, "yaw": 90.0, "range": 100, "horizontal_fov": 30,
         "vertical_fov": 30, "id": "RADAR_BACK_RIGHT"},
        {"type": "sensor.camera.rgb", "x": 0.0, "y": 0.0, "z": 50.0, "roll": 0.0,
         "pitch": -90.0, "yaw": 0.0, "width": 1600, "height": 900, "fov": 110,
         "id": "TOP_DOWN"},
    ]
    return sensors


def merge_sensor_specs(collection: List[Dict], expert: List[Dict]) -> List[Dict]:
    merged = {item["id"]: dict(item) for item in collection}
    for item in expert:
        sensor_id = item["id"]
        if sensor_id in merged and merged[sensor_id] != item:
            raise ValueError("采集器与专家的传感器 ID 冲突: %s" % sensor_id)
        merged[sensor_id] = dict(item)
    return list(merged.values())
