from __future__ import annotations

import gzip
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import h5py
import laspy
import numpy as np

from .annotations import WorldAnnotator
from .dataset_profiles import DatasetProfile
from .navigation import NavigationSignal


RGB_IDS = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT")
RADAR_IDS = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK_LEFT", "BACK_RIGHT")
BASE_SCHEMA = "b2d-base-e2e-v1"


def _jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("not JSON serializable: %r" % (type(value),))


def _sensor(input_data: Dict, sensor_id: str, default=None):
    item = input_data.get(sensor_id)
    return default if item is None else item[1]


def _sensor_frame(input_data: Dict, sensor_id: str) -> Optional[int]:
    item = input_data.get(sensor_id)
    if not isinstance(item, (tuple, list)) or not item:
        return None
    try:
        return int(item[0])
    except Exception:
        return None


def _tmp_path(destination: Path) -> Path:
    # Keep the real suffix as the final suffix so cv2 / np.savez / h5py can
    # infer the file format. Example: 00001.jpg -> .00001.tmp.jpg.
    suffixes = "".join(destination.suffixes)
    base = destination.name[:-len(suffixes)] if suffixes else destination.name
    return destination.with_name(".%s.tmp%s" % (base, suffixes))


def _replace(tmp: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(tmp), str(destination))


def _cleanup_tmp(tmp: Path) -> None:
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass


def _world_xy_to_ego(world_xy, ego_state: Dict[str, Any]):
    location = ego_state.get("location") or [0.0, 0.0, 0.0]
    rotation = ego_state.get("rotation") or [0.0, 0.0, 0.0]
    dx = float(world_xy[0]) - float(location[0])
    dy = float(world_xy[1]) - float(location[1])
    yaw = math.radians(float(rotation[2]))
    c = math.cos(yaw)
    s = math.sin(yaw)
    # CARLA ego frame: x forward, y right.
    return [c * dx + s * dy, -s * dx + c * dy]


class Bench2DriveWriter:
    """Training-oriented Bench2Drive-style writer.

    Base v1 intentionally keeps the raw sequential information needed by
    end-to-end trajectory learners, while retaining the familiar Bench2Drive
    camera/lidar/radar/anno layout. It is not advertised as a byte-for-byte
    clone of the full official Bench2Drive release.

    Commit rule: ``anno/<frame>.json.gz`` is written LAST. Therefore an anno
    file means every required modality for that frame was successfully saved.
    """

    def __init__(
        self,
        root: str,
        route: Any,
        weather_id: int,
        frequency_hz: float,
        jpeg_quality: int,
        profile: str,
    ):
        created = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        clip_name = "%s_%s_Route%s_Weather%d_%s" % (
            route.scenario_name,
            route.town,
            route.id,
            weather_id,
            created,
        )
        self.path = Path(root) / clip_name
        self.frequency_hz = float(frequency_hz)
        self.jpeg_quality = int(jpeg_quality)
        self.profile = profile
        self.dataset_profile = DatasetProfile.from_environment()
        self.dataset_profile.validate_collection_sensor_profile(self.profile)
        self.frame = 0
        self.last_saved_timestamp: Optional[float] = None
        self.annotator = WorldAnnotator()
        self.meta_path = self.path / "_collector_meta"
        self._prepare()
        self._write_clip_metadata(route, weather_id)

    def _prepare(self) -> None:
        dirs = ["anno", "measurements", "_collector_meta"]
        if self.profile == "full":
            dirs += ["lidar", "radar"]
        for name in RGB_IDS:
            dirs.append("camera/rgb_%s" % name.lower())
            if self.profile == "full":
                dirs += [
                    "camera/depth_%s" % name.lower(),
                    "camera/semantic_%s" % name.lower(),
                    "camera/instance_%s" % name.lower(),
                ]
        if self.profile == "full":
            dirs.append("camera/rgb_top_down")
        for directory in dirs:
            (self.path / directory).mkdir(parents=True, exist_ok=True)

    def _write_clip_metadata(self, route: Any, weather_id: int) -> None:
        metadata = {
            "schema": BASE_SCHEMA,
            "bench2drive_compatibility": "partial-training-oriented",
            "collector_version": "0.1.0+base-save-v1",
            "route_id": route.id,
            "town": route.town,
            "scenario_name": route.scenario_name,
            "scenario_types": [item.type for item in route.scenarios],
            "weather_id": int(weather_id),
            "frequency_hz": self.frequency_hz,
            "sensor_profile": self.profile,
            "dataset_profile": self.dataset_profile.metadata(),
            "jpeg_quality": self.jpeg_quality,
            "frame_commit_marker": "anno/<frame>.json.gz",
            "coordinate_convention": {
                "ego": "CARLA local: +x forward, +y right, +z up",
                "world_rotation": "[pitch, roll, yaw] degrees",
                "imu_gyro": "raw leaderboard IMU value",
                "actor_angular_velocity": "CARLA actor angular velocity (deg/s)",
            },
            "training_note": (
                "Future trajectories are derived losslessly from consecutive "
                "measurements/<frame>.json.gz ego poses."
            ),
            "created_unix": time.time(),
        }
        self._atomic_json(self.meta_path / "clip.json", metadata, gzip_output=False)
        self._atomic_json(
            self.meta_path / "dataset_profile.json",
            self.dataset_profile.metadata(),
            gzip_output=False,
        )

    def due(self, timestamp: float) -> bool:
        if self.last_saved_timestamp is None:
            return True
        return timestamp - self.last_saved_timestamp >= (1.0 / self.frequency_hz) - 1e-4

    def _required_sensor_ids(self):
        required = ["CAM_%s" % name for name in RGB_IDS]
        required += ["GPS", "IMU", "SPEED"]
        if self.profile == "full":
            for name in RGB_IDS:
                required += [
                    "CAM_%s_DEPTH" % name,
                    "CAM_%s_SEM_SEG" % name,
                    "CAM_%s_INS_SEG" % name,
                ]
            required += ["TOP_DOWN", "LIDAR_TOP"]
            required += ["RADAR_%s" % name for name in RADAR_IDS]
        return required

    def _validate_input_frame(self, input_data: Dict) -> None:
        missing = [sensor_id for sensor_id in self._required_sensor_ids() if sensor_id not in input_data]
        if missing:
            raise RuntimeError("Refusing partial dataset frame; missing sensors: %s" % ", ".join(missing))

        frames = {
            sensor_id: _sensor_frame(input_data, sensor_id)
            for sensor_id in self._required_sensor_ids()
        }
        valid_frames = [value for value in frames.values() if value is not None]
        if valid_frames and len(set(valid_frames)) != 1:
            raise RuntimeError("Refusing unsynchronized dataset frame: %r" % frames)

    @staticmethod
    def _decode_depth_meters(image: np.ndarray) -> np.ndarray:
        raw = np.asarray(image)[:, :, :3].astype(np.float32)
        # CARLA raw_data arrives as BGRA. RGB packed depth is therefore
        # R=raw[...,2], G=raw[...,1], B=raw[...,0].
        normalized = (
            raw[:, :, 2]
            + raw[:, :, 1] * 256.0
            + raw[:, :, 0] * 65536.0
        ) / 16777215.0
        return (normalized * 1000.0).astype(np.float32)

    def _atomic_image(self, destination: Path, image: np.ndarray, params=None) -> None:
        tmp = _tmp_path(destination)
        try:
            ok = cv2.imwrite(str(tmp), image, params or [])
            if not ok:
                raise IOError("cv2.imwrite failed: %s" % destination)
            _replace(tmp, destination)
        finally:
            _cleanup_tmp(tmp)

    def _atomic_npz(self, destination: Path, **arrays) -> None:
        tmp = _tmp_path(destination)
        try:
            np.savez_compressed(str(tmp), **arrays)
            _replace(tmp, destination)
        finally:
            _cleanup_tmp(tmp)

    def _atomic_json(self, destination: Path, payload: Dict, gzip_output: bool) -> None:
        tmp = _tmp_path(destination)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if gzip_output:
                with gzip.open(str(tmp), "wt", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, default=_jsonable)
            else:
                with tmp.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2, default=_jsonable)
            _replace(tmp, destination)
        finally:
            _cleanup_tmp(tmp)

    def _save_lidar(self, points: np.ndarray, destination: Path) -> None:
        xyz = np.asarray(points, dtype=np.float32)
        xyz = xyz[:, :3].copy() if xyz.size else np.empty((0, 3), dtype=np.float32)
        # Current LIDAR_TOP is axis-aligned with ego, so translation is enough.
        if xyz.shape[0]:
            xyz += np.array([-0.39, 0.0, 1.84], dtype=np.float32)

        header = laspy.LasHeader(point_format=0)
        header.scales = np.array([0.001, 0.001, 0.001])
        header.offsets = np.min(xyz, axis=0) if xyz.shape[0] else np.zeros(3, dtype=np.float64)
        record = laspy.ScaleAwarePointRecord.zeros(xyz.shape[0], header=header)
        if xyz.shape[0]:
            record.x, record.y, record.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        tmp = _tmp_path(destination)
        try:
            with laspy.open(str(tmp), mode="w", header=header) as output:
                output.write_points(record)
            _replace(tmp, destination)
        finally:
            _cleanup_tmp(tmp)

    def _save_radar(self, input_data: Dict, destination: Path) -> None:
        tmp = _tmp_path(destination)
        try:
            with h5py.File(str(tmp), "w") as handle:
                for name in RADAR_IDS:
                    data = _sensor(
                        input_data,
                        "RADAR_%s" % name,
                        np.empty((0, 4), dtype=np.float32),
                    )
                    handle.create_dataset(
                        "radar_%s" % name.lower(),
                        data=np.asarray(data, dtype=np.float16),
                        compression="gzip",
                        compression_opts=9,
                        chunks=True,
                    )
            _replace(tmp, destination)
        finally:
            _cleanup_tmp(tmp)

    def _save_sensors(self, input_data: Dict, stem: str) -> None:
        for name in RGB_IDS:
            image = np.asarray(_sensor(input_data, "CAM_%s" % name))
            self._atomic_image(
                self.path / "camera" / ("rgb_%s" % name.lower()) / (stem + ".jpg"),
                image[:, :, :3],
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )

            if self.profile != "full":
                continue

            depth = np.asarray(_sensor(input_data, "CAM_%s_DEPTH" % name))
            semantic = np.asarray(_sensor(input_data, "CAM_%s_SEM_SEG" % name))
            instance = np.asarray(_sensor(input_data, "CAM_%s_INS_SEG" % name))

            self._atomic_npz(
                self.path / "camera" / ("depth_%s" % name.lower()) / (stem + ".npz"),
                depth=self._decode_depth_meters(depth),
            )
            self._atomic_image(
                self.path / "camera" / ("semantic_%s" % name.lower()) / (stem + ".png"),
                semantic[:, :, 2],
            )
            # Preserve the CARLA packed instance payload losslessly.
            self._atomic_image(
                self.path / "camera" / ("instance_%s" % name.lower()) / (stem + ".png"),
                instance,
            )

        if self.profile != "full":
            return

        top_down = np.asarray(_sensor(input_data, "TOP_DOWN"))
        self._atomic_image(
            self.path / "camera" / "rgb_top_down" / (stem + ".jpg"),
            top_down[:, :, :3],
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )

        self._save_lidar(
            np.asarray(_sensor(input_data, "LIDAR_TOP")),
            self.path / "lidar" / (stem + ".laz"),
        )
        self._save_radar(input_data, self.path / "radar" / (stem + ".h5"))

    def _measurement(
        self,
        input_data: Dict,
        control: Any,
        timestamp: float,
        nav: NavigationSignal,
        source: str,
        failure_active: bool,
    ) -> Dict[str, Any]:
        gps = np.asarray(_sensor(input_data, "GPS", [0.0, 0.0, 0.0]), dtype=np.float64)
        imu = np.asarray(_sensor(input_data, "IMU", [0.0] * 7), dtype=np.float64)
        speed_value = _sensor(input_data, "SPEED", {"speed": 0.0})
        speed = float(speed_value.get("speed", 0.0)) if isinstance(speed_value, dict) else float(speed_value)
        ego_state = self.annotator.ego_state()

        if not ego_state:
            ego_state = {
                "location": [float(gps[0]), float(gps[1]), float(gps[2]) if gps.size > 2 else 0.0],
                "rotation": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "acceleration": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
            }

        sensor_frames = {}
        for sensor_id, item in input_data.items():
            frame = _sensor_frame(input_data, sensor_id)
            if frame is not None:
                sensor_frames[sensor_id] = frame

        return {
            "schema": BASE_SCHEMA,
            "frame_index": int(self.frame),
            "timestamp": float(timestamp),
            "sensor_frames": sensor_frames,
            "ego": {
                "location": ego_state.get("location", [0.0, 0.0, 0.0]),
                "rotation": ego_state.get("rotation", [0.0, 0.0, 0.0]),
                "world2ego": ego_state.get("world2ego"),
                "ego2world": ego_state.get("ego2world"),
                "velocity": ego_state.get("velocity", [0.0, 0.0, 0.0]),
                "acceleration": ego_state.get("acceleration", [0.0, 0.0, 0.0]),
                "angular_velocity_deg_s": ego_state.get("angular_velocity", [0.0, 0.0, 0.0]),
                "speed": speed,
                "gnss": gps.tolist(),
                "imu_accelerometer": imu[:3].tolist() if imu.size >= 3 else [0.0, 0.0, 0.0],
                "imu_gyroscope": imu[3:6].tolist() if imu.size >= 6 else [0.0, 0.0, 0.0],
                "compass": float(imu[-1]) if imu.size else 0.0,
            },
            "control": {
                "throttle": float(control.throttle),
                "steer": float(control.steer),
                "brake": float(control.brake),
                "reverse": bool(control.reverse),
                "hand_brake": bool(getattr(control, "hand_brake", False)),
                "manual_gear_shift": bool(getattr(control, "manual_gear_shift", False)),
                "gear": int(getattr(control, "gear", 0)),
            },
            "navigation": {
                "near_world_xy": [float(nav.near_xy[0]), float(nav.near_xy[1])],
                "far_world_xy": [float(nav.far_xy[0]), float(nav.far_xy[1])],
                "near_ego_xy": _world_xy_to_ego(nav.near_xy, ego_state),
                "far_ego_xy": _world_xy_to_ego(nav.far_xy, ego_state),
                "near_command": int(nav.near_command),
                "far_command": int(nav.far_command),
            },
            "collector": {
                "control_source": str(source),
                "failure_segment": bool(failure_active),
            },
        }

    def _annotation(self, input_data: Dict, control: Any, nav: NavigationSignal) -> Dict[str, Any]:
        gps = np.asarray(_sensor(input_data, "GPS", [0.0, 0.0, 0.0]))
        imu = np.asarray(_sensor(input_data, "IMU", [0.0] * 7))
        speed_value = _sensor(input_data, "SPEED", {"speed": 0.0})
        speed = float(speed_value.get("speed", 0.0)) if isinstance(speed_value, dict) else float(speed_value)
        ego_location = self.annotator.ego_location()

        return {
            "x": float(ego_location.x) if ego_location is not None else float(gps[0]),
            "y": float(ego_location.y) if ego_location is not None else float(gps[1]),
            "throttle": float(control.throttle),
            "steer": float(control.steer),
            "brake": float(control.brake),
            "reverse": bool(control.reverse),
            "theta": float(imu[-1]) if imu.size else 0.0,
            "speed": speed,
            "x_command_far": float(nav.far_xy[0]),
            "y_command_far": float(nav.far_xy[1]),
            "command_far": int(nav.far_command),
            "x_command_near": float(nav.near_xy[0]),
            "y_command_near": float(nav.near_xy[1]),
            "command_near": int(nav.near_command),
            "should_brake": bool(control.brake > 0.1),
            "only_ap_brake": False,
            "x_target": float(nav.near_xy[0]),
            "y_target": float(nav.near_xy[1]),
            "next_command": int(nav.near_command),
            "weather": self.annotator.weather(),
            "acceleration": imu[:3].tolist() if imu.size >= 3 else [0.0, 0.0, 0.0],
            "angular_velocity": imu[3:6].tolist() if imu.size >= 6 else [0.0, 0.0, 0.0],
            "sensors": self.annotator.sensor_calibration(),
            "bounding_boxes": self.annotator.bounding_boxes(),
        }

    def record(
        self,
        input_data: Dict,
        control: Any,
        timestamp: float,
        nav: NavigationSignal,
        source: str,
        failure_active: bool,
        expert_assessment: Optional[np.ndarray] = None,
    ) -> bool:
        if not self.due(timestamp):
            return False

        self._validate_input_frame(input_data)
        stem = "%05d" % self.frame

        # 1) sensor blobs
        self._save_sensors(input_data, stem)

        # 2) generic E2E supervision/state sidecar
        measurement = self._measurement(
            input_data,
            control,
            timestamp,
            nav,
            source,
            failure_active,
        )
        self._atomic_json(
            self.path / "measurements" / (stem + ".json.gz"),
            measurement,
            gzip_output=True,
        )

        # Optional non-canonical teacher output: keep it outside anno.
        if expert_assessment is not None:
            assessment_dir = self.meta_path / "expert_assessment"
            assessment_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_npz(
                assessment_dir / (stem + ".npz"),
                assessment=np.asarray(expert_assessment),
            )

        # 3) annotation is the LAST write and therefore the frame commit marker.
        annotation = self._annotation(input_data, control, nav)
        self._atomic_json(
            self.path / "anno" / (stem + ".json.gz"),
            annotation,
            gzip_output=True,
        )

        self.frame += 1
        self.last_saved_timestamp = float(timestamp)
        return True

    def append_event(self, timestamp: float, event: str, active: bool) -> None:
        destination = self.meta_path / "events.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"timestamp": float(timestamp), "event": event, "active": bool(active)},
                    ensure_ascii=False,
                )
                + "\n"
            )
