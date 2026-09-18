from __future__ import print_function

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import laspy
import numpy as np


CAMERAS = (
    "front",
    "front_left",
    "front_right",
    "back",
    "back_left",
    "back_right",
)

PRIMARY_CALS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

EXPECTED_CAM_XYZ = {
    "CAM_FRONT": np.array([0.80, 0.0, 1.60]),
    "CAM_FRONT_LEFT": np.array([0.27, -0.55, 1.60]),
    "CAM_FRONT_RIGHT": np.array([0.27, 0.55, 1.60]),
    "CAM_BACK": np.array([-2.0, 0.0, 1.60]),
    "CAM_BACK_LEFT": np.array([-0.32, -0.55, 1.60]),
    "CAM_BACK_RIGHT": np.array([-0.32, 0.55, 1.60]),
}

REQUIRED_SENSOR_FRAMES = {
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "GPS",
    "IMU",
    "SPEED",

    "CAM_FRONT_DEPTH",
    "CAM_FRONT_LEFT_DEPTH",
    "CAM_FRONT_RIGHT_DEPTH",
    "CAM_BACK_DEPTH",
    "CAM_BACK_LEFT_DEPTH",
    "CAM_BACK_RIGHT_DEPTH",

    "CAM_FRONT_SEM_SEG",
    "CAM_FRONT_LEFT_SEM_SEG",
    "CAM_FRONT_RIGHT_SEM_SEG",
    "CAM_BACK_SEM_SEG",
    "CAM_BACK_LEFT_SEM_SEG",
    "CAM_BACK_RIGHT_SEM_SEG",

    "CAM_FRONT_INS_SEG",
    "CAM_FRONT_LEFT_INS_SEG",
    "CAM_FRONT_RIGHT_INS_SEG",
    "CAM_BACK_INS_SEG",
    "CAM_BACK_LEFT_INS_SEG",
    "CAM_BACK_RIGHT_INS_SEG",

    "TOP_DOWN",
    "LIDAR_TOP",

    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
}

RADAR_DATASETS = (
    "radar_front",
    "radar_front_left",
    "radar_front_right",
    "radar_back_left",
    "radar_back_right",
)


def finite_array(value, shape=None):
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return False

    if shape is not None and arr.shape != shape:
        return False

    return np.isfinite(arr).all()


def load_gzip_json(path):
    with gzip.open(str(path), "rt", encoding="utf-8") as f:
        return json.load(f)


def fail(errors, frame, message):
    errors.append(
        "%s: %s" % (
            frame,
            message,
        )
    )


def audit(clip):
    errors = []
    warnings = []

    meta_path = (
        clip
        / "_collector_meta"
        / "clip.json"
    )

    if not meta_path.exists():
        raise RuntimeError(
            "missing %s" % meta_path
        )

    with meta_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        meta = json.load(f)

    print("=" * 78)
    print("B2D COLLECTOR FINAL DEEP AUDIT")
    print("=" * 78)
    print("clip:", clip)
    print("schema:", meta.get("schema"))
    print("sensor_profile:", meta.get("sensor_profile"))
    print("frequency_hz:", meta.get("frequency_hz"))
    print()

    if meta.get("schema") != "b2d-base-e2e-v1":
        errors.append(
            "unexpected schema %r"
            % meta.get("schema")
        )

    if meta.get("sensor_profile") != "full":
        errors.append(
            "sensor_profile is not full"
        )

    frequency = float(
        meta.get("frequency_hz", 10.0)
    )

    expected_dt = (
        1.0 / frequency
        if frequency > 0.0
        else 0.1
    )

    anno_files = sorted(
        (clip / "anno").glob(
            "*.json.gz"
        )
    )

    measurement_files = sorted(
        (clip / "measurements").glob(
            "*.json.gz"
        )
    )

    if not anno_files:
        errors.append("no annotation files")
        return errors, warnings

    expected_stems = [
        "%05d" % i
        for i in range(len(anno_files))
    ]

    anno_stems = [
        p.name.split(".")[0]
        for p in anno_files
    ]

    measurement_stems = [
        p.name.split(".")[0]
        for p in measurement_files
    ]

    if anno_stems != expected_stems:
        errors.append(
            "annotation frame sequence is not contiguous"
        )

    if measurement_stems != expected_stems:
        errors.append(
            "measurement frame set differs from annotation"
        )

    print(
        "annotation frames:",
        len(anno_files),
    )

    print(
        "measurement frames:",
        len(measurement_files),
    )

    #
    # Filesystem integrity
    #
    zero_files = []
    tmp_files = []

    for path in clip.rglob("*"):
        if not path.is_file():
            continue

        try:
            if path.stat().st_size == 0:
                zero_files.append(
                    str(path.relative_to(clip))
                )
        except OSError:
            pass

        name = path.name.lower()

        if (
            ".tmp" in name
            or name.endswith(".partial")
            or name.endswith(".part")
        ):
            tmp_files.append(
                str(path.relative_to(clip))
            )

    if zero_files:
        errors.append(
            "zero-byte files: %d"
            % len(zero_files)
        )

    if tmp_files:
        errors.append(
            "temporary/partial files: %d"
            % len(tmp_files)
        )

    #
    # Aggregate statistics
    #
    timestamps = []
    throttle_values = []
    steer_values = []
    brake_values = []

    sensor_cal_counts = set()
    max_primary_mount_error = defaultdict(float)

    static_tracks = defaultdict(list)
    static_total = 0
    dynamic_total = 0

    lidar_counts = []
    depth_min = float("inf")
    depth_max = float("-inf")

    radar_detection_counts = defaultdict(list)

    #
    # JSON / temporal / calibration / controls / annotations
    #
    print()
    print("[1/6] JSON + temporal + control + calibration + boxes")

    for index, stem in enumerate(expected_stems):
        anno_path = (
            clip / "anno"
            / (stem + ".json.gz")
        )

        measurement_path = (
            clip / "measurements"
            / (stem + ".json.gz")
        )

        try:
            anno = load_gzip_json(
                anno_path
            )
        except Exception as exc:
            fail(
                errors,
                stem,
                "annotation unreadable: %s"
                % exc,
            )
            continue

        try:
            measurement = (
                load_gzip_json(
                    measurement_path
                )
            )
        except Exception as exc:
            fail(
                errors,
                stem,
                "measurement unreadable: %s"
                % exc,
            )
            continue

        #
        # Frame / timestamp
        #
        if int(
            measurement.get(
                "frame_index",
                -1,
            )
        ) != index:
            fail(
                errors,
                stem,
                "measurement frame_index mismatch",
            )

        try:
            timestamp = float(
                measurement["timestamp"]
            )

            if not math.isfinite(
                timestamp
            ):
                raise ValueError(
                    "non-finite"
                )

            timestamps.append(
                timestamp
            )
        except Exception:
            fail(
                errors,
                stem,
                "bad timestamp",
            )

        #
        # Sensor synchronization
        #
        sensor_frames = (
            measurement.get(
                "sensor_frames",
                {}
            )
        )

        missing_sensor_frames = (
            REQUIRED_SENSOR_FRAMES
            - set(sensor_frames)
        )

        if missing_sensor_frames:
            fail(
                errors,
                stem,
                "missing sensor_frames: %s"
                % sorted(
                    missing_sensor_frames
                ),
            )

        synced_values = []

        for sensor_id in (
            REQUIRED_SENSOR_FRAMES
            & set(sensor_frames)
        ):
            try:
                synced_values.append(
                    int(
                        sensor_frames[
                            sensor_id
                        ]
                    )
                )
            except Exception:
                fail(
                    errors,
                    stem,
                    "invalid sensor frame %s"
                    % sensor_id,
                )

        if (
            synced_values
            and len(
                set(synced_values)
            ) != 1
        ):
            fail(
                errors,
                stem,
                "required sensors are not frame-synchronized",
            )

        #
        # Ego state
        #
        ego = measurement.get(
            "ego",
            {},
        )

        for key in (
            "location",
            "rotation",
            "velocity",
            "acceleration",
        ):
            if not finite_array(
                ego.get(key),
                (3,),
            ):
                fail(
                    errors,
                    stem,
                    "bad ego.%s" % key,
                )

        try:
            ego_speed = float(
                ego["speed"]
            )

            if (
                not math.isfinite(
                    ego_speed
                )
                or ego_speed < -1e-6
            ):
                raise ValueError()
        except Exception:
            fail(
                errors,
                stem,
                "bad ego.speed",
            )

        for matrix_key in (
            "world2ego",
            "ego2world",
        ):
            value = ego.get(
                matrix_key
            )

            if (
                value is not None
                and not finite_array(
                    value,
                    (4, 4),
                )
            ):
                fail(
                    errors,
                    stem,
                    "bad ego.%s"
                    % matrix_key,
                )

        #
        # Executed control
        #
        control = measurement.get(
            "control",
            {},
        )

        try:
            throttle = float(
                control["throttle"]
            )

            steer = float(
                control["steer"]
            )

            brake = float(
                control["brake"]
            )
        except Exception:
            fail(
                errors,
                stem,
                "missing/bad control values",
            )

            throttle = float("nan")
            steer = float("nan")
            brake = float("nan")

        if not (
            math.isfinite(throttle)
            and 0.0 <= throttle <= 1.0
        ):
            fail(
                errors,
                stem,
                "throttle outside [0,1]",
            )

        if not (
            math.isfinite(steer)
            and -1.0 <= steer <= 1.0
        ):
            fail(
                errors,
                stem,
                "steer outside [-1,1]",
            )

        if not (
            math.isfinite(brake)
            and 0.0 <= brake <= 1.0
        ):
            fail(
                errors,
                stem,
                "brake outside [0,1]",
            )

        if math.isfinite(throttle):
            throttle_values.append(
                throttle
            )

        if math.isfinite(steer):
            steer_values.append(
                steer
            )

        if math.isfinite(brake):
            brake_values.append(
                brake
            )

        #
        # Annotation control must equal executed control.
        #
        for key, measurement_value in (
            ("throttle", throttle),
            ("steer", steer),
            ("brake", brake),
        ):
            try:
                anno_value = float(
                    anno[key]
                )

                if abs(
                    anno_value
                    - measurement_value
                ) > 1e-7:
                    fail(
                        errors,
                        stem,
                        "anno/control mismatch: %s"
                        % key,
                    )
            except Exception:
                fail(
                    errors,
                    stem,
                    "bad anno control %s"
                    % key,
                )

        if bool(
            anno.get(
                "reverse",
                False,
            )
        ) != bool(
            control.get(
                "reverse",
                False,
            )
        ):
            fail(
                errors,
                stem,
                "anno/control mismatch: reverse",
            )

        #
        # Calibration
        #
        sensors = anno.get(
            "sensors",
            {},
        )

        if not isinstance(
            sensors,
            dict,
        ):
            fail(
                errors,
                stem,
                "sensors calibration is not dict",
            )
            sensors = {}

        sensor_cal_counts.add(
            len(sensors)
        )

        for forbidden in (
            "front",
            "default",
        ):
            if forbidden in sensors:
                fail(
                    errors,
                    stem,
                    "forbidden calibration alias %r"
                    % forbidden,
                )

        for sensor_id in PRIMARY_CALS:
            cal = sensors.get(
                sensor_id
            )

            if not isinstance(
                cal,
                dict,
            ):
                fail(
                    errors,
                    stem,
                    "missing %s calibration"
                    % sensor_id,
                )
                continue

            for key in (
                "location",
                "rotation",
                "intrinsic",
                "world2cam",
                "cam2ego",
                "fov",
                "image_size_x",
                "image_size_y",
            ):
                if key not in cal:
                    fail(
                        errors,
                        stem,
                        "%s missing %s"
                        % (
                            sensor_id,
                            key,
                        ),
                    )

            if not finite_array(
                cal.get("intrinsic"),
                (3, 3),
            ):
                fail(
                    errors,
                    stem,
                    "%s bad intrinsic"
                    % sensor_id,
                )

            if not finite_array(
                cal.get("world2cam"),
                (4, 4),
            ):
                fail(
                    errors,
                    stem,
                    "%s bad world2cam"
                    % sensor_id,
                )

            if not finite_array(
                cal.get("cam2ego"),
                (4, 4),
            ):
                fail(
                    errors,
                    stem,
                    "%s bad cam2ego"
                    % sensor_id,
                )
            else:
                c2e = np.asarray(
                    cal["cam2ego"],
                    dtype=np.float64,
                )

                actual_xyz = (
                    c2e[:3, 3]
                )

                error = float(
                    np.linalg.norm(
                        actual_xyz
                        - EXPECTED_CAM_XYZ[
                            sensor_id
                        ]
                    )
                )

                max_primary_mount_error[
                    sensor_id
                ] = max(
                    max_primary_mount_error[
                        sensor_id
                    ],
                    error,
                )

                if error > 0.02:
                    fail(
                        errors,
                        stem,
                        "%s mount translation error %.4f m"
                        % (
                            sensor_id,
                            error,
                        ),
                    )

            try:
                if int(
                    cal[
                        "image_size_x"
                    ]
                ) != 1600:
                    raise ValueError()

                if int(
                    cal[
                        "image_size_y"
                    ]
                ) != 900:
                    raise ValueError()
            except Exception:
                fail(
                    errors,
                    stem,
                    "%s image size != 1600x900"
                    % sensor_id,
                )

        #
        # Bounding boxes
        #
        boxes = anno.get(
            "bounding_boxes",
            [],
        )

        if not isinstance(
            boxes,
            list,
        ):
            fail(
                errors,
                stem,
                "bounding_boxes is not list",
            )
            boxes = []

        ego_boxes = 0
        seen_vehicle_ids = set()

        for box in boxes:
            if not isinstance(
                box,
                dict,
            ):
                fail(
                    errors,
                    stem,
                    "non-dict bounding box",
                )
                continue

            if (
                box.get("class")
                != "vehicle"
            ):
                continue

            vehicle_id = str(
                box.get("id")
            )

            if vehicle_id in seen_vehicle_ids:
                fail(
                    errors,
                    stem,
                    "duplicate vehicle id %s"
                    % vehicle_id,
                )

            seen_vehicle_ids.add(
                vehicle_id
            )

            if "world2ego" in box:
                ego_boxes += 1

            state = box.get(
                "state"
            )

            if state not in (
                "dynamic",
                "static",
            ):
                fail(
                    errors,
                    stem,
                    "vehicle %s has invalid state %r"
                    % (
                        vehicle_id,
                        state,
                    ),
                )
                continue

            if state == "dynamic":
                dynamic_total += 1
            else:
                static_total += 1

            extent = np.asarray(
                box.get(
                    "extent",
                    [],
                ),
                dtype=np.float64,
            )

            if (
                extent.shape != (3,)
                or not np.isfinite(
                    extent
                ).all()
                or np.any(
                    extent <= 0.0
                )
            ):
                fail(
                    errors,
                    stem,
                    "vehicle %s has bad extent"
                    % vehicle_id,
                )

            if not finite_array(
                box.get(
                    "world2vehicle"
                ),
                (4, 4),
            ):
                fail(
                    errors,
                    stem,
                    "vehicle %s bad/missing world2vehicle"
                    % vehicle_id,
                )

            if state == "static":
                if not finite_array(
                    box.get(
                        "world_cord"
                    ),
                    (8, 3),
                ):
                    fail(
                        errors,
                        stem,
                        "static %s bad world_cord"
                        % vehicle_id,
                    )

                try:
                    match_distance = float(
                        box[
                            "bbox_match_distance"
                        ]
                    )

                    if not (
                        math.isfinite(
                            match_distance
                        )
                        and 0.0
                        <= match_distance
                        <= 20.0
                    ):
                        raise ValueError()

                except Exception:
                    fail(
                        errors,
                        stem,
                        "static %s bad bbox_match_distance"
                        % vehicle_id,
                    )

                if (
                    box.get(
                        "bbox_source"
                    )
                    != "carla_world_level_bb"
                ):
                    fail(
                        errors,
                        stem,
                        "static %s unexpected bbox_source"
                        % vehicle_id,
                    )

                static_tracks[
                    vehicle_id
                ].append(
                    (
                        stem,
                        np.asarray(
                            box["center"],
                            dtype=np.float64,
                        ),
                        np.asarray(
                            box["extent"],
                            dtype=np.float64,
                        ),
                    )
                )

        if ego_boxes != 1:
            fail(
                errors,
                stem,
                "expected exactly 1 ego box, got %d"
                % ego_boxes,
            )

    #
    # Timestamp statistics
    #
    if len(timestamps) >= 2:
        dt = np.diff(
            np.asarray(
                timestamps,
                dtype=np.float64,
            )
        )

        if np.any(
            ~np.isfinite(dt)
        ):
            errors.append(
                "timestamp delta contains non-finite values"
            )

        if np.any(dt <= 0.0):
            errors.append(
                "timestamps are not strictly increasing"
            )

        dt_error = np.abs(
            dt - expected_dt
        )

        # The collector is synchronous; allow only a small
        # numerical/scheduling tolerance.
        if np.max(dt_error) > 0.005:
            errors.append(
                "timestamp cadence error >5 ms; max error %.9f s"
                % np.max(dt_error)
            )

        print(
            "  dt min/median/max:",
            "%.9f / %.9f / %.9f s"
            % (
                np.min(dt),
                np.median(dt),
                np.max(dt),
            ),
        )

        print(
            "  max |dt-expected|:",
            "%.9f s"
            % np.max(dt_error),
        )

    #
    # Static temporal geometry
    #
    max_static_center_change = 0.0
    max_static_extent_change = 0.0

    for actor_id, observations in (
        static_tracks.items()
    ):
        if not observations:
            continue

        ref_center = observations[0][1]
        ref_extent = observations[0][2]

        for (
            stem,
            center,
            extent,
        ) in observations:
            dc = float(
                np.linalg.norm(
                    center
                    - ref_center
                )
            )

            de = float(
                np.linalg.norm(
                    extent
                    - ref_extent
                )
            )

            max_static_center_change = max(
                max_static_center_change,
                dc,
            )

            max_static_extent_change = max(
                max_static_extent_change,
                de,
            )

            if dc > 0.05:
                fail(
                    errors,
                    stem,
                    "static %s center changed %.4f m"
                    % (
                        actor_id,
                        dc,
                    ),
                )

            if de > 0.02:
                fail(
                    errors,
                    stem,
                    "static %s extent changed %.4f m"
                    % (
                        actor_id,
                        de,
                    ),
                )

    #
    # Camera modalities
    #
    print("[2/6] RGB / semantic / instance images")

    for stem in expected_stems:
        for camera in CAMERAS:
            rgb_path = (
                clip
                / "camera"
                / ("rgb_" + camera)
                / (stem + ".jpg")
            )

            image = cv2.imread(
                str(rgb_path),
                cv2.IMREAD_UNCHANGED,
            )

            if (
                image is None
                or image.shape[:2]
                != (900, 1600)
            ):
                fail(
                    errors,
                    stem,
                    "bad RGB %s"
                    % camera,
                )

            sem_path = (
                clip
                / "camera"
                / ("semantic_" + camera)
                / (stem + ".png")
            )

            semantic = cv2.imread(
                str(sem_path),
                cv2.IMREAD_UNCHANGED,
            )

            if (
                semantic is None
                or semantic.shape[:2]
                != (900, 1600)
            ):
                fail(
                    errors,
                    stem,
                    "bad semantic %s"
                    % camera,
                )

            ins_path = (
                clip
                / "camera"
                / ("instance_" + camera)
                / (stem + ".png")
            )

            instance = cv2.imread(
                str(ins_path),
                cv2.IMREAD_UNCHANGED,
            )

            if (
                instance is None
                or instance.shape[:2]
                != (900, 1600)
            ):
                fail(
                    errors,
                    stem,
                    "bad instance %s"
                    % camera,
                )

        top_path = (
            clip
            / "camera"
            / "rgb_top_down"
            / (stem + ".jpg")
        )

        top = cv2.imread(
            str(top_path),
            cv2.IMREAD_UNCHANGED,
        )

        if (
            top is None
            or top.shape[:2]
            != (900, 1600)
        ):
            fail(
                errors,
                stem,
                "bad TOP_DOWN RGB",
            )

    #
    # Depth
    #
    print("[3/6] Metric depth NPZ")

    for stem in expected_stems:
        for camera in CAMERAS:
            path = (
                clip
                / "camera"
                / ("depth_" + camera)
                / (stem + ".npz")
            )

            try:
                with np.load(
                    str(path)
                ) as payload:
                    if "depth" not in payload:
                        raise RuntimeError(
                            "missing depth key"
                        )

                    depth = np.asarray(
                        payload["depth"]
                    )
            except Exception as exc:
                fail(
                    errors,
                    stem,
                    "depth %s unreadable: %s"
                    % (
                        camera,
                        exc,
                    ),
                )
                continue

            if depth.shape != (
                900,
                1600,
            ):
                fail(
                    errors,
                    stem,
                    "depth %s shape %r"
                    % (
                        camera,
                        depth.shape,
                    ),
                )
                continue

            if not np.isfinite(
                depth
            ).all():
                fail(
                    errors,
                    stem,
                    "depth %s contains non-finite values"
                    % camera,
                )
                continue

            local_min = float(
                np.min(depth)
            )

            local_max = float(
                np.max(depth)
            )

            depth_min = min(
                depth_min,
                local_min,
            )

            depth_max = max(
                depth_max,
                local_max,
            )

            if local_min < -1e-4:
                fail(
                    errors,
                    stem,
                    "depth %s negative"
                    % camera,
                )

            if local_max > 1000.1:
                fail(
                    errors,
                    stem,
                    "depth %s exceeds CARLA 1000 m encoding range"
                    % camera,
                )

    #
    # LiDAR
    #
    print("[4/6] LiDAR LAZ")

    for stem in expected_stems:
        path = (
            clip
            / "lidar"
            / (stem + ".laz")
        )

        try:
            cloud = laspy.read(
                str(path)
            )

            xyz = np.column_stack(
                (
                    np.asarray(cloud.x),
                    np.asarray(cloud.y),
                    np.asarray(cloud.z),
                )
            )
        except Exception as exc:
            fail(
                errors,
                stem,
                "LiDAR unreadable: %s"
                % exc,
            )
            continue

        lidar_counts.append(
            len(xyz)
        )

        if len(xyz) < 1000:
            fail(
                errors,
                stem,
                "suspiciously small LiDAR cloud: %d"
                % len(xyz),
            )

        if not np.isfinite(
            xyz
        ).all():
            fail(
                errors,
                stem,
                "LiDAR contains non-finite XYZ",
            )

    #
    # Radar
    #
    print("[5/6] Radar H5")

    for stem in expected_stems:
        path = (
            clip
            / "radar"
            / (stem + ".h5")
        )

        try:
            with h5py.File(
                str(path),
                "r",
            ) as handle:
                for dataset_name in (
                    RADAR_DATASETS
                ):
                    if dataset_name not in handle:
                        fail(
                            errors,
                            stem,
                            "missing radar dataset %s"
                            % dataset_name,
                        )
                        continue

                    data = np.asarray(
                        handle[
                            dataset_name
                        ]
                    )

                    if (
                        data.ndim != 2
                        or data.shape[1] != 4
                    ):
                        fail(
                            errors,
                            stem,
                            "%s bad shape %r"
                            % (
                                dataset_name,
                                data.shape,
                            ),
                        )
                        continue

                    if not np.isfinite(
                        data
                    ).all():
                        fail(
                            errors,
                            stem,
                            "%s contains non-finite values"
                            % dataset_name,
                        )

                    radar_detection_counts[
                        dataset_name
                    ].append(
                        int(data.shape[0])
                    )

        except Exception as exc:
            fail(
                errors,
                stem,
                "Radar H5 unreadable: %s"
                % exc,
            )

    #
    # Summary
    #
    print("[6/6] Filesystem + aggregate statistics")
    print()

    print(
        "sensor calibration counts observed:",
        sorted(sensor_cal_counts),
    )

    print(
        "primary RGB max mount errors:"
    )

    for sensor_id in PRIMARY_CALS:
        print(
            "  %-17s %.6f m"
            % (
                sensor_id,
                max_primary_mount_error[
                    sensor_id
                ],
            )
        )

    print()
    print(
        "control throttle min/max:",
        "%.6f / %.6f"
        % (
            min(throttle_values),
            max(throttle_values),
        ),
    )

    print(
        "control steer min/max:",
        "%.6f / %.6f"
        % (
            min(steer_values),
            max(steer_values),
        ),
    )

    print(
        "control brake min/max:",
        "%.6f / %.6f"
        % (
            min(brake_values),
            max(brake_values),
        ),
    )

    print()
    print(
        "dynamic vehicle annotations:",
        dynamic_total,
    )

    print(
        "static vehicle annotations:",
        static_total,
    )

    print(
        "unique static vehicle ids:",
        len(static_tracks),
    )

    print(
        "max static center change:",
        "%.6f m"
        % max_static_center_change,
    )

    print(
        "max static extent change:",
        "%.6f m"
        % max_static_extent_change,
    )

    if static_total == 0:
        errors.append(
            "no static vehicle annotations found"
        )

    print()
    print(
        "depth global min/max:",
        "%.6f / %.6f m"
        % (
            depth_min,
            depth_max,
        ),
    )

    if lidar_counts:
        print(
            "LiDAR points min/median/max:",
            "%d / %.1f / %d"
            % (
                min(lidar_counts),
                float(
                    np.median(
                        lidar_counts
                    )
                ),
                max(lidar_counts),
            ),
        )

    print()
    print("Radar detections min/median/max:")

    for name in RADAR_DATASETS:
        values = radar_detection_counts[
            name
        ]

        if values:
            print(
                "  %-18s %d / %.1f / %d"
                % (
                    name,
                    min(values),
                    float(
                        np.median(
                            values
                        )
                    ),
                    max(values),
                )
            )

    print()
    print(
        "zero-byte files:",
        len(zero_files),
    )

    print(
        "temporary/partial files:",
        len(tmp_files),
    )

    print()
    print(
        "warnings:",
        len(warnings),
    )

    for item in warnings[:30]:
        print(
            "WARNING:",
            item,
        )

    print(
        "errors:",
        len(errors),
    )

    for item in errors[:50]:
        print(
            "ERROR:",
            item,
        )

    if errors:
        print()
        print(
            "=== FINAL DEEP AUDIT FAIL ==="
        )
        return errors, warnings

    print()
    print(
        "=== FINAL DEEP AUDIT PASS ==="
    )

    return errors, warnings


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "clip",
        type=Path,
    )

    args = parser.parse_args()

    clip = (
        args.clip
        .expanduser()
        .resolve()
    )

    errors, _ = audit(clip)

    raise SystemExit(
        1 if errors else 0
    )


if __name__ == "__main__":
    main()
