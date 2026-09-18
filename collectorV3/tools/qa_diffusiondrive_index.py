from __future__ import print_function

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np


CAMERAS = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
)

EXPECTED_SAMPLES = 169
EXPECTED_POSES = 8
EXPECTED_INTERVAL = 0.5
EXPECTED_HORIZON = 4.0
MAX_AGENTS = 30

LIDAR_MIN_X = -32.0
LIDAR_MAX_X = 32.0
LIDAR_MIN_Y = -32.0
LIDAR_MAX_Y = 32.0


def load_gz(path):
    with gzip.open(str(path), "rt", encoding="utf-8") as f:
        return json.load(f)


def wrap_pi(value):
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def world_xy_to_ego(x, y, measurement):
    ego = measurement["ego"]

    ex = float(ego["location"][0])
    ey = float(ego["location"][1])

    yaw = math.radians(
        float(ego["rotation"][2])
    )

    dx = float(x) - ex
    dy = float(y) - ey

    c = math.cos(yaw)
    s = math.sin(yaw)

    return np.asarray(
        [
            c * dx + s * dy,
            -s * dx + c * dy,
        ],
        dtype=np.float64,
    )


def relative_pose(origin, target):
    xy = world_xy_to_ego(
        target["ego"]["location"][0],
        target["ego"]["location"][1],
        origin,
    )

    origin_yaw = math.radians(
        float(origin["ego"]["rotation"][2])
    )

    target_yaw = math.radians(
        float(target["ego"]["rotation"][2])
    )

    return np.asarray(
        [
            xy[0],
            xy[1],
            wrap_pi(target_yaw - origin_yaw),
        ],
        dtype=np.float64,
    )


def eligible_agents(annotation, measurement):
    dynamic = 0
    static = 0
    total = 0

    for box in annotation.get(
        "bounding_boxes",
        [],
    ):
        if box.get("class") != "vehicle":
            continue

        if "world2ego" in box:
            continue

        center = (
            box.get("center")
            or box.get("location")
        )

        if center is None:
            continue

        xy = world_xy_to_ego(
            center[0],
            center[1],
            measurement,
        )

        x = float(xy[0])
        y = float(xy[1])

        if not (
            LIDAR_MIN_X <= x <= LIDAR_MAX_X
            and
            LIDAR_MIN_Y <= y <= LIDAR_MAX_Y
        ):
            continue

        total += 1

        if box.get("state") == "static":
            static += 1
        else:
            dynamic += 1

    return total, dynamic, static


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    args = parser.parse_args()

    clip = Path(
        args.clip
    ).expanduser().resolve()

    index_path = (
        clip
        / "_collector_meta"
        / "diffusiondrive_samples.jsonl"
    )

    if not index_path.exists():
        raise SystemExit(
            "FAIL: missing index: %s"
            % index_path
        )

    with index_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        samples = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    measurement_paths = sorted(
        (clip / "measurements").glob(
            "*.json.gz"
        )
    )

    anno_paths = sorted(
        (clip / "anno").glob(
            "*.json.gz"
        )
    )

    measurements = [
        load_gz(path)
        for path in measurement_paths
    ]

    annotations = [
        load_gz(path)
        for path in anno_paths
    ]

    errors = []

    samples_with_agents = 0
    samples_with_static_candidates = 0
    total_true_agents = 0
    total_static_candidates = 0
    total_dynamic_candidates = 0

    max_trajectory_error = 0.0
    max_agent_count = 0

    print("=" * 78)
    print("DIFFUSIONDRIVE INDEX FINAL AUDIT")
    print("=" * 78)

    print("clip:", clip)
    print("index:", index_path)
    print("samples:", len(samples))

    if len(samples) != EXPECTED_SAMPLES:
        errors.append(
            "expected %d samples, got %d"
            % (
                EXPECTED_SAMPLES,
                len(samples),
            )
        )

    expected_frames = [
        "%05d" % i
        for i in range(
            EXPECTED_SAMPLES
        )
    ]

    actual_frames = [
        str(sample.get("frame"))
        for sample in samples
    ]

    if actual_frames != expected_frames:
        errors.append(
            "sample frame sequence is not 00000..00168"
        )

    for index, sample in enumerate(samples):
        frame = "%05d" % index

        if sample.get("schema") != "diffusiondrive-carla-v1":
            errors.append(
                "%s bad schema"
                % frame
            )

        inputs = sample.get(
            "inputs",
            {},
        )

        camera_paths = inputs.get(
            "cameras",
            {},
        )

        if set(camera_paths) != set(CAMERAS):
            errors.append(
                "%s camera set mismatch: %r"
                % (
                    frame,
                    sorted(camera_paths),
                )
            )

        for camera_id in CAMERAS:
            relative = camera_paths.get(
                camera_id
            )

            if not relative:
                errors.append(
                    "%s missing %s path"
                    % (
                        frame,
                        camera_id,
                    )
                )
                continue

            if not (
                clip / relative
            ).exists():
                errors.append(
                    "%s missing file %s"
                    % (
                        frame,
                        relative,
                    )
                )

        lidar_path = inputs.get(
            "lidar"
        )

        if not lidar_path:
            errors.append(
                "%s missing lidar path"
                % frame
            )
        elif not (
            clip / lidar_path
        ).exists():
            errors.append(
                "%s missing lidar file %s"
                % (
                    frame,
                    lidar_path,
                )
            )

        measurement_path = inputs.get(
            "measurement"
        )

        if (
            not measurement_path
            or not (
                clip / measurement_path
            ).exists()
        ):
            errors.append(
                "%s bad measurement path"
                % frame
            )

        #

        # Trajectory configuration

        #

        sampling = sample.get(
            "trajectory_sampling",
            {},
        )

        try:
            if abs(
                float(
                    sampling["time_horizon"]
                )
                - EXPECTED_HORIZON
            ) > 1e-9:
                raise ValueError()

            if abs(
                float(
                    sampling["interval_length"]
                )
                - EXPECTED_INTERVAL
            ) > 1e-9:
                raise ValueError()

            if int(
                sampling["num_poses"]
            ) != EXPECTED_POSES:
                raise ValueError()

        except Exception:
            errors.append(
                "%s bad trajectory_sampling"
                % frame
            )

        trajectory = np.asarray(
            sample.get(
                "targets",
                {},
            ).get(
                "trajectory",
                [],
            ),
            dtype=np.float64,
        )

        if trajectory.shape != (
            EXPECTED_POSES,
            3,
        ):
            errors.append(
                "%s trajectory shape %r"
                % (
                    frame,
                    trajectory.shape,
                )
            )
        elif not np.isfinite(
            trajectory
        ).all():
            errors.append(
                "%s trajectory non-finite"
                % frame
            )
        else:
            expected = []

            for step in range(
                1,
                EXPECTED_POSES + 1,
            ):
                future_index = (
                    index
                    + 5 * step
                )

                expected.append(
                    relative_pose(
                        measurements[index],
                        measurements[
                            future_index
                        ],
                    )
                )

            expected = np.asarray(
                expected,
                dtype=np.float64,
            )

            trajectory_error = float(
                np.max(
                    np.abs(
                        trajectory
                        - expected
                    )
                )
            )

            max_trajectory_error = max(
                max_trajectory_error,
                trajectory_error,
            )

            if trajectory_error > 1e-7:
                errors.append(
                    "%s trajectory mismatch %.9g"
                    % (
                        frame,
                        trajectory_error,
                    )
                )

        #

        # Agent target integrity

        #

        targets = sample.get(
            "targets",
            {},
        )

        states = np.asarray(
            targets.get(
                "agent_states",
                [],
            ),
            dtype=np.float64,
        )

        labels = np.asarray(
            targets.get(
                "agent_labels",
                [],
            )
        )

        if states.shape != (
            MAX_AGENTS,
            5,
        ):
            errors.append(
                "%s agent_states shape %r"
                % (
                    frame,
                    states.shape,
                )
            )
            continue

        if labels.shape != (
            MAX_AGENTS,
        ):
            errors.append(
                "%s agent_labels shape %r"
                % (
                    frame,
                    labels.shape,
                )
            )
            continue

        if not np.isfinite(
            states
        ).all():
            errors.append(
                "%s agent_states non-finite"
                % frame
            )

        labels_bool = labels.astype(
            np.bool_
        )

        true_count = int(
            np.sum(labels_bool)
        )

        total_true_agents += true_count
        max_agent_count = max(
            max_agent_count,
            true_count,
        )

        if true_count:
            samples_with_agents += 1

        # Builder writes valid agents first,

        # padding after them.

        expected_label_pattern = (
            [True] * true_count
            + [False] * (
                MAX_AGENTS
                - true_count
            )
        )

        if labels_bool.tolist() != expected_label_pattern:
            errors.append(
                "%s agent label padding is non-contiguous"
                % frame
            )

        if (
            true_count < MAX_AGENTS
            and not np.allclose(
                states[true_count:],
                0.0,
                atol=0.0,
            )
        ):
            errors.append(
                "%s padded agent states are nonzero"
                % frame
            )

        if true_count > 0:
            valid_states = states[
                :true_count
            ]

            # [x, y, heading, length, width]

            if np.any(
                valid_states[:, 3] <= 0.0
            ):
                errors.append(
                    "%s non-positive agent length"
                    % frame
                )

            if np.any(
                valid_states[:, 4] <= 0.0
            ):
                errors.append(
                    "%s non-positive agent width"
                    % frame
                )

            if np.any(
                valid_states[:, 0]
                < LIDAR_MIN_X - 1e-6
            ) or np.any(
                valid_states[:, 0]
                > LIDAR_MAX_X + 1e-6
            ):
                errors.append(
                    "%s agent x outside crop"
                    % frame
                )

            if np.any(
                valid_states[:, 1]
                < LIDAR_MIN_Y - 1e-6
            ) or np.any(
                valid_states[:, 1]
                > LIDAR_MAX_Y + 1e-6
            ):
                errors.append(
                    "%s agent y outside crop"
                    % frame
                )

        (
            candidate_count,
            dynamic_count,
            static_count,
        ) = eligible_agents(
            annotations[index],
            measurements[index],
        )

        expected_agent_count = min(
            candidate_count,
            MAX_AGENTS,
        )

        if true_count != expected_agent_count:
            errors.append(
                (
                    "%s agent count mismatch: "
                    "index=%d annotation=%d capped=%d"
                )
                % (
                    frame,
                    true_count,
                    candidate_count,
                    expected_agent_count,
                )
            )

        if static_count > 0:
            samples_with_static_candidates += 1

        total_static_candidates += (
            static_count
        )

        total_dynamic_candidates += (
            dynamic_count
        )

        #

        # Ego status

        #

        ego_status = inputs.get(
            "ego_status",
            {},
        )

        command = np.asarray(
            ego_status.get(
                "driving_command",
                [],
            ),
            dtype=np.float64,
        )

        if (
            command.shape != (4,)
            or not np.isfinite(
                command
            ).all()
            or abs(
                float(np.sum(command))
                - 1.0
            ) > 1e-9
        ):
            errors.append(
                "%s bad driving command one-hot"
                % frame
            )

        velocity = np.asarray(
            ego_status.get(
                "ego_velocity",
                [],
            ),
            dtype=np.float64,
        )

        acceleration = np.asarray(
            ego_status.get(
                "ego_acceleration",
                [],
            ),
            dtype=np.float64,
        )

        if (
            velocity.shape != (2,)
            or not np.isfinite(
                velocity
            ).all()
        ):
            errors.append(
                "%s bad ego_velocity"
                % frame
            )

        if (
            acceleration.shape != (2,)
            or not np.isfinite(
                acceleration
            ).all()
        ):
            errors.append(
                "%s bad ego_acceleration"
                % frame
            )

        if targets.get(
            "bev_semantic_map",
            "MISSING",
        ) is not None:
            errors.append(
                "%s bev_semantic_map should be null"
                % frame
            )

    print()
    print(
        "max trajectory reconstruction error:",
        "%.12g"
        % max_trajectory_error,
    )

    print(
        "samples with agents:",
        samples_with_agents,
    )

    print(
        "total indexed agents:",
        total_true_agents,
    )

    print(
        "max indexed agents/sample:",
        max_agent_count,
    )

    print()
    print(
        "eligible dynamic candidates:",
        total_dynamic_candidates,
    )

    print(
        "eligible static candidates:",
        total_static_candidates,
    )

    print(
        "samples containing static candidates:",
        samples_with_static_candidates,
    )

    print()
    print("errors:", len(errors))

    for error in errors[:50]:
        print(
            "ERROR:",
            error,
        )

    if errors:
        print()
        print(
            "=== DIFFUSIONDRIVE INDEX AUDIT FAIL ==="
        )
        raise SystemExit(1)

    if (
        total_static_candidates <= 0
        or samples_with_static_candidates <= 0
    ):
        print(
            "FAIL: static vehicles never reached "
            "DiffusionDrive agent candidate set"
        )
        raise SystemExit(1)

    print()
    print(
        "=== DIFFUSIONDRIVE INDEX AUDIT PASS ==="
    )


if __name__ == "__main__":
    main()
