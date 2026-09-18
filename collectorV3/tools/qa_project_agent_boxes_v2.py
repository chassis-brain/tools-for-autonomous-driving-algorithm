from __future__ import print_function

import argparse
import gzip
import json
import math
from pathlib import Path

import cv2
import numpy as np


EDGES = (
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
)


def carla_rotation_matrix(rotation):
    pitch, roll, yaw = [
        math.radians(float(x))
        for x in rotation
    ]

    cp = math.cos(pitch)
    sp = math.sin(pitch)

    cr = math.cos(roll)
    sr = math.sin(roll)

    cy = math.cos(yaw)
    sy = math.sin(yaw)

    return np.asarray(
        [
            [
                cp * cy,
                cy * sp * sr - sy * cr,
                -cy * sp * cr - sy * sr,
            ],
            [
                cp * sy,
                sy * sp * sr + cy * cr,
                -sy * sp * cr + cy * sr,
            ],
            [
                sp,
                -cp * sr,
                cp * cr,
            ],
        ],
        dtype=np.float64,
    )


def box_world_corners(box):
    # Static vehicle geometry from the patch is already
    # saved directly in world coordinates.
    world_cord = box.get(
        "world_cord"
    )

    if (
        box.get("state") == "static"
        and world_cord is not None
    ):
        corners = np.asarray(
            world_cord,
            dtype=np.float64,
        )

        if (
            corners.shape == (8, 3)
            and np.isfinite(
                corners
            ).all()
        ):
            return corners

    # Dynamic vehicles keep using the geometry that
    # already passed our previous projection QA.
    center = np.asarray(
        box["center"],
        dtype=np.float64,
    )

    extent = np.asarray(
        box["extent"],
        dtype=np.float64,
    )

    R = carla_rotation_matrix(
        box["rotation"]
    )

    ex, ey, ez = extent.tolist()

    local = np.asarray(
        [
            [-ex, -ey, -ez],
            [-ex, -ey, +ez],
            [-ex, +ey, -ez],
            [-ex, +ey, +ez],
            [+ex, -ey, -ez],
            [+ex, -ey, +ez],
            [+ex, +ey, -ez],
            [+ex, +ey, +ez],
        ],
        dtype=np.float64,
    )

    return (
        center[None, :]
        + (R @ local.T).T
    )


def world_to_camera_cv(
    points_world,
    world2cam,
):
    ones = np.ones(
        (len(points_world), 1),
        dtype=np.float64,
    )

    points_h = np.concatenate(
        [
            points_world,
            ones,
        ],
        axis=1,
    )

    ue = (
        world2cam
        @ points_h.T
    ).T[:, :3]

    # CARLA/UE:
    # x forward, y right, z up
    #
    # CV:
    # X right, Y down, Z forward
    return np.column_stack(
        [
            ue[:, 1],
            -ue[:, 2],
            ue[:, 0],
        ]
    )


def project(points_cv, K):
    uvw = (
        K
        @ points_cv.T
    ).T

    uv = np.empty(
        (len(points_cv), 2),
        dtype=np.float64,
    )

    uv[:, 0] = (
        uvw[:, 0]
        / uvw[:, 2]
    )

    uv[:, 1] = (
        uvw[:, 1]
        / uvw[:, 2]
    )

    return uv


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clip",
        required=True,
    )

    parser.add_argument(
        "--frame",
        default="00080",
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=85.0,
    )

    args = parser.parse_args()

    clip = Path(args.clip)
    frame = args.frame

    anno_path = (
        clip
        / "anno"
        / (frame + ".json.gz")
    )

    rgb_path = (
        clip
        / "camera"
        / "rgb_front"
        / (frame + ".jpg")
    )

    out_dir = clip / "qa"

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        out_dir
        / (
            "agent_boxes_front_v2_"
            + frame
            + ".jpg"
        )
    )

    with gzip.open(
        str(anno_path),
        "rt",
    ) as f:
        anno = json.load(f)

    image = cv2.imread(
        str(rgb_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            "failed to read RGB: %s"
            % rgb_path
        )

    cam = (
        anno["sensors"][
            "CAM_FRONT"
        ]
    )

    K = np.asarray(
        cam["intrinsic"],
        dtype=np.float64,
    )

    world2cam = np.asarray(
        cam["world2cam"],
        dtype=np.float64,
    )

    height, width = (
        image.shape[:2]
    )

    counts = {
        "dynamic": 0,
        "static": 0,
        "walker": 0,
        "visible_dynamic": 0,
        "visible_static": 0,
        "visible_walker": 0,
    }

    for box in anno.get(
        "bounding_boxes",
        [],
    ):
        class_name = box.get(
            "class"
        )

        if class_name not in (
            "vehicle",
            "walker",
        ):
            continue

        # Ego.
        if "world2ego" in box:
            continue

        distance = float(
            box.get(
                "distance",
                1e9,
            )
        )

        if (
            distance
            > args.max_distance
        ):
            continue

        if class_name == "vehicle":
            state = box.get(
                "state",
                "dynamic",
            )

            if state == "static":
                key = "static"
            else:
                key = "dynamic"
        else:
            key = "walker"

        counts[key] += 1

        corners_world = (
            box_world_corners(box)
        )

        corners_cv = (
            world_to_camera_cv(
                corners_world,
                world2cam,
            )
        )

        if np.all(
            corners_cv[:, 2]
            <= 0.05
        ):
            continue

        uv = project(
            corners_cv,
            K,
        )

        # BGR:
        # dynamic = green
        # static  = orange
        # walker  = cyan
        if key == "static":
            color = (
                0,
                165,
                255,
            )

        elif key == "dynamic":
            color = (
                0,
                255,
                0,
            )

        else:
            color = (
                255,
                255,
                0,
            )

        any_line = False

        for i0, i1 in EDGES:
            if (
                corners_cv[
                    i0,
                    2,
                ] <= 0.05
                or corners_cv[
                    i1,
                    2,
                ] <= 0.05
            ):
                continue

            p0 = (
                int(
                    round(
                        uv[i0, 0]
                    )
                ),
                int(
                    round(
                        uv[i0, 1]
                    )
                ),
            )

            p1 = (
                int(
                    round(
                        uv[i1, 0]
                    )
                ),
                int(
                    round(
                        uv[i1, 1]
                    )
                ),
            )

            ok, q0, q1 = (
                cv2.clipLine(
                    (
                        0,
                        0,
                        width,
                        height,
                    ),
                    p0,
                    p1,
                )
            )

            if not ok:
                continue

            cv2.line(
                image,
                q0,
                q1,
                color,
                2,
                cv2.LINE_AA,
            )

            any_line = True

        if not any_line:
            continue

        counts[
            "visible_" + key
        ] += 1

        valid = (
            corners_cv[:, 2]
            > 0.05
        )

        if valid.any():
            uv_valid = uv[valid]

            x = int(
                np.clip(
                    np.min(
                        uv_valid[:, 0]
                    ),
                    0,
                    width - 1,
                )
            )

            y = int(
                np.clip(
                    np.min(
                        uv_valid[:, 1]
                    ) - 5,
                    20,
                    height - 1,
                )
            )

            if key == "static":
                label = (
                    "STATIC id=%s %.1fm"
                    % (
                        box.get(
                            "id",
                            "?",
                        ),
                        distance,
                    )
                )

            elif key == "dynamic":
                label = (
                    "DYNAMIC id=%s %.1fm"
                    % (
                        box.get(
                            "id",
                            "?",
                        ),
                        distance,
                    )
                )

            else:
                label = (
                    "WALKER id=%s %.1fm"
                    % (
                        box.get(
                            "id",
                            "?",
                        ),
                        distance,
                    )
                )

            cv2.putText(
                image,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        image,
        (
            "green dynamic | "
            "orange static | "
            "cyan walker"
        ),
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    if not cv2.imwrite(
        str(out_path),
        image,
    ):
        raise RuntimeError(
            "failed to write: %s"
            % out_path
        )

    print("=" * 72)
    print("AGENT BOX QA V2")
    print("=" * 72)

    print(
        "frame:",
        frame,
    )

    print(
        "dynamic annotated:",
        counts["dynamic"],
    )

    print(
        "static annotated:",
        counts["static"],
    )

    print(
        "walker annotated:",
        counts["walker"],
    )

    print(
        "visible dynamic:",
        counts[
            "visible_dynamic"
        ],
    )

    print(
        "visible static:",
        counts[
            "visible_static"
        ],
    )

    print(
        "visible walker:",
        counts[
            "visible_walker"
        ],
    )

    print(
        "output:",
        out_path,
    )

    print(
        "=== AGENT BOX QA V2 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
