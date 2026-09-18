from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .sensors import CAMERAS


def _xyz(value: Any) -> List[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _rotation(value: Any) -> List[float]:
    return [float(value.pitch), float(value.roll), float(value.yaw)]


def _speed(actor: Any) -> float:
    velocity = actor.get_velocity()
    return float(math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))


def _matrix(transform: Any, inverse: bool = False):
    getter = transform.get_inverse_matrix if inverse else transform.get_matrix
    return np.asarray(getter(), dtype=np.float64).tolist()


class WorldAnnotator:
    """Privileged training annotations. Returns empty annotations when ScenarioRunner is unavailable."""

    def __init__(self) -> None:
        self.world = None
        self.ego = None
        self._ensure()

    def _ensure(self) -> None:
        if self.world is not None and self.ego is not None:
            return
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

            self.world = CarlaDataProvider.get_world()
            hero_getter = getattr(CarlaDataProvider, "get_hero_actor", None)
            self.ego = hero_getter() if hero_getter is not None else getattr(CarlaDataProvider, "_ego_actor", None)
            if self.ego is None and self.world is not None:
                candidates = [
                    actor for actor in self.world.get_actors().filter("vehicle.*")
                    if actor.attributes.get("role_name") in {"hero", "ego_vehicle"}
                ]
                self.ego = candidates[0] if candidates else None
        except Exception:
            pass

    def ego_location(self) -> Optional[Any]:
        self._ensure()
        return None if self.ego is None else self.ego.get_location()

    def ego_state(self) -> Dict[str, Any]:
        """Return raw ego kinematics needed to build trajectory-learning targets offline."""
        self._ensure()
        if self.ego is None:
            return {}
        transform = self.ego.get_transform()
        velocity = self.ego.get_velocity()
        acceleration = self.ego.get_acceleration()
        angular_velocity = self.ego.get_angular_velocity()
        return {
            "location": _xyz(transform.location),
            "rotation": _rotation(transform.rotation),
            "ego2world": _matrix(transform, inverse=False),
            "world2ego": _matrix(transform, inverse=True),
            "velocity": _xyz(velocity),
            "acceleration": _xyz(acceleration),
            "angular_velocity": _xyz(angular_velocity),
        }

    def sensor_calibration(self) -> Dict[str, dict]:
        self._ensure()
        if self.world is None or self.ego is None:
            return {}

        result = {}
        ego_inverse = np.asarray(
            self.ego.get_transform().get_inverse_matrix(),
            dtype=np.float64,
        )

        # Leaderboard does not reliably preserve the collector's declared
        # IDs in actor.role_name for the six primary RGB cameras.
        #
        # Recover their canonical IDs from their unique ego-relative
        # mounting positions defined in sensors.CAMERAS.
        expected_rgb = {
            "CAM_%s" % name: np.asarray(
                [x, y, z],
                dtype=np.float64,
            )
            for name, x, y, z, _yaw, _fov in CAMERAS
        }

        expected_rgb_fov = {
            "CAM_%s" % name: float(fov)
            for name, _x, _y, _z, _yaw, fov in CAMERAS
        }

        matched_primary_rgb = set()

        for actor in self.world.get_actors().filter("sensor.*"):
            transform = actor.get_transform()
            lower = actor.type_id.lower()

            # Calibration is a geometric sensor map.
            # GNSS / IMU / other non-camera sensors do not belong here.
            if not (
                "camera" in lower
                or "lidar" in lower
                or "radar" in lower
            ):
                continue

            # Keep role_name for auxiliary geometric sensors.
            sensor_id = actor.attributes.get("role_name", "")

            cam2ego = None

            if "camera" in lower:
                cam2ego = (
                    ego_inverse
                    @ np.asarray(
                        transform.get_matrix(),
                        dtype=np.float64,
                    )
                )

            # Primary Leaderboard RGB cameras need canonical-ID recovery.
            if (
                lower == "sensor.camera.rgb"
                and cam2ego is not None
            ):
                fov = float(
                    actor.attributes.get("fov", 90.0)
                )

                position = cam2ego[:3, 3]

                candidates = []

                for canonical_id, expected_position in expected_rgb.items():
                    if canonical_id in matched_primary_rgb:
                        continue

                    distance = float(
                        np.linalg.norm(
                            position - expected_position
                        )
                    )

                    fov_error = abs(
                        fov
                        - expected_rgb_fov[canonical_id]
                    )

                    candidates.append(
                        (
                            distance,
                            fov_error,
                            canonical_id,
                        )
                    )

                if candidates:
                    distance, fov_error, canonical_id = min(
                        candidates
                    )

                    # Camera mounts are separated by >= 0.59 m.
                    # 0.15 m is therefore conservative while allowing
                    # tiny CARLA floating-point differences.
                    if (
                        distance <= 0.15
                        and fov_error <= 0.5
                    ):
                        sensor_id = canonical_id
                        matched_primary_rgb.add(
                            canonical_id
                        )

            if not sensor_id:
                continue

            item = {
                "location": _xyz(transform.location),
                "rotation": _rotation(transform.rotation),
            }

            if "camera" in lower:
                width = int(
                    actor.attributes.get(
                        "image_size_x",
                        0,
                    )
                )
                height = int(
                    actor.attributes.get(
                        "image_size_y",
                        0,
                    )
                )
                fov = float(
                    actor.attributes.get(
                        "fov",
                        90.0,
                    )
                )

                focal = (
                    width
                    / (
                        2.0
                        * math.tan(
                            math.radians(fov) / 2.0
                        )
                    )
                    if width
                    else 0.0
                )

                if cam2ego is None:
                    cam2ego = (
                        ego_inverse
                        @ np.asarray(
                            transform.get_matrix(),
                            dtype=np.float64,
                        )
                    )

                item.update(
                    {
                        "intrinsic": [
                            [
                                focal,
                                0.0,
                                width / 2.0,
                            ],
                            [
                                0.0,
                                focal,
                                height / 2.0,
                            ],
                            [
                                0.0,
                                0.0,
                                1.0,
                            ],
                        ],
                        "world2cam": _matrix(
                            transform,
                            inverse=True,
                        ),
                        "cam2ego": cam2ego.tolist(),
                        "fov": fov,
                        "image_size_x": width,
                        "image_size_y": height,
                    }
                )

            elif "lidar" in lower:
                item["world2lidar"] = _matrix(
                    transform,
                    inverse=True,
                )

                item["lidar2ego"] = (
                    ego_inverse
                    @ np.asarray(
                        transform.get_matrix(),
                        dtype=np.float64,
                    )
                ).tolist()

            elif "radar" in lower:
                item["world2radar"] = _matrix(
                    transform,
                    inverse=True,
                )

                item["radar2ego"] = (
                    ego_inverse
                    @ np.asarray(
                        transform.get_matrix(),
                        dtype=np.float64,
                    )
                ).tolist()

            result[sensor_id] = item

        # Data correctness is more important than silently continuing.
        # Once recording begins, all six primary RGB camera calibrations
        # must be recoverable under their canonical IDs.
        required_rgb = {
            "CAM_%s" % camera[0]
            for camera in CAMERAS
        }

        missing_rgb = sorted(
            required_rgb - set(result)
        )

        if missing_rgb:
            raise RuntimeError(
                "Primary RGB calibration missing canonical sensors: %s"
                % ", ".join(missing_rgb)
            )

        return result

    def weather(self) -> Dict[str, float]:
        self._ensure()
        if self.world is None:
            return {}
        weather = self.world.get_weather()
        names = (
            "cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
            "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance",
            "fog_falloff", "wetness", "scattering_intensity", "mie_scattering_scale",
            "rayleigh_scattering_scale", "dust_storm",
        )
        return {name: float(getattr(weather, name)) for name in names if hasattr(weather, name)}

    # STATIC_VEHICLE_ANNOTATION_V1
    def bounding_boxes(self, max_distance: float = 85.0) -> List[dict]:
        self._ensure()

        if self.world is None or self.ego is None:
            return []

        ego_loc = self.ego.get_location()

        boxes = [
            self._actor_box(
                self.ego,
                "vehicle",
                ego_loc,
                is_ego=True,
            )
        ]

        filters = (
            ("vehicle.*", "vehicle"),
            ("walker.pedestrian.*", "walker"),
            (
                "traffic.traffic_light*",
                "traffic_light",
            ),
            ("traffic.stop*", "traffic_sign"),
            (
                "traffic.speed_limit*",
                "traffic_sign",
            ),
        )

        seen = {self.ego.id}

        for actor_filter, class_name in filters:
            actors = (
                self.world
                .get_actors()
                .filter(actor_filter)
            )

            for actor in actors:
                if actor.id in seen:
                    continue

                if (
                    actor.get_location()
                    .distance(ego_loc)
                    > max_distance
                ):
                    continue

                seen.add(actor.id)

                boxes.append(
                    self._actor_box(
                        actor,
                        class_name,
                        ego_loc,
                    )
                )

        # Bench2Drive parked vehicles are often represented
        # as static.prop.mesh instead of vehicle.* actors.
        boxes.extend(
            self._static_vehicle_boxes(
                ego_loc,
                max_distance,
            )
        )

        return boxes

    @staticmethod
    def _distance_2d(a: Any, b: Any) -> float:
        dx = float(a.x) - float(b.x)
        dy = float(a.y) - float(b.y)

        return float(
            math.sqrt(
                dx * dx + dy * dy
            )
        )

    @staticmethod
    def _static_vehicle_kind(
        actor: Any,
    ) -> Optional[str]:
        mesh_path = str(
            actor.attributes.get(
                "mesh_path",
                "",
            )
        )

        if not mesh_path:
            return None

        # Remove "/Game/Carla/" first so that
        # "Carla" itself cannot accidentally match "car".
        relative = mesh_path.split(
            "/Game/Carla/",
            1,
        )[-1].lower()

        if "motorcycle" in relative:
            return "Motorcycle"

        if "bicycle" in relative:
            return "Bicycle"

        if "truck" in relative:
            return "Truck"

        if "bus" in relative:
            return "Bus"

        if "train" in relative:
            return "Train"

        if "car" in relative:
            return "Car"

        return None

    @staticmethod
    def _level_bbox_geometry(
        level_bbox: Any,
    ):
        """
        Build a self-consistent world-space box.

        CARLA get_level_bbs() returns a bounding box whose
        location/rotation describe the world-level object.

        The returned transform is the box coordinate frame.
        """
        import carla

        transform = carla.Transform(
            level_bbox.location,
            level_bbox.rotation,
        )

        ex = float(level_bbox.extent.x)
        ey = float(level_bbox.extent.y)
        ez = float(level_bbox.extent.z)

        local = (
            (-ex, -ey, -ez),
            (-ex, -ey, +ez),
            (-ex, +ey, -ez),
            (-ex, +ey, +ez),
            (+ex, -ey, -ez),
            (+ex, -ey, +ez),
            (+ex, +ey, -ez),
            (+ex, +ey, +ez),
        )

        world_cord = []

        for x, y, z in local:
            p = transform.transform(
                carla.Location(
                    x=x,
                    y=y,
                    z=z,
                )
            )

            world_cord.append(
                [
                    float(p.x),
                    float(p.y),
                    float(p.z),
                ]
            )

        return transform, world_cord

    def _static_vehicle_boxes(
        self,
        ego_loc: Any,
        max_distance: float,
    ) -> List[dict]:
        try:
            import carla
        except Exception:
            return []

        labels = {}

        for name in (
            "Car",
            "Bicycle",
            "Bus",
            "Motorcycle",
            "Train",
            "Truck",
        ):
            label = getattr(
                carla.CityObjectLabel,
                name,
                None,
            )

            if label is None:
                continue

            try:
                labels[name] = list(
                    self.world.get_level_bbs(
                        label
                    )
                )
            except Exception:
                labels[name] = []

        # Keep a generous association margin.
        # Bench2Drive's official collector also uses
        # a 20 m static actor <-> level-bbox association
        # threshold.
        nearby_limit = (
            float(max_distance) + 20.0
        )

        remaining = {}

        for name, bbs in labels.items():
            remaining[name] = [
                bb
                for bb in bbs
                if self._distance_2d(
                    bb.location,
                    ego_loc,
                )
                <= nearby_limit
            ]

        result = []

        static_actors = (
            self.world
            .get_actors()
            .filter("*static.prop.mesh*")
        )

        for actor in static_actors:
            if not getattr(
                actor,
                "is_alive",
                True,
            ):
                continue

            kind = self._static_vehicle_kind(
                actor
            )

            if kind is None:
                continue

            actor_transform = (
                actor.get_transform()
            )

            actor_loc = (
                actor_transform.location
            )

            if (
                self._distance_2d(
                    actor_loc,
                    ego_loc,
                )
                > float(max_distance)
            ):
                continue

            candidates = remaining.get(
                kind,
                [],
            )

            if not candidates:
                continue

            best_index = -1
            best_distance = float("inf")

            for index, level_bbox in enumerate(
                candidates
            ):
                distance = (
                    self._distance_2d(
                        actor_loc,
                        level_bbox.location,
                    )
                )

                if distance < best_distance:
                    best_distance = distance
                    best_index = index

            if (
                best_index < 0
                or best_distance > 20.0
            ):
                continue

            # One level bbox can only be associated
            # with one static actor per frame.
            level_bbox = candidates.pop(
                best_index
            )

            result.append(
                self._static_vehicle_box(
                    actor=actor,
                    level_bbox=level_bbox,
                    ego_loc=ego_loc,
                    match_distance=(
                        best_distance
                    ),
                )
            )

        return result

    def _static_vehicle_box(
        self,
        actor: Any,
        level_bbox: Any,
        ego_loc: Any,
        match_distance: float,
    ) -> dict:
        (
            bbox_transform,
            world_cord,
        ) = self._level_bbox_geometry(
            level_bbox
        )

        waypoint = None

        try:
            waypoint = (
                self.world
                .get_map()
                .get_waypoint(
                    level_bbox.location
                )
            )
        except Exception:
            pass

        actor_transform = (
            actor.get_transform()
        )

        actor_box = getattr(
            actor,
            "bounding_box",
            None,
        )

        mesh_path = str(
            actor.attributes.get(
                "mesh_path",
                actor.type_id,
            )
        )

        result = {
            "class": "vehicle",
            "state": "static",
            "id": str(actor.id),
            "type_id": mesh_path,

            # Authoritative static geometry:
            "location": _xyz(
                level_bbox.location
            ),
            "rotation": _rotation(
                level_bbox.rotation
            ),
            "center": _xyz(
                level_bbox.location
            ),
            "extent": _xyz(
                level_bbox.extent
            ),
            "bbox_rotation": _rotation(
                level_bbox.rotation
            ),

            # The static box coordinate frame is
            # centered at the world-level bbox.
            "bbx_loc": [
                0.0,
                0.0,
                0.0,
            ],

            "world_cord": world_cord,

            "world2vehicle": _matrix(
                bbox_transform,
                inverse=True,
            ),

            "distance": (
                self._distance_2d(
                    level_bbox.location,
                    ego_loc,
                )
            ),

            "speed": 0.0,

            "road_id": getattr(
                waypoint,
                "road_id",
                None,
            ),

            "lane_id": getattr(
                waypoint,
                "lane_id",
                None,
            ),

            "section_id": getattr(
                waypoint,
                "section_id",
                None,
            ),

            "bbox_source":
                "carla_world_level_bb",

            "bbox_match_distance":
                float(match_distance),

            # Raw static actor pose is retained
            # only for diagnostics/provenance.
            "actor_location": _xyz(
                actor_transform.location
            ),

            "actor_rotation": _rotation(
                actor_transform.rotation
            ),
        }

        if actor_box is not None:
            result["actor_bbx_loc"] = _xyz(
                actor_box.location
            )

            result[
                "actor_bbox_extent"
            ] = _xyz(
                actor_box.extent
            )

        return result

    def _actor_box(
        self,
        actor: Any,
        class_name: str,
        ego_loc: Any,
        is_ego: bool = False,
    ) -> dict:
        transform = actor.get_transform()

        waypoint = None

        try:
            waypoint = (
                self.world
                .get_map()
                .get_waypoint(
                    transform.location
                )
            )
        except Exception:
            pass

        box = getattr(
            actor,
            "bounding_box",
            None,
        )

        center = (
            transform.location
            if box is None
            else transform.transform(
                box.location
            )
        )

        result = {
            "class": class_name,
            "id": str(actor.id),
            "type_id": actor.type_id,
            "location": _xyz(
                transform.location
            ),
            "rotation": _rotation(
                transform.rotation
            ),
            "center": _xyz(center),

            "extent": (
                _xyz(box.extent)
                if box is not None
                else [
                    0.0,
                    0.0,
                    0.0,
                ]
            ),

            "distance": float(
                transform.location.distance(
                    ego_loc
                )
            ),

            "speed": (
                _speed(actor)
                if hasattr(
                    actor,
                    "get_velocity",
                )
                else 0.0
            ),

            "road_id": getattr(
                waypoint,
                "road_id",
                None,
            ),

            "lane_id": getattr(
                waypoint,
                "lane_id",
                None,
            ),

            "section_id": getattr(
                waypoint,
                "section_id",
                None,
            ),
        }

        if class_name == "vehicle":
            result["state"] = "dynamic"

            if box is not None:
                result["bbx_loc"] = _xyz(
                    box.location
                )

            result["world2vehicle"] = (
                _matrix(
                    transform,
                    inverse=True,
                )
            )

        if is_ego:
            result["world2ego"] = (
                _matrix(
                    transform,
                    inverse=True,
                )
            )

        if class_name == "traffic_light":
            result["state"] = (
                str(actor.state)
                .split(".")[-1]
                .lower()
            )

        return result
