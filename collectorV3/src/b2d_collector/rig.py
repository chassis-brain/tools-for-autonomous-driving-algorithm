from __future__ import annotations

import threading
import time
from typing import Dict, List

import numpy as np

from .sensors import auxiliary_sensor_specs


class AuxiliarySensorRig:
    """Spawn privileged annotation sensors on the ego vehicle without patching Leaderboard."""

    def __init__(self, profile: str, frequency_hz: float):
        self.specs = auxiliary_sensor_specs(profile)
        self.sensor_tick = 1.0 / frequency_hz
        self.actors: List = []
        self.latest: Dict[str, tuple] = {}
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.sync_timeout_s = 2.0
        self.started = False

    def _callback(self, sensor_id: str, sensor_type: str, data) -> None:
        if sensor_type.startswith("sensor.camera"):
            value = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((data.height, data.width, 4)).copy()
        elif sensor_type == "sensor.other.radar":
            value = np.frombuffer(data.raw_data, dtype=np.float32).reshape((-1, 4)).copy()
        elif sensor_type == "sensor.lidar.ray_cast":
            value = np.frombuffer(data.raw_data, dtype=np.float32).reshape((-1, 4)).copy()
        else:
            return
        with self.condition:
            self.latest[sensor_id] = (int(data.frame), value)
            self.condition.notify_all()

    def start(self) -> bool:
        if self.started or not self.specs:
            self.started = True
            return True
        try:
            import carla
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

            world = CarlaDataProvider.get_world()
            hero_getter = getattr(CarlaDataProvider, "get_hero_actor", None)
            ego = hero_getter() if hero_getter is not None else getattr(CarlaDataProvider, "_ego_actor", None)
            if ego is None and world is not None:
                candidates = [
                    actor for actor in world.get_actors().filter("vehicle.*")
                    if actor.attributes.get("role_name") in {"hero", "ego_vehicle"}
                ]
                ego = candidates[0] if candidates else None
            if world is None or ego is None:
                return False
            library = world.get_blueprint_library()
            for spec in self.specs:
                blueprint = library.find(spec["type"])
                for source, target in (("width", "image_size_x"), ("height", "image_size_y")):
                    if source in spec:
                        blueprint.set_attribute(target, str(spec[source]))
                for key in (
                    "fov", "range", "horizontal_fov", "vertical_fov", "rotation_frequency",
                    "channels", "points_per_second", "dropoff_general_rate",
                    "dropoff_intensity_limit", "dropoff_zero_intensity",
                ):
                    if key in spec:
                        blueprint.set_attribute(key, str(spec[key]))
                if blueprint.has_attribute("sensor_tick"):
                    blueprint.set_attribute("sensor_tick", str(self.sensor_tick))
                blueprint.set_attribute("role_name", spec["id"])
                transform = carla.Transform(
                    carla.Location(x=spec["x"], y=spec["y"], z=spec["z"]),
                    carla.Rotation(roll=spec.get("roll", 0.0), pitch=spec.get("pitch", 0.0), yaw=spec.get("yaw", 0.0)),
                )
                actor = world.spawn_actor(blueprint, transform, attach_to=ego)
                actor.listen(lambda data, sid=spec["id"], st=spec["type"]: self._callback(sid, st, data))
                self.actors.append(actor)
            self.started = True
            return True
        except Exception:
            self.destroy()
            raise

    def enrich(self, input_data: Dict) -> bool:
        if not self.start():
            return False
        if not self.specs:
            return True

        primary_frames = [
            int(item[0])
            for item in input_data.values()
            if isinstance(item, (tuple, list))
        ]
        expected_frame = max(primary_frames) if primary_frames else None

        if expected_frame is None:
            return False

        required_ids = [spec["id"] for spec in self.specs]
        deadline = time.monotonic() + self.sync_timeout_s

        with self.condition:
            while True:
                missing = [
                    sensor_id
                    for sensor_id in required_ids
                    if sensor_id not in self.latest
                ]

                wrong_frame = [
                    sensor_id
                    for sensor_id in required_ids
                    if sensor_id in self.latest
                    and self.latest[sensor_id][0] != expected_frame
                ]

                if not missing and not wrong_frame:
                    input_data.update(
                        {
                            sensor_id: self.latest[sensor_id]
                            for sensor_id in required_ids
                        }
                    )
                    return True

                # If a sensor has already advanced beyond the requested
                # frame, that exact frame can no longer be assembled.
                if any(
                    sensor_id in self.latest
                    and self.latest[sensor_id][0] > expected_frame
                    for sensor_id in required_ids
                ):
                    return False

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False

                self.condition.wait(timeout=remaining)

    def destroy(self) -> None:
        # Detach our Python-side ownership first. destroy() may be called
        # repeatedly during Leaderboard / ScenarioRunner teardown.
        actors = self.actors
        self.actors = []
        self.started = False

        with self.condition:
            self.latest.clear()
            self.condition.notify_all()

        if not actors:
            return

        # Do not trust a stale Actor proxy as proof that the server-side
        # actor still exists. ScenarioRunner may already have destroyed
        # the ego vehicle and all sensors attached to it.
        #
        # Query the world's current actor list once and only issue
        # stop/destroy RPCs for IDs that are still present there.
        try:
            world = actors[0].get_world()
            live_actors = {
                int(actor.id): actor
                for actor in world.get_actors()
            }
        except Exception:
            # At this point teardown owns the world lifecycle. If the
            # server/world can no longer be queried, simply release all
            # local references instead of sending risky cleanup RPCs.
            return

        for old_actor in reversed(actors):
            try:
                actor_id = int(old_actor.id)
            except Exception:
                continue

            actor = live_actors.get(actor_id)

            # Parent/ScenarioRunner already destroyed this sensor.
            if actor is None:
                continue

            try:
                listening = getattr(
                    actor,
                    "is_listening",
                    False,
                )

                if callable(listening):
                    listening = listening()

                if listening:
                    actor.stop()

            except Exception:
                pass

            try:
                actor.destroy()
            except Exception:
                # A concurrent teardown may remove the actor between the
                # actor-list snapshot and DestroyActor. Never let teardown
                # change the route result.
                pass
