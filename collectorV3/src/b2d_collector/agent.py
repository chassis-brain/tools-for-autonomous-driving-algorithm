from __future__ import annotations

import math
import os
from typing import Optional

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from .config import CollectorConfig
from .controller import ControlMux, ExpertAdapter, KeyboardController
from .navigation import RouteProgress
from .route_xml import RouteCatalog
from .rig import AuxiliarySensorRig
from .sensors import collection_sensors, merge_sensor_specs
from .writer import Bench2DriveWriter




def _finalize_vehicle_control(proposed):
    """Return the exact legal VehicleControl that will be recorded/executed."""

    throttle = float(proposed.throttle)
    steer = float(proposed.steer)
    brake = float(proposed.brake)

    values = {
        "throttle": throttle,
        "steer": steer,
        "brake": brake,
    }

    for name, value in values.items():
        if not math.isfinite(value):
            raise RuntimeError(
                "Non-finite control value: %s=%r"
                % (name, value)
            )

    executed = carla.VehicleControl(
        throttle=max(0.0, min(1.0, throttle)),
        steer=max(-1.0, min(1.0, steer)),
        brake=max(0.0, min(1.0, brake)),
        hand_brake=bool(proposed.hand_brake),
        reverse=bool(proposed.reverse),
        manual_gear_shift=bool(proposed.manual_gear_shift),
        gear=int(proposed.gear),
    )

    clipped = (
        executed.throttle != throttle
        or executed.steer != steer
        or executed.brake != brake
    )

    return executed, clipped, values

def get_entry_point():
    return "ManualExpertCollectorAgent"


class ManualExpertCollectorAgent(AutonomousAgent):
    """Leaderboard agent that controls the ego vehicle and records Bench2Drive-compatible frames."""

    track = Track.SENSORS

    def __init__(self, carla_host, carla_port, debug=False):
        super().__init__(carla_host, carla_port, debug)

        # Bench2Drive calls set_global_plan() BEFORE setup(),
        # so anything needed by set_global_plan must be initialized here.
        self.progress = RouteProgress()
        self.expert: Optional[ExpertAdapter] = None

        # Preserve the raw route so an expert created later in setup()
        # can also receive the global plan.
        self._collector_global_plan_gps = None
        self._collector_global_plan_world_coord = None

        # Bench2Drive appends "+<save_name>" to agent-config.
        self.run_name = ""

        # Dataset synchronization gate.
        # Do not write training data until auxiliary sensors have been
        # continuously synchronized for several consecutive simulation frames.
        self._sync_warmup_frames = int(
            os.environ.get("B2D_SYNC_WARMUP_FRAMES", "5")
        )
        self._sync_streak = 0
        self._recording_started = False
        self._last_record_timestamp = None
        self._control_clip_count = 0

    def setup(self, path_to_conf_file):
        # Bench2Drive passes:
        #   /path/to/manual.yaml+RouteScenario_...
        # while CollectorConfig needs the actual YAML path.
        config_path, _, self.run_name = path_to_conf_file.partition("+")
        self.config = CollectorConfig.load(config_path)

        self.track = (
            Track.MAP
            if self.config.mode in {"expert", "hybrid"}
            else Track.SENSORS
        )

        self.expert = None
        if self.config.mode in {"expert", "hybrid"}:
            self.expert = ExpertAdapter(
                self.config.expert.module,
                self.config.expert.entry_point,
                self.config.expert.config,
            )

            # set_global_plan() runs before setup() in Bench2Drive.
            if self._collector_global_plan_gps is not None:
                self.expert.set_global_plan(
                    self._collector_global_plan_gps,
                    self._collector_global_plan_world_coord,
                )

        self.keyboard = KeyboardController(self.config.display)
        self.control_mux = ControlMux(self.config.mode, self.keyboard, self.expert)
        route = RouteCatalog(self.config.route_xml).get(self.config.route_id)
        self.writer = Bench2DriveWriter(
            self.config.output_root,
            route,
            self.config.weather_id,
            self.config.frequency_hz,
            self.config.jpeg_quality,
            self.config.sensor_profile,
        )
        self.rig = AuxiliarySensorRig(self.config.sensor_profile, self.config.frequency_hz)

    def sensors(self):
        ours = collection_sensors(self.config.sensor_profile)
        return merge_sensor_specs(ours, self.expert.sensors() if self.expert is not None else [])

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        self._collector_global_plan_gps = global_plan_gps
        self._collector_global_plan_world_coord = global_plan_world_coord

        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self.progress.set_plan(global_plan_world_coord)

        if self.expert is not None:
            self.expert.set_global_plan(global_plan_gps, global_plan_world_coord)

    def run_step(self, input_data, timestamp):
        timestamp = float(timestamp)
        auxiliary_ready = self.rig.enrich(input_data)

        proposed_control, source = self.control_mux.step(
            input_data,
            timestamp,
        )

        control, clipped, raw_control = _finalize_vehicle_control(
            proposed_control
        )

        if clipped:
            self._control_clip_count += 1

            if self._control_clip_count <= 3:
                print(
                    "[Collector] control clamped to CARLA range: "
                    "throttle={:.6f} steer={:.6f} brake={:.6f}".format(
                        raw_control["throttle"],
                        raw_control["steer"],
                        raw_control["brake"],
                    ),
                    flush=True,
                )

        # Finalize emergency/quit control BEFORE recording so that the
        # recorded action is exactly the VehicleControl returned to CARLA.
        if self.keyboard.quit_requested:
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = True

        self.keyboard.render(input_data, source, timestamp)
        for event_time, event, active in self.keyboard.pop_events():
            self.writer.append_event(event_time, event, active)
        if self.keyboard.paused:
            if self._recording_started:
                raise RuntimeError(
                    "Recording was paused after dataset capture started; "
                    "aborting to avoid a timestamp gap."
                )
            self._sync_streak = 0

        elif not auxiliary_ready:
            if self._recording_started:
                raise RuntimeError(
                    "Auxiliary sensor synchronization was lost after "
                    "dataset capture started; aborting this clip."
                )
            self._sync_streak = 0

        else:
            self._sync_streak += 1

            # Warm up until several consecutive fully synchronized frames
            # have been observed.
            if (
                self._recording_started
                or self._sync_streak >= self._sync_warmup_frames
            ):
                if not self._recording_started:
                    self._recording_started = True
                    print(
                        "[Collector] Sensor synchronization stable; "
                        "dataset recording starts now.",
                        flush=True,
                    )

                expected_dt = 1.0 / float(self.config.frequency_hz)

                if self._last_record_timestamp is not None:
                    dt = timestamp - self._last_record_timestamp

                    if abs(dt - expected_dt) > 0.001:
                        raise RuntimeError(
                            "Dataset timestamp discontinuity: "
                            "expected {:.6f}s, got {:.6f}s".format(
                                expected_dt, dt
                            )
                        )

                nav = self.progress.update(
                    self.writer.annotator.ego_location()
                )
                assessment = (
                    self.expert.assessment()
                    if self.expert is not None
                    else None
                )

                self.writer.record(
                    input_data,
                    control,
                    timestamp,
                    nav,
                    source,
                    self.keyboard.failure_active,
                    assessment,
                )

                self._last_record_timestamp = timestamp
        return control

    def destroy(self):
        clip_count = getattr(self, "_control_clip_count", 0)
        if clip_count:
            print(
                "[Collector] finalized/clamped control frames: %d"
                % clip_count,
                flush=True,
            )

        if getattr(self, "rig", None) is not None:
            self.rig.destroy()
        if getattr(self, "expert", None) is not None:
            self.expert.destroy()
        if getattr(self, "keyboard", None) is not None:
            self.keyboard.destroy()
