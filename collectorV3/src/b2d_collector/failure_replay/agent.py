from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from b2d_collector.agent import _finalize_vehicle_control
from b2d_collector.controller import ExpertAdapter
from b2d_collector.navigation import RouteProgress
from b2d_collector.rig import AuxiliarySensorRig
from b2d_collector.route_xml import RouteCatalog
from b2d_collector.sensors import collection_sensors, merge_sensor_specs
from b2d_collector.writer import Bench2DriveWriter

from .config import ReplayCollectorConfig
from .spec import InterventionSpec, load_intervention_spec, normalize_replacement_mode
from .manual_pid import ManualTrajectoryFollower
from .tape import BehaviorTape, TapeFrame


# Frozen by the 1711 golden replay validation:
# CARLA applies the VehicleControl returned at world tick k to the next physics
# state (k + 1). Replay therefore selects the recorded control one tape tick
# ahead of the current CARLA world timestamp.
REPLAY_CONTROL_SHIFT_TICKS = 1
_EPS = 1e-6


def get_entry_point():
    return "ReplayToPdmCollectorAgent"


def _neutral_control() -> carla.VehicleControl:
    return carla.VehicleControl(
        throttle=0.0,
        steer=0.0,
        brake=0.0,
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False,
        gear=0,
    )


def _control_from_tape(frame: Optional[TapeFrame]) -> carla.VehicleControl:
    if frame is None:
        return _neutral_control()
    control = frame.control
    return carla.VehicleControl(
        throttle=float(control.get("throttle", 0.0)),
        steer=float(control.get("steer", 0.0)),
        brake=float(control.get("brake", 0.0)),
        hand_brake=bool(control.get("hand_brake", False)),
        reverse=bool(control.get("reverse", False)),
        manual_gear_shift=bool(control.get("manual_gear_shift", False)),
        gear=int(control.get("gear", 0)),
    )


def _yaw_error_deg(a: float, b: float) -> float:
    delta = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(delta)


class ReplayToPdmCollectorAgent(AutonomousAgent):
    """Replay recorded E2E controls, then hand control to PDM and collect Base data.

    Before handoff, control is selected by simulation timestamp from the Behavior
    Tape. PDM runs in shadow mode from route start so its route planner/internal
    state is warm at handoff. Base collection is only active inside the manually
    chosen [record_start, record_end] window.
    """

    track = Track.MAP

    def __init__(self, carla_host, carla_port, debug=False):
        super().__init__(carla_host, carla_port, debug)
        self.progress = RouteProgress()
        self.expert: Optional[ExpertAdapter] = None
        self._collector_global_plan_gps = None
        self._collector_global_plan_world_coord = None

        self._recording_started = False
        self._recording_finished = False
        self._sync_streak = 0
        self._last_record_timestamp = None
        self._last_status_second = None
        self._handoff_done = False
        self._replacement_complete = False
        self._end_streak = 0
        self._fast_finish_applied = False
        self._replacement_mode = "pdm_expert"
        self.manual_follower: Optional[ManualTrajectoryFollower] = None

        self._divergence_checks = 0
        self._max_position_error = 0.0
        self._max_yaw_error = 0.0

        self._calls = 0
        self._first_game_time = None
        self._first_world_time = None
        self._last_game_time = None
        self._last_world_time = None
        self._first_control_frame = None
        self._first_state_frame = None

    def setup(self, path_to_conf_file):
        config_path, _, self.run_name = path_to_conf_file.partition("+")
        self.config = ReplayCollectorConfig.load(config_path)
        self.tape = BehaviorTape(self.config.replay.tape_dir)

        if not math.isfinite(float(self.tape.dt)) or float(self.tape.dt) <= 0.0:
            raise RuntimeError("invalid Behavior Tape dt: %r" % (self.tape.dt,))

        self._replay_control_shift_seconds = (
            REPLAY_CONTROL_SHIFT_TICKS * float(self.tape.dt)
        )
        self._shadow_enabled = False

        if self.config.replay.require_complete_tape:
            complete = self.tape.run_dir / "COMPLETE"
            if not complete.is_file():
                raise RuntimeError(
                    "refusing incomplete Behavior Tape (missing COMPLETE): %s"
                    % self.tape.run_dir
                )
            manifest_path = self.tape.run_dir / "manifest.json"
            if manifest_path.is_file():
                with manifest_path.open("r", encoding="utf-8") as handle:
                    probe_manifest = json.load(handle)
                result = probe_manifest.get("result") or {}
                if result and not bool(result.get("integrity_ok", False)):
                    raise RuntimeError(
                        "refusing Behavior Tape whose manifest integrity_ok is false: %s"
                        % manifest_path
                    )

        self.spec: InterventionSpec = load_intervention_spec(
            self.config.replay.intervention_spec,
            self.tape,
        )

        self._replacement_mode = normalize_replacement_mode(self.spec.replacement_mode)

        # GUI/runtime contract: when Recovery is launched from the Workbench,
        # the GUI exports the mode/plan it verified.  Refuse to continue if the
        # evaluator loads a different case, stale YAML, or stale agent.
        expected_mode = normalize_replacement_mode(os.environ.get("RECOVERY_EXPECT_MODE", ""))
        if os.environ.get("RECOVERY_EXPECT_MODE") and expected_mode != self._replacement_mode:
            raise RuntimeError(
                "GUI/runtime replacement mismatch: GUI expected %s, agent loaded %s"
                % (expected_mode, self._replacement_mode)
            )
        expected_plan = os.environ.get("RECOVERY_EXPECT_PLAN", "").strip()
        if expected_plan and self._replacement_mode == "manual_pid":
            loaded_plan = str(Path(self.spec.replacement_plan or "").expanduser().resolve())
            wanted_plan = str(Path(expected_plan).expanduser().resolve())
            if loaded_plan != wanted_plan:
                raise RuntimeError(
                    "GUI/runtime manual plan mismatch: GUI expected %s, agent loaded %s"
                    % (wanted_plan, loaded_plan)
                )

        # PDM is only instantiated for the PDM branch.  A manual_pid case must
        # never silently fall back to PDM if its plan is missing or malformed.
        if self._replacement_mode == "manual_pid":
            if not self.spec.replacement_plan:
                raise RuntimeError("manual_pid case is missing replacement.plan")
            self.manual_follower = ManualTrajectoryFollower(
                self.spec.replacement_plan,
                frequency_hz=self.config.frequency_hz,
            )

            # Fail closed if the dense path does not actually correspond to the
            # case the GUI selected.  Old Manual Planner V1 files could carry a
            # different Planning End; those must not be executed accidentally.
            if self.spec.handoff_x is not None and self.spec.handoff_y is not None:
                start_gap = math.hypot(
                    float(self.manual_follower.x[0]) - float(self.spec.handoff_x),
                    float(self.manual_follower.y[0]) - float(self.spec.handoff_y),
                )
                if start_gap > 2.5:
                    raise RuntimeError(
                        "manual trajectory start is %.3fm from case handoff; "
                        "re-save the Manual Plan for this case" % start_gap
                    )
            else:
                start_gap = None

            if self.spec.has_spatial_end:
                end_gap = math.hypot(
                    float(self.manual_follower.x[-1]) - float(self.spec.end_x),
                    float(self.manual_follower.y[-1]) - float(self.spec.end_y),
                )
                if end_gap > 2.5:
                    raise RuntimeError(
                        "manual trajectory end is %.3fm from case spatial End; "
                        "re-save the Manual Plan so End is shared" % end_gap
                    )
            else:
                end_gap = None

            self._manual_alignment = (start_gap, end_gap)
            self.expert = None
            self._shadow_enabled = False
        elif self._replacement_mode == "pdm_expert":
            self.expert = ExpertAdapter(
                self.config.expert.module,
                self.config.expert.entry_point,
                self.config.expert.config,
            )
            self._shadow_enabled = bool(self.config.replay.shadow_pdm)
            self._set_expert_shadow_mode(self._shadow_enabled)
            if self._collector_global_plan_gps is not None:
                self.expert.set_global_plan(
                    self._collector_global_plan_gps,
                    self._collector_global_plan_world_coord,
                )
        else:
            raise RuntimeError("unsupported replacement mode: %s" % self._replacement_mode)

        route = RouteCatalog(self.config.route_xml).get(self.config.route_id)
        self.writer = Bench2DriveWriter(
            self.config.output_root,
            route,
            self.config.weather_id,
            self.config.frequency_hz,
            self.config.jpeg_quality,
            self.config.sensor_profile,
        )
        self.rig = AuxiliarySensorRig(
            self.config.sensor_profile,
            self.config.frequency_hz,
        )
        self._sync_warmup_frames = max(
            1,
            int(round(self.config.frequency_hz * 0.5)),
        )
        self._write_recovery_metadata()

        print(
            "[Recovery] tape=%s samples=%d t=%.3f..%.3f dt=%.3f"
            % (
                self.tape.run_dir,
                len(self.tape.frames),
                self.tape.first_time,
                self.tape.last_time,
                self.tape.dt,
            ),
            flush=True,
        )
        if self.spec.has_spatial_end:
            end_desc = "spatial=(%.3f, %.3f) r=%.2fm confirm=%d" % (
                self.spec.end_x, self.spec.end_y, self.spec.end_radius_m,
                self.spec.end_confirm_frames,
            )
        else:
            end_desc = "time=%.3f" % float(self.spec.record_end_time)
        print(
            "[Recovery] record_start=%.3f handoff=%.3f end=%s mode=%s "
            "case_offset=%+.3f causal_shift=%+d tick (%.3fs) shadow=%s"
            % (
                self.spec.record_start_time,
                self.spec.handoff_time,
                end_desc,
                self._replacement_mode,
                self.spec.time_offset_seconds,
                REPLAY_CONTROL_SHIFT_TICKS,
                self._replay_control_shift_seconds,
                self._shadow_enabled,
            ),
            flush=True,
        )
        if self._replacement_mode == "manual_pid":
            assert self.manual_follower is not None
            print("[Recovery][MANUAL] plan=%s" % self.manual_follower.plan_path, flush=True)
            if getattr(self, "_manual_alignment", None) is not None:
                sg, eg = self._manual_alignment
                print(
                    "[Recovery][MANUAL] alignment start_gap=%s end_gap=%s"
                    % (
                        ("%.3fm" % sg) if sg is not None else "n/a",
                        ("%.3fm" % eg) if eg is not None else "n/a",
                    ),
                    flush=True,
                )
            print(
                "[Recovery][MANUAL] trajectory=%s samples=%d length=%.2fm speed=%.2f..%.2fm/s"
                % (
                    self.manual_follower.trajectory_file or "embedded/generated-from-plan",
                    len(self.manual_follower.x),
                    self.manual_follower.path_length_m,
                    float(self.manual_follower.target_speed.min()),
                    float(self.manual_follower.target_speed.max()),
                ),
                flush=True,
            )
        print(
            "[Recovery] replay mapping: control_target_t = "
            "CARLA_world_t + case_offset + 1*tape_dt",
            flush=True,
        )

    def _set_expert_shadow_mode(self, enabled: bool) -> None:
        if self.expert is None:
            if enabled:
                raise RuntimeError("shadow PDM requested but PDM expert is not active")
            return
        setter = getattr(self.expert, "set_shadow_mode", None)
        if setter is None:
            # Compatible with the older ExpertAdapter: PdmLiteExpert itself
            # exposes set_shadow_mode().
            inner = getattr(self.expert, "agent", None)
            setter = getattr(inner, "set_shadow_mode", None)
        if setter is None:
            if enabled:
                raise RuntimeError(
                    "configured expert does not support shadow mode: %s"
                    % self.config.expert.module
                )
            return
        setter(bool(enabled))

    def _write_recovery_metadata(self) -> None:
        payload = {
            "schema": "b2d-recovery-run-v2",
            "source_tape": str(self.tape.run_dir),
            "intervention_spec": str(Path(self.config.replay.intervention_spec).resolve()),
            "case_id": self.spec.case_id,
            "intervention_type": self.spec.intervention_type,
            "description": self.spec.description,
            "record_start_time": self.spec.record_start_time,
            "handoff_time": self.spec.handoff_time,
            "record_end_time": self.spec.record_end_time,
            "spatial_end": (
                {
                    "x": self.spec.end_x,
                    "y": self.spec.end_y,
                    "radius_m": self.spec.end_radius_m,
                    "confirm_frames": self.spec.end_confirm_frames,
                }
                if self.spec.has_spatial_end else None
            ),
            "replacement_mode": self._replacement_mode,
            "replacement_plan": self.spec.replacement_plan,
            "time_offset_seconds": self.spec.time_offset_seconds,
            "shadow_pdm": self._shadow_enabled,
            "window_clock": "carla_world_elapsed_seconds",
            "expert_clock": "leaderboard_run_step_timestamp",
            "replay_control_shift_ticks": REPLAY_CONTROL_SHIFT_TICKS,
            "replay_control_shift_seconds": self._replay_control_shift_seconds,
            "replay_contract": (
                "control_target_t = carla_world_t + time_offset_seconds + 1*tape_dt"
            ),
            "state_contract": (
                "state_reference_t = carla_world_t + time_offset_seconds"
            ),
        }
        path = self.writer.meta_path / "recovery.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def sensors(self):
        base = collection_sensors(self.config.sensor_profile)
        if self.expert is None:
            return base
        return merge_sensor_specs(base, self.expert.sensors())

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        self._collector_global_plan_gps = global_plan_gps
        self._collector_global_plan_world_coord = global_plan_world_coord
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self.progress.set_plan(global_plan_world_coord)
        if self.expert is not None:
            self.expert.set_global_plan(global_plan_gps, global_plan_world_coord)

    def _hero(self):
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        getter = getattr(CarlaDataProvider, "get_hero_actor", None)
        if getter is not None:
            hero = getter()
            if hero is not None:
                return hero
        return getattr(CarlaDataProvider, "_ego_actor", None)

    def _world_time(self) -> float:
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        world = CarlaDataProvider.get_world()
        if world is None:
            hero = self._hero()
            world = None if hero is None else hero.get_world()
        if world is None:
            raise RuntimeError("CARLA world is not available to recovery agent")
        return float(world.get_snapshot().timestamp.elapsed_seconds)

    def _frame_for_state_time(self, world_time: float) -> Optional[TapeFrame]:
        # Current physical state is compared with the tape at the same world time.
        # The +1 control-causality shift must NOT be applied here.
        return self.tape.frame_for_replay_time(
            world_time,
            self.spec.time_offset_seconds,
        )

    def _frame_for_control_time(self, world_time: float) -> Optional[TapeFrame]:
        # The control returned now is applied by CARLA on the next physics tick.
        return self.tape.frame_for_replay_time(
            world_time,
            self.spec.time_offset_seconds + self._replay_control_shift_seconds,
        )

    def _check_divergence(
        self,
        world_time: float,
        state_frame: Optional[TapeFrame],
    ) -> None:
        if state_frame is None:
            return
        # Check through the exact handoff state. PDM's first returned control only
        # affects the following physics tick.
        if world_time > self.spec.handoff_time + _EPS:
            return
        if world_time < self.tape.first_time + self.spec.divergence_grace_seconds:
            return

        hero = self._hero()
        if hero is None:
            return
        transform = hero.get_transform()
        dx = float(transform.location.x) - state_frame.ego_location[0]
        dy = float(transform.location.y) - state_frame.ego_location[1]
        dz = float(transform.location.z) - state_frame.ego_location[2]
        position_error = math.sqrt(dx * dx + dy * dy + dz * dz)
        yaw_error = _yaw_error_deg(transform.rotation.yaw, state_frame.ego_rotation[2])
        self._divergence_checks += 1
        self._max_position_error = max(self._max_position_error, position_error)
        self._max_yaw_error = max(self._max_yaw_error, yaw_error)

        if (
            position_error > self.spec.max_position_error_m
            or yaw_error > self.spec.max_yaw_error_deg
        ):
            message = (
                "REPLAY_DIVERGED at world_t=%.3f ref_state=%.3f: "
                "position=%.3fm (limit %.3f), yaw=%.3fdeg (limit %.3f)"
                % (
                    world_time,
                    state_frame.sim_time,
                    position_error,
                    self.spec.max_position_error_m,
                    yaw_error,
                    self.spec.max_yaw_error_deg,
                )
            )
            if self.spec.abort_on_divergence:
                raise RuntimeError(message)
            print("[Recovery][WARN] " + message, flush=True)

    def _update_status(
        self,
        world_time: float,
        game_time: float,
        source: str,
        state_frame: Optional[TapeFrame],
        control_frame: Optional[TapeFrame],
    ) -> None:
        second = int(world_time)
        if second == self._last_status_second:
            return
        self._last_status_second = second
        state_ref = (
            "none"
            if state_frame is None
            else "%.3f#%d" % (state_frame.sim_time, state_frame.sample_index)
        )
        control_ref = (
            "none"
            if control_frame is None
            else "%.3f#%d" % (control_frame.sim_time, control_frame.sample_index)
        )
        print(
            "[Recovery] world_t=%.3f game_t=%.3f source=%s "
            "state_ref=%s control_ref=%s rec=%s"
            % (
                world_time,
                game_time,
                source,
                state_ref,
                control_ref,
                self._recording_started and not self._recording_finished,
            ),
            flush=True,
        )

    def _distance_to_spatial_end(self) -> Optional[float]:
        if not self.spec.has_spatial_end:
            return None
        hero = self._hero()
        if hero is None:
            return None
        loc = hero.get_transform().location
        dx = float(loc.x) - float(self.spec.end_x)
        dy = float(loc.y) - float(self.spec.end_y)
        return math.sqrt(dx * dx + dy * dy)

    def _update_spatial_end(self, world_time: float) -> None:
        if self._replacement_complete or not self.spec.has_spatial_end:
            return
        if world_time < self.spec.handoff_time - _EPS:
            self._end_streak = 0
            return

        # Guard against an accidental early pass near End on a looped/manual path.
        progress_ok = True
        if self.manual_follower is not None and self.manual_follower.path_length_m > 1e-3:
            progress_ok = (
                self.manual_follower.progress_s
                >= 0.75 * self.manual_follower.path_length_m
            )

        distance = self._distance_to_spatial_end()
        if distance is not None and progress_ok and distance <= float(self.spec.end_radius_m):
            self._end_streak += 1
        else:
            self._end_streak = 0

        if self._end_streak >= int(self.spec.end_confirm_frames):
            self._replacement_complete = True
            self._recording_finished = True
            print(
                "[Recovery] SPATIAL END reached world_t=%.3f distance=%.3fm "
                "confirm=%d frames=%d"
                % (world_time, float(distance), self._end_streak, self.writer.frame),
                flush=True,
            )

    def _fast_finish_route(self) -> None:
        """Move ego to route final transform *after* the final recorded frame.

        Bench2Drive has no normal agent 'done' return value.  Without this, a
        spatial-end correction would sit/brake until route timeout.  Teleporting
        only after Collector is closed lets leaderboard finish the route promptly
        without contaminating the correction dataset.
        """
        if self._fast_finish_applied or not self._replacement_complete:
            return
        self._fast_finish_applied = True
        hero = self._hero()
        plan = self._collector_global_plan_world_coord
        if hero is None or not plan:
            print("[Recovery][WARN] replacement complete but fast-finish target unavailable", flush=True)
            return
        try:
            item = plan[-1]
            target = item[0] if isinstance(item, (tuple, list)) else item
            transform = target if isinstance(target, carla.Transform) else None
            if transform is None and hasattr(target, "location") and hasattr(target, "rotation"):
                transform = target
            if transform is None:
                raise RuntimeError("unsupported final global-plan transform: %r" % (target,))
            finish = carla.Transform(
                carla.Location(
                    x=float(transform.location.x),
                    y=float(transform.location.y),
                    z=float(transform.location.z) + 0.25,
                ),
                carla.Rotation(
                    pitch=float(transform.rotation.pitch),
                    yaw=float(transform.rotation.yaw),
                    roll=float(transform.rotation.roll),
                ),
            )
            hero.set_transform(finish)
            hero.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            hero.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            print("[Recovery] replacement COMPLETE -> fast-finish route", flush=True)
        except Exception as exc:
            print("[Recovery][WARN] fast-finish route failed: %s" % exc, flush=True)

    def _record_if_needed(
        self,
        input_data,
        world_time: float,
        control: carla.VehicleControl,
        source: str,
        auxiliary_ready: bool,
    ) -> None:
        # V2: record starts on the original E2E timeline and ends only when the
        # vehicle reaches the free spatial End.  Legacy v1 keeps its time end.
        if self._recording_finished:
            return
        if world_time < self.spec.record_start_time - _EPS:
            return
        if (
            self.spec.record_end_time is not None
            and world_time > self.spec.record_end_time + _EPS
        ):
            if self._recording_started:
                self._recording_finished = True
                print(
                    "[Recovery] Base recording finished before world_t=%.3f frames=%d"
                    % (world_time, self.writer.frame),
                    flush=True,
                )
            return

        if not auxiliary_ready:
            if self._recording_started:
                raise RuntimeError(
                    "auxiliary sensor synchronization lost during recovery recording"
                )
            self._sync_streak = 0
            raise RuntimeError(
                "record_start reached before auxiliary sensors became synchronized; "
                "increase replay.sensor_warmup_seconds"
            )

        self._sync_streak += 1
        if self._sync_streak < self._sync_warmup_frames:
            raise RuntimeError(
                "record_start reached before synchronization warmup completed; "
                "increase replay.sensor_warmup_seconds"
            )

        if not self._recording_started:
            self._recording_started = True
            print(
                "[Recovery] Base recording starts at world_t=%.3f" % world_time,
                flush=True,
            )

        expected_dt = 1.0 / float(self.config.frequency_hz)
        if self._last_record_timestamp is not None:
            dt = world_time - self._last_record_timestamp
            if abs(dt - expected_dt) > 0.001:
                raise RuntimeError(
                    "dataset timestamp discontinuity: expected %.6fs got %.6fs"
                    % (expected_dt, dt)
                )

        nav = self.progress.update(self.writer.annotator.ego_location())
        assessment = self.expert.assessment() if self.expert is not None else None
        failure_active = self.spec.intervention_type.lower() == "failure"
        if self.writer.record(
            input_data,
            control,
            world_time,
            nav,
            source,
            failure_active,
            assessment,
        ):
            self._last_record_timestamp = world_time

        # Legacy time-window case: include the end sample itself.
        if (
            self.spec.record_end_time is not None
            and world_time >= self.spec.record_end_time - _EPS
        ):
            self._recording_finished = True
            print(
                "[Recovery] Base recording finished at world_t=%.3f frames=%d"
                % (world_time, self.writer.frame),
                flush=True,
            )

    def run_step(self, input_data, timestamp):
        game_time = float(timestamp)
        world_time = self._world_time()
        previous_world_time = self._last_world_time

        self._calls += 1
        self._last_game_time = game_time
        self._last_world_time = world_time

        if self._first_game_time is None:
            self._first_game_time = game_time
            self._first_world_time = world_time
            print(
                "[Recovery] FIRST tick game_t=%.3f world_t=%.3f delta=%.3f"
                % (game_time, world_time, world_time - game_time),
                flush=True,
            )
        elif previous_world_time is not None and world_time + _EPS < previous_world_time:
            raise RuntimeError(
                "CARLA world time moved backwards: %.6f -> %.6f"
                % (previous_world_time, world_time)
            )

        # Two references are intentionally different:
        #   state_frame   = current world-time state (for divergence)
        #   control_frame = +1 tick control (for CARLA next-tick causality)
        state_frame = self._frame_for_state_time(world_time)
        control_frame = self._frame_for_control_time(world_time)
        control_target_time = (
            world_time
            + self.spec.time_offset_seconds
            + self._replay_control_shift_seconds
        )

        if self._first_state_frame is None and state_frame is not None:
            self._first_state_frame = (
                world_time,
                state_frame.sample_index,
                state_frame.sim_time,
            )
        if self._first_control_frame is None and control_frame is not None:
            self._first_control_frame = (
                world_time,
                control_target_time,
                control_frame.sample_index,
                control_frame.sim_time,
            )
            print(
                "[Recovery] FIRST replay selection world_t=%.3f target_t=%.3f "
                "idx=%d tape_t=%.3f"
                % self._first_control_frame,
                flush=True,
            )

        if (
            self.config.replay.abort_after_tape
            and world_time < self.spec.handoff_time - _EPS
            and control_target_time > self.tape.last_time + self.tape.dt * 0.51
        ):
            raise RuntimeError(
                "replay tape ended before handoff: control target %.3f > tape %.3f"
                % (control_target_time, self.tape.last_time)
            )

        # Validate the replayed physical state through the exact handoff tick.
        if world_time <= self.spec.handoff_time + _EPS:
            self._check_divergence(world_time, state_frame)

        if self._replacement_complete:
            self._fast_finish_route()
            proposed = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
            source = self._replacement_mode
        elif world_time < self.spec.handoff_time - _EPS:
            # PDM observes every replay tick only for a PDM recovery. Manual mode
            # is deliberately isolated: replay is the only controller pre-handoff.
            if self._shadow_enabled and self.expert is not None:
                self.expert.run_step(input_data, game_time)
            proposed = _control_from_tape(control_frame)
            source = "e2e_replay"
        else:
            if not self._handoff_done:
                if self._shadow_enabled:
                    self._set_expert_shadow_mode(False)
                self._handoff_done = True
                if self.manual_follower is not None:
                    self.manual_follower.reset(world_time)
                print(
                    "[Recovery] HANDOFF E2E -> %s at world_t=%.3f game_t=%.3f"
                    % (self._replacement_mode.upper(), world_time, game_time),
                    flush=True,
                )

            if self._replacement_mode == "manual_pid":
                assert self.manual_follower is not None
                proposed = self.manual_follower.run_step(self._hero(), world_time)
                source = "manual_pid"
            else:
                if self.expert is None:
                    raise RuntimeError("PDM replacement selected but expert is not initialized")
                proposed = self.expert.run_step(input_data, game_time)
                source = "pdm_expert"

        control, clipped, raw = _finalize_vehicle_control(proposed)
        if clipped:
            print(
                "[Recovery][WARN] control clamped throttle=%.6f steer=%.6f brake=%.6f"
                % (raw["throttle"], raw["steer"], raw["brake"]),
                flush=True,
            )

        warmup_start = self.spec.record_start_time - float(
            self.config.replay.sensor_warmup_seconds
        )
        auxiliary_ready = False
        if world_time >= warmup_start - _EPS:
            auxiliary_ready = self.rig.enrich(input_data)
            if auxiliary_ready and world_time < self.spec.record_start_time - _EPS:
                self._sync_streak += 1
            elif not auxiliary_ready and world_time < self.spec.record_start_time - _EPS:
                self._sync_streak = 0

        self._record_if_needed(
            input_data,
            world_time,
            control,
            source,
            auxiliary_ready,
        )

        # End detection happens after writer.record(), so the confirming final
        # frame is included in the dataset.
        if self.spec.has_spatial_end and not self._recording_finished:
            self._update_spatial_end(world_time)

        if source == "manual_pid" and self.manual_follower is not None:
            second = int(world_time)
            if second != getattr(self, "_last_manual_status_second", None):
                self._last_manual_status_second = second
                st = self.manual_follower.last_status
                if st:
                    print(
                        "[Recovery][MANUAL] s=%.2f/%.2fm cte=%+.3fm heading=%+.2fdeg "
                        "v=%.2f/%.2fmps speed_ref_s=%.2fm steer=%+.3f throttle=%.3f brake=%.3f end_dist=%s"
                        % (
                            st["progress_s"], st["path_length_m"],
                            st["cross_track_error_m"], st["heading_error_deg"],
                            st["speed_mps"], st["target_speed_mps"],
                            st.get("speed_preview_s", st["progress_s"]),
                            st["steer"], st["throttle"], st["brake"],
                            ("%.2fm" % self._distance_to_spatial_end())
                            if self._distance_to_spatial_end() is not None else "n/a",
                        ),
                        flush=True,
                    )

        self._update_status(
            world_time,
            game_time,
            source,
            state_frame,
            control_frame if source == "e2e_replay" else None,
        )
        return control

    def destroy(self):
        print(
            "[Recovery] summary calls=%d handoff=%s mode=%s complete=%s recorded_frames=%s "
            "recording_finished=%s divergence_checks=%d "
            "max_position_error=%.3fm max_yaw_error=%.3fdeg "
            "first_game_t=%r first_world_t=%r last_game_t=%r last_world_t=%r "
            "first_state_ref=%r first_control_ref=%r"
            % (
                self._calls,
                self._handoff_done,
                self._replacement_mode,
                self._replacement_complete,
                getattr(getattr(self, "writer", None), "frame", 0),
                self._recording_finished,
                self._divergence_checks,
                self._max_position_error,
                self._max_yaw_error,
                self._first_game_time,
                self._first_world_time,
                self._last_game_time,
                self._last_world_time,
                self._first_state_frame,
                self._first_control_frame,
            ),
            flush=True,
        )
        if getattr(self, "rig", None) is not None:
            self.rig.destroy()
        if getattr(self, "expert", None) is not None:
            self.expert.destroy()

