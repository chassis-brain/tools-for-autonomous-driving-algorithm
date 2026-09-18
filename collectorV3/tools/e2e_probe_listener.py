#!/usr/bin/env python3
"""Passive, model-agnostic E2E Behavior Tape recorder for CARLA/Bench2Drive.

Run this beside the normal E2E evaluator. It never calls world.tick() and never
changes vehicle control. The tape records the *actually applied* VehicleControl
from CARLA plus ego/world state for later timestamp-indexed replay.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shutil
import signal
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Set

import carla


def _vec(v) -> Dict[str, float]:
    return {"x": float(v.x), "y": float(v.y), "z": float(v.z)}


def _rot(r) -> Dict[str, float]:
    return {"pitch": float(r.pitch), "yaw": float(r.yaw), "roll": float(r.roll)}


def _transform(t) -> Dict[str, Any]:
    return {"location": _vec(t.location), "rotation": _rot(t.rotation)}


def _control(c) -> Dict[str, Any]:
    return {
        "throttle": float(c.throttle),
        "steer": float(c.steer),
        "brake": float(c.brake),
        "hand_brake": bool(c.hand_brake),
        "reverse": bool(c.reverse),
        "manual_gear_shift": bool(c.manual_gear_shift),
        "gear": int(c.gear),
    }


def _bbox(actor) -> Dict[str, Any]:
    box = actor.bounding_box
    return {
        "location": _vec(box.location),
        "extent": _vec(box.extent),
        "rotation": _rot(box.rotation),
    }


def _snapshot_state(item) -> Dict[str, Any]:
    transform = item.get_transform()
    return {
        "actor_id": int(item.id),
        "transform": _transform(transform),
        "velocity": _vec(item.get_velocity()),
        "acceleration": _vec(item.get_acceleration()),
        "angular_velocity": _vec(item.get_angular_velocity()),
    }


class ProbeListener:
    def __init__(self, args):
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(args.rpc_timeout)
        self.world = None
        self.world_key = None
        self.ego = None
        self.ego_id = None
        self.callback_id = None
        self.stop_event = threading.Event()
        self.rows = queue.Queue()
        self.writer_thread = None
        self.frames_handle = None
        self.actors_handle = None
        self.integrity_handle = None
        self.run_dir = None
        self.native_recorder_started = False

        self.lock = threading.Lock()
        self.capture_count = 0
        self.frames_written = 0
        self.tick_callbacks_received = 0
        self.last_callback_frame = None
        self.frame_gap_count = 0
        self.acquisition_edge_gap_count = 0
        self.control_capture_errors = 0
        self.tick_buffer_max_depth = 0
        self.first_world_frame = None
        self.last_world_frame = None
        self.first_elapsed_seconds = None
        self.last_elapsed_seconds = None
        self.seen_actor_ids: Set[int] = set()
        self.metadata_written: Set[int] = set()
        self.started_wall_time = time.time()
        self.finish_reason = "unknown"
        self._signal_handlers_installed = False

    def _handle_signal(self, signum, _frame):
        if self.finish_reason == "unknown":
            try:
                name = signal.Signals(signum).name
            except Exception:
                name = str(signum)
            self.finish_reason = "signal_%s" % name
        print("[probe] stop requested: %s" % self.finish_reason, flush=True)
        self.stop_event.set()

    def _install_signal_handlers(self):
        if self._signal_handlers_installed:
            return
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._handle_signal)
            except Exception:
                pass
        self._signal_handlers_installed = True

    def _update_manifest_runtime(self):
        if self.run_dir is None:
            return
        path = self.run_dir / "manifest.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            manifest = {"schema": "e2e_probe_manifest_v2"}
        manifest["world"] = {
            "id": self.world_key[0] if self.world_key else None,
            "map": self.world_key[1] if self.world_key else None,
        }
        manifest["ego_actor_id"] = self.ego_id
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

    def _safe_world_key(self, world) -> Optional[tuple]:
        try:
            world_id = getattr(world, "id", None)
            if world_id is None:
                world_id = hash(world)
            return (int(world_id), str(world.get_map().name))
        except Exception:
            return (id(world), "<map-unavailable>")

    def refresh_world(self) -> None:
        world = self.client.get_world()
        key = self._safe_world_key(world)
        if self.world is None or key != self.world_key:
            self.world = world
            self.world_key = key
            self.ego = None
            self.ego_id = None
            print("[probe] world -> %r" % (key,), flush=True)

    def find_ego(self):
        self.refresh_world()
        if self.world is None:
            return None
        try:
            actors = self.world.get_actors().filter("vehicle.*")
        except RuntimeError as exc:
            print("[probe] get_actors timeout while waiting for ego: %s" % exc, flush=True)
            return None
        for actor in actors:
            role = actor.attributes.get("role_name", "")
            if role == self.args.ego_role or (
                self.args.ego_role == "hero" and role == "ego_vehicle"
            ):
                return actor
        return None

    def wait_for_ego(self):
        last_print = 0.0
        while not self.stop_event.is_set():
            try:
                ego = self.find_ego()
            except RuntimeError as exc:
                print("[probe] transient CARLA RPC timeout: %s" % exc, flush=True)
                time.sleep(self.args.poll_interval)
                continue
            if ego is not None:
                self.ego = ego
                self.ego_id = int(ego.id)
                print(
                    "[probe] ego found id=%d type=%s role=%s"
                    % (self.ego_id, ego.type_id, ego.attributes.get("role_name", "")),
                    flush=True,
                )
                return
            now = time.time()
            if now - last_print >= 2.0:
                print(
                    "[probe] waiting for ego role=%r (passive; RPC timeout is non-fatal)"
                    % self.args.ego_role,
                    flush=True,
                )
                last_print = now
            time.sleep(self.args.poll_interval)
        raise KeyboardInterrupt

    def prepare_output(self):
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        run_id = "probe_%s_%s" % (stamp, uuid.uuid4().hex[:8])
        self.run_dir = Path(self.args.output_root).expanduser().resolve() / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.frames_handle = (self.run_dir / "frames.jsonl").open("w", encoding="utf-8", buffering=1)
        self.actors_handle = (self.run_dir / "actors.jsonl").open("w", encoding="utf-8", buffering=1)
        self.integrity_handle = (self.run_dir / "integrity.jsonl").open("w", encoding="utf-8", buffering=1)

        if self.args.route_xml:
            src = Path(self.args.route_xml).expanduser().resolve()
            if src.is_file():
                shutil.copy2(str(src), str(self.run_dir / "route.xml"))

        manifest = {
            "schema": "e2e_probe_manifest_v2",
            "run_id": run_id,
            "created_unix": time.time(),
            "host": self.args.host,
            "port": self.args.port,
            "ego_role": self.args.ego_role,
            "route_xml": self.args.route_xml or None,
            "route_id": self.args.route_id or None,
            "seed": self.args.seed,
            "traffic_manager_seed": self.args.tm_seed,
            "actor_radius_m": self.args.actor_radius,
            "world": {
                "id": self.world_key[0] if self.world_key else None,
                "map": self.world_key[1] if self.world_key else None,
            },
            "result": None,
        }
        with (self.run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

        # Reserve the output directory before the E2E ego exists.  This gives the
        # GUI a concrete artifact immediately and guarantees that even a failed
        #/zero-sample run leaves a manifest explaining what happened.
        print("[probe] output reserved=%s" % self.run_dir, flush=True)
        print("[probe] READY waiting-for-hero", flush=True)

        if self.args.native_recorder:
            try:
                self.client.start_recorder(str(self.run_dir / "carla_native_recorder.log"), True)
                self.native_recorder_started = True
                print("[probe] native CARLA recorder started", flush=True)
            except Exception as exc:
                self._integrity("native_recorder_error", {"error": repr(exc)})

    def _integrity(self, event: str, payload: Dict[str, Any]) -> None:
        if self.integrity_handle is None:
            return
        row = {"event": event, "wall_time": time.time()}
        row.update(payload)
        self.integrity_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _capture(self, snapshot):
        try:
            if self.ego is None or self.ego_id is None:
                return
            ego_snap = snapshot.find(self.ego_id)
            if ego_snap is None:
                return

            timestamp = snapshot.timestamp
            frame = int(snapshot.frame)
            elapsed = float(timestamp.elapsed_seconds)
            delta = float(timestamp.delta_seconds)

            try:
                applied_control = _control(self.ego.get_control())
            except Exception as exc:
                with self.lock:
                    self.control_capture_errors += 1
                self._integrity(
                    "control_capture_error",
                    {"world_frame": frame, "error": repr(exc)},
                )
                return

            ego_state = _snapshot_state(ego_snap)
            velocity = ego_snap.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            ego_loc = ego_snap.get_transform().location

            actor_states = []
            seen = set()
            radius2 = float(self.args.actor_radius) ** 2
            if self.args.actor_radius > 0:
                for item in snapshot:
                    actor_id = int(item.id)
                    if actor_id == self.ego_id:
                        continue
                    transform = item.get_transform()
                    dx = float(transform.location.x - ego_loc.x)
                    dy = float(transform.location.y - ego_loc.y)
                    dz = float(transform.location.z - ego_loc.z)
                    if dx * dx + dy * dy + dz * dz > radius2:
                        continue
                    actor_states.append(_snapshot_state(item))
                    seen.add(actor_id)

            with self.lock:
                self.tick_callbacks_received += 1
                if self.last_callback_frame is not None and frame != self.last_callback_frame + 1:
                    gap = max(0, frame - self.last_callback_frame - 1)
                    self.acquisition_edge_gap_count += gap
                    self.frame_gap_count += gap
                    self._integrity(
                        "frame_gap",
                        {"previous": self.last_callback_frame, "current": frame, "gap": gap},
                    )
                self.last_callback_frame = frame
                sample_index = self.capture_count
                self.capture_count += 1
                self.seen_actor_ids.update(seen)
                depth = self.rows.qsize() + 1
                self.tick_buffer_max_depth = max(self.tick_buffer_max_depth, depth)
                if self.first_world_frame is None:
                    self.first_world_frame = frame
                    self.first_elapsed_seconds = elapsed
                self.last_world_frame = frame
                self.last_elapsed_seconds = elapsed

            row = {
                "schema": "e2e_probe_frame_v2",
                "sample_index": int(sample_index),
                "world": {
                    "frame": frame,
                    "elapsed_seconds": elapsed,
                    "delta_seconds": delta,
                },
                "ego": {
                    "actor_id": self.ego_id,
                    "state": {
                        "transform": ego_state["transform"],
                        "velocity": ego_state["velocity"],
                        "acceleration": ego_state["acceleration"],
                        "angular_velocity": ego_state["angular_velocity"],
                    },
                    "speed": float(speed),
                    "applied_control": applied_control,
                },
                "scene": {"actors": actor_states},
            }
            self.rows.put_nowait(row)
        except Exception as exc:
            self._integrity("tick_callback_error", {"error": repr(exc)})

    def _writer_loop(self):
        while not self.stop_event.is_set() or not self.rows.empty():
            try:
                row = self.rows.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.frames_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                with self.lock:
                    self.frames_written += 1
            finally:
                self.rows.task_done()

    def _write_actor_metadata(self):
        if self.world is None:
            return
        try:
            actors = list(self.world.get_actors())
        except RuntimeError:
            return

        with self.lock:
            wanted = set(self.seen_actor_ids)
        if self.ego_id is not None:
            wanted.add(self.ego_id)

        for actor in actors:
            actor_id = int(actor.id)
            if actor_id not in wanted or actor_id in self.metadata_written:
                continue
            try:
                row = {
                    "schema": "e2e_probe_actor_v2",
                    "actor_id": actor_id,
                    "type_id": actor.type_id,
                    "role_name": actor.attributes.get("role_name", ""),
                    "attributes": dict(actor.attributes),
                    "bounding_box": _bbox(actor),
                    "parent_id": int(actor.parent.id) if actor.parent is not None else None,
                    "semantic_tags": [int(v) for v in getattr(actor, "semantic_tags", [])],
                }
                self.actors_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                self.metadata_written.add(actor_id)
            except Exception as exc:
                self._integrity(
                    "actor_metadata_error",
                    {"actor_id": actor_id, "error": repr(exc)},
                )

    def _detach_tick_callback(self):
        if self.callback_id is not None and self.world is not None:
            try:
                self.world.remove_on_tick(self.callback_id)
            except Exception:
                pass
        self.callback_id = None

    def _attach_tick_callback(self):
        if self.world is None:
            raise RuntimeError("cannot arm probe without a CARLA world")
        self.callback_id = self.world.on_tick(self._capture)
        print("[probe] tick callback armed world=%r" % (self.world_key,), flush=True)

    def arm(self):
        if self.writer_thread is None:
            self.writer_thread = threading.Thread(target=self._writer_loop, name="probe-writer")
            self.writer_thread.daemon = True
            self.writer_thread.start()
        self._attach_tick_callback()

    def monitor(self):
        missing_since = None
        last_meta = 0.0
        while not self.stop_event.is_set():
            now = time.time()

            # Bench2Drive may reload the CARLA world while the listener is already
            # alive.  Rebind the passive on_tick callback to the new world instead
            # of silently recording the stale world.
            try:
                current_world = self.client.get_world()
                current_key = self._safe_world_key(current_world)
            except Exception:
                current_world = None
                current_key = self.world_key
            if current_world is not None and current_key != self.world_key:
                print(
                    "[probe] world reload %r -> %r; rebinding"
                    % (self.world_key, current_key),
                    flush=True,
                )
                self._detach_tick_callback()
                self.world = current_world
                self.world_key = current_key
                self.ego = None
                self.ego_id = None
                self.wait_for_ego()
                self._update_manifest_runtime()
                self._attach_tick_callback()
                missing_since = None
                continue

            if now - last_meta >= self.args.metadata_refresh_seconds:
                self._write_actor_metadata()
                last_meta = now
            alive = True
            try:
                alive = bool(self.ego is not None and self.ego.is_alive)
            except Exception:
                alive = False
            if not alive:
                if missing_since is None:
                    missing_since = now
                elif now - missing_since >= self.args.ego_gone_grace_seconds:
                    if self.capture_count <= 0:
                        # If an old/stale ego disappeared before the first sample,
                        # keep waiting for the evaluator's real ego instead of
                        # producing an empty run and exiting too early.
                        print("[probe] ego disappeared before first sample; reacquiring", flush=True)
                        self.ego = None
                        self.ego_id = None
                        self.wait_for_ego()
                        self._update_manifest_runtime()
                        missing_since = None
                        continue
                    self.finish_reason = "ego_gone"
                    return
            else:
                missing_since = None
            time.sleep(self.args.poll_interval)

    def finalize(self):
        self.stop_event.set()
        self._detach_tick_callback()
        if self.writer_thread is not None:
            self.writer_thread.join(timeout=10.0)
        try:
            self._write_actor_metadata()
        except Exception:
            pass
        if self.native_recorder_started:
            try:
                self.client.stop_recorder()
            except Exception:
                pass

        for handle in (self.frames_handle, self.actors_handle, self.integrity_handle):
            if handle is not None:
                try:
                    handle.flush()
                    handle.close()
                except Exception:
                    pass

        if self.run_dir is None:
            return
        integrity_ok = (
            self.capture_count > 0
            and self.frames_written == self.capture_count
            and self.frame_gap_count == 0
            and self.acquisition_edge_gap_count == 0
            and self.control_capture_errors == 0
        )
        manifest_path = self.run_dir / "manifest.json"
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            manifest = {"schema": "e2e_probe_manifest_v2"}
        manifest["result"] = {
            "reason": self.finish_reason,
            "samples": self.capture_count,
            "frames_written": self.frames_written,
            "first_world_frame": self.first_world_frame,
            "last_world_frame": self.last_world_frame,
            "first_elapsed_seconds": self.first_elapsed_seconds,
            "last_elapsed_seconds": self.last_elapsed_seconds,
            "frame_gap_count": self.frame_gap_count,
            "acquisition_edge_gap_count": self.acquisition_edge_gap_count,
            "tick_callbacks_received": self.tick_callbacks_received,
            "tick_buffer_max_depth": self.tick_buffer_max_depth,
            "control_capture_errors": self.control_capture_errors,
            "integrity_ok": integrity_ok,
            "finished_unix": time.time(),
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        if integrity_ok:
            (self.run_dir / "COMPLETE").write_text("ok\n", encoding="utf-8")
        print(
            "[probe] samples=%d gaps=%d integrity_ok=%s reason=%s"
            % (self.capture_count, self.frame_gap_count, integrity_ok, self.finish_reason),
            flush=True,
        )
        print("[probe] output=%s" % self.run_dir, flush=True)

    def run(self):
        self._install_signal_handlers()
        try:
            # Create the artifact before the hero appears.  The GUI uses this as
            # the readiness handshake before launching the selected E2E route.
            self.refresh_world()
            self.prepare_output()
            self.wait_for_ego()
            self._update_manifest_runtime()
            self._write_actor_metadata()
            self.arm()
            self.monitor()
        except KeyboardInterrupt:
            if self.finish_reason == "unknown":
                self.finish_reason = "keyboard_interrupt"
        except Exception as exc:
            self.finish_reason = "exception"
            self._integrity("fatal_error", {"error": repr(exc)})
            raise
        finally:
            self.finalize()


def build_parser():
    parser = argparse.ArgumentParser(description="Passive E2E Behavior Tape recorder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--output-root", default="./probe_runs")
    parser.add_argument("--ego-role", default="hero")
    parser.add_argument("--route-xml", default="")
    parser.add_argument("--route-id", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tm-seed", type=int, default=0)
    parser.add_argument("--actor-radius", type=float, default=80.0)
    parser.add_argument("--rpc-timeout", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--metadata-refresh-seconds", type=float, default=1.0)
    parser.add_argument("--ego-gone-grace-seconds", type=float, default=2.0)
    parser.add_argument("--native-recorder", action="store_true")
    parser.add_argument(
        "--no-native-recorder",
        action="store_false",
        dest="native_recorder",
        help="Compatibility flag; native recorder is already disabled by default.",
    )
    parser.set_defaults(native_recorder=False)
    return parser


def main():
    args = build_parser().parse_args()
    ProbeListener(args).run()


if __name__ == "__main__":
    main()
