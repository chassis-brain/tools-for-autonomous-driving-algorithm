from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import carla

from .manual_plan import build_dense_trajectory, cubic_bspline_dense, path_geometry, polyline_cumulative_s


def _normalize_angle(rad: float) -> float:
    return (float(rad) + math.pi) % (2.0 * math.pi) - math.pi


def _nested(doc: Dict[str, Any], *keys: str) -> Any:
    value: Any = doc
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _case_stem(plan_path: Path) -> str:
    name = plan_path.name
    if name.endswith(".plan.json"):
        return name[:-len(".plan.json")]
    return plan_path.stem


def _trajectory_candidates(plan_path: Path, doc: Dict[str, Any]) -> Iterable[Path]:
    values = [
        _nested(doc, "generated", "trajectory_file"),
        doc.get("trajectory_file"),
        doc.get("trajectory_npz"),
        _nested(doc, "trajectory", "file"),
    ]
    for value in values:
        if not value:
            continue
        p = Path(str(value)).expanduser()
        if not p.is_absolute():
            p = plan_path.parent / p
        yield p.resolve()

    stem = _case_stem(plan_path)
    yield (plan_path.parent / (stem + ".trajectory.npz")).resolve()
    yield (plan_path.parent / "generated" / (stem + ".trajectory.npz")).resolve()


def _speed_ctrl_array(value: Any) -> Optional[np.ndarray]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    rows = []
    for item in value:
        if isinstance(item, dict):
            s = item.get("s", item.get("position_m", item.get("position")))
            v = item.get("v", item.get("speed_mps", item.get("speed")))
            if s is None or v is None:
                return None
            rows.append([float(s), float(v)])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            rows.append([float(item[0]), float(item[1])])
        else:
            return None
    arr = np.asarray(rows, dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    # De-duplicate s positions; the latest point wins.
    unique = []
    for row in arr:
        if unique and abs(float(row[0]) - float(unique[-1][0])) < 1e-6:
            unique[-1] = row
        else:
            unique.append(row)
    return np.asarray(unique, dtype=float) if len(unique) >= 2 else None


def _path_ctrl_array(doc: Dict[str, Any]) -> Optional[np.ndarray]:
    candidates = [
        doc.get("path_control_points"),
        doc.get("anchors"),
        _nested(doc, "path", "control_points"),
        _nested(doc, "path", "anchors"),
        _nested(doc, "manual_path", "anchors"),
    ]
    for value in candidates:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        rows = []
        ok = True
        for item in value:
            if isinstance(item, dict):
                if item.get("x") is None or item.get("y") is None:
                    ok = False
                    break
                rows.append([float(item["x"]), float(item["y"])])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                rows.append([float(item[0]), float(item[1])])
            else:
                ok = False
                break
        if ok:
            return np.asarray(rows, dtype=float)
    return None


def _speed_ctrl_from_doc(doc: Dict[str, Any]) -> Optional[np.ndarray]:
    for value in (
        doc.get("speed_control_points"),
        _nested(doc, "speed", "control_points"),
        _nested(doc, "speed", "points"),
        _nested(doc, "speed_profile", "points"),
    ):
        out = _speed_ctrl_array(value)
        if out is not None:
            return out
    return None


def _dense_from_embedded(doc: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    holders = [doc.get("dense_trajectory"), doc.get("trajectory")]
    for holder in holders:
        if not isinstance(holder, dict):
            continue
        x = holder.get("x")
        y = holder.get("y")
        if not isinstance(x, (list, tuple)) or not isinstance(y, (list, tuple)):
            continue
        if len(x) < 2 or len(x) != len(y):
            continue
        pts = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
        s, yaw, curvature = path_geometry(pts)
        v = holder.get("target_speed", holder.get("speed_mps"))
        if isinstance(v, (list, tuple)) and len(v) == len(x):
            target_speed = np.asarray(v, dtype=float)
        else:
            ctrl = _speed_ctrl_from_doc(doc)
            if ctrl is None:
                return None
            dense = build_dense_trajectory(pts, ctrl)
            target_speed = dense["target_speed"]
        return {
            "s": s,
            "x": pts[:, 0],
            "y": pts[:, 1],
            "yaw": yaw,
            "curvature": curvature,
            "target_speed": target_speed,
        }
    return None


def load_manual_trajectory(plan_path: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Optional[Path]]:
    path = Path(plan_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("manual plan JSON not found: %s" % path)
    with path.open("r", encoding="utf-8") as handle:
        doc = json.load(handle)

    checked = []
    for candidate in _trajectory_candidates(path, doc):
        if candidate in checked:
            continue
        checked.append(candidate)
        if not candidate.is_file():
            continue
        with np.load(str(candidate)) as data:
            if "x" not in data or "y" not in data:
                raise RuntimeError("trajectory NPZ lacks x/y: %s" % candidate)
            x = np.asarray(data["x"], dtype=float).reshape(-1)
            y = np.asarray(data["y"], dtype=float).reshape(-1)
            if len(x) < 2 or len(x) != len(y):
                raise RuntimeError("invalid x/y arrays in %s" % candidate)
            pts = np.column_stack([x, y])
            if "s" in data and len(data["s"]) == len(x):
                s = np.asarray(data["s"], dtype=float).reshape(-1)
            else:
                s = polyline_cumulative_s(pts)
            if "yaw" in data and len(data["yaw"]) == len(x):
                yaw = np.asarray(data["yaw"], dtype=float).reshape(-1)
            else:
                _, yaw, _ = path_geometry(pts)
            if "curvature" in data and len(data["curvature"]) == len(x):
                curvature = np.asarray(data["curvature"], dtype=float).reshape(-1)
            else:
                _, _, curvature = path_geometry(pts)
            speed_key = "target_speed" if "target_speed" in data else ("speed" if "speed" in data else None)
            if speed_key is None:
                raise RuntimeError("trajectory NPZ lacks target_speed: %s" % candidate)
            target_speed = np.asarray(data[speed_key], dtype=float).reshape(-1)
            if len(target_speed) != len(x):
                raise RuntimeError("target_speed length mismatch in %s" % candidate)
        traj = {
            "s": s,
            "x": x,
            "y": y,
            "yaw": yaw,
            "curvature": curvature,
            "target_speed": np.maximum(target_speed, 0.0),
        }
        _validate_trajectory(traj, path)
        return doc, traj, candidate

    embedded = _dense_from_embedded(doc)
    if embedded is not None:
        _validate_trajectory(embedded, path)
        return doc, embedded, None

    ctrl = _path_ctrl_array(doc)
    speed_ctrl = _speed_ctrl_from_doc(doc)
    if ctrl is None:
        raise RuntimeError(
            "manual plan has no usable trajectory NPZ and no path control points: %s; checked: %s"
            % (path, ", ".join(str(p) for p in checked))
        )
    if speed_ctrl is None:
        raise RuntimeError("manual plan has no usable speed control points: %s" % path)

    settings = doc.get("settings") or {}
    spacing = float(settings.get("dense_sample_spacing_m", 0.2))
    path_xy = cubic_bspline_dense(ctrl, spacing)
    dense = build_dense_trajectory(path_xy, speed_ctrl)
    traj = {key: np.asarray(dense[key], dtype=float) for key in (
        "s", "x", "y", "yaw", "curvature", "target_speed"
    )}
    _validate_trajectory(traj, path)
    return doc, traj, None


def _validate_trajectory(traj: Dict[str, np.ndarray], plan_path: Path) -> None:
    n = len(traj["x"])
    if n < 2:
        raise RuntimeError("manual trajectory needs at least two samples: %s" % plan_path)
    for key in ("s", "x", "y", "yaw", "curvature", "target_speed"):
        arr = np.asarray(traj[key], dtype=float).reshape(-1)
        if len(arr) != n:
            raise RuntimeError("manual trajectory %s length mismatch: %s" % (key, plan_path))
        if not np.all(np.isfinite(arr)):
            raise RuntimeError("manual trajectory %s contains non-finite values: %s" % (key, plan_path))
    if np.any(np.diff(np.asarray(traj["s"], dtype=float)) < -1e-6):
        raise RuntimeError("manual trajectory s must be non-decreasing: %s" % plan_path)


class ManualTrajectoryFollower:
    """Spatial P(s)+v(s) tracker for the handoff -> free End correction segment.

    Lateral control is a cross-track PID plus heading feedback and curvature
    feed-forward.  Longitudinal control is a speed PID.  There is no planned
    time axis; target speed is sampled only by path position s.
    """

    def __init__(self, plan_path: str, frequency_hz: float = 10.0):
        self.plan_path = str(Path(plan_path).expanduser().resolve())
        self.plan_doc, traj, self.trajectory_file = load_manual_trajectory(self.plan_path)
        self.s = np.asarray(traj["s"], dtype=float)
        self.x = np.asarray(traj["x"], dtype=float)
        self.y = np.asarray(traj["y"], dtype=float)
        self.yaw = np.asarray(traj["yaw"], dtype=float)
        self.curvature = np.asarray(traj["curvature"], dtype=float)
        self.target_speed = np.asarray(traj["target_speed"], dtype=float)

        # Manual Planner V2 writes PID/vehicle tuning under "controller".
        # Older builds used "pid" or "tracking"; retain them as fallbacks.
        cfg = (
            self.plan_doc.get("controller")
            or self.plan_doc.get("pid")
            or self.plan_doc.get("tracking")
            or {}
        )
        self.frequency_hz = max(float(frequency_hz), 1.0)
        self.default_dt = 1.0 / self.frequency_hz
        self.wheelbase_m = float(cfg.get("wheelbase_m", 2.875))

        if cfg.get("max_steer_angle_rad") is not None:
            self.max_steer_angle_rad = max(1e-3, float(cfg["max_steer_angle_rad"]))
        else:
            self.max_steer_angle_rad = math.radians(
                float(cfg.get("max_steer_angle_deg", 35.0))
            )

        self.cte_kp = float(cfg.get("lateral_kp", cfg.get("cte_kp", 0.14)))
        self.cte_ki = float(cfg.get("lateral_ki", cfg.get("cte_ki", 0.006)))
        self.cte_kd = float(cfg.get("lateral_kd", cfg.get("cte_kd", 0.10)))
        self.heading_kp = float(cfg.get("heading_kp", 0.90))
        self.curvature_ff = float(
            cfg.get("curvature_feedforward", cfg.get("curvature_ff", 1.0))
        )

        self.speed_kp = float(cfg.get("speed_kp", 0.45))
        self.speed_ki = float(cfg.get("speed_ki", 0.08))
        self.speed_kd = float(cfg.get("speed_kd", 0.035))
        self.max_throttle = float(
            np.clip(float(cfg.get("max_throttle", 1.0)), 0.0, 1.0)
        )
        self.max_brake = float(
            np.clip(float(cfg.get("max_brake", 1.0)), 0.0, 1.0)
        )

        self.lookahead_base_m = float(cfg.get("lookahead_base_m", 1.2))
        self.lookahead_speed_gain = float(cfg.get("lookahead_speed_gain", 0.20))
        self.lookahead_max_m = float(cfg.get("lookahead_max_m", 4.0))
        self.speed_preview_seconds = max(
            0.0, float(cfg.get("speed_preview_seconds", 0.55))
        )

        self._last_world_time: Optional[float] = None
        self._last_index = 0
        self._cte_integral = 0.0
        self._last_cte: Optional[float] = None
        self._speed_integral = 0.0
        self._last_speed_error: Optional[float] = None
        self.last_status: Dict[str, float] = {}

    @property
    def path_length_m(self) -> float:
        return float(self.s[-1])

    @property
    def progress_s(self) -> float:
        return float(self.s[min(max(self._last_index, 0), len(self.s) - 1)])

    def reset(self, world_time: Optional[float] = None) -> None:
        self._last_world_time = world_time
        self._cte_integral = 0.0
        self._last_cte = None
        self._speed_integral = 0.0
        self._last_speed_error = None

    def _nearest_index(self, px: float, py: float) -> int:
        n = len(self.x)
        if self._last_index <= 0:
            lo, hi = 0, n
        else:
            # Keep projection local so a path crossing itself cannot jump to a
            # distant future branch.  Allow a small backward margin for noise.
            lo = max(0, self._last_index - 12)
            hi = min(n, self._last_index + 260)
        dx = self.x[lo:hi] - float(px)
        dy = self.y[lo:hi] - float(py)
        idx = lo + int(np.argmin(dx * dx + dy * dy))
        # Progress is effectively monotonic; permit only a tiny regression.
        idx = max(idx, max(0, self._last_index - 3))
        self._last_index = idx
        return idx

    @staticmethod
    def _ego_speed_mps(hero: Any) -> float:
        v = hero.get_velocity()
        return math.sqrt(float(v.x) ** 2 + float(v.y) ** 2 + float(v.z) ** 2)

    def run_step(self, hero: Any, world_time: float) -> carla.VehicleControl:
        if hero is None:
            raise RuntimeError("manual_pid cannot find hero vehicle")

        transform = hero.get_transform()
        px = float(transform.location.x)
        py = float(transform.location.y)
        ego_yaw = math.radians(float(transform.rotation.yaw))
        speed = self._ego_speed_mps(hero)

        if self._last_world_time is None:
            dt = self.default_dt
        else:
            dt = max(1e-3, min(0.5, float(world_time) - float(self._last_world_time)))
        self._last_world_time = float(world_time)

        nearest = self._nearest_index(px, py)
        lookahead_m = min(
            self.lookahead_max_m,
            max(self.lookahead_base_m, self.lookahead_base_m + self.lookahead_speed_gain * speed),
        )
        target_s = min(self.path_length_m, float(self.s[nearest]) + lookahead_m)
        target = int(np.searchsorted(self.s, target_s, side="left"))
        target = min(max(target, nearest), len(self.s) - 1)

        # Positive cross-track error means ego is to the path's right in CARLA's
        # x-forward/y-right coordinate convention; returning to path needs left steer.
        dx = px - float(self.x[nearest])
        dy = py - float(self.y[nearest])
        path_yaw_near = float(self.yaw[nearest])
        right_nx = -math.sin(path_yaw_near)
        right_ny = math.cos(path_yaw_near)
        cte = dx * right_nx + dy * right_ny

        heading_error = _normalize_angle(float(self.yaw[target]) - ego_yaw)
        cte_d = 0.0 if self._last_cte is None else (cte - self._last_cte) / dt
        self._last_cte = cte
        self._cte_integral = float(np.clip(self._cte_integral + cte * dt, -8.0, 8.0))

        ff_angle = math.atan(self.wheelbase_m * float(self.curvature[target]))
        ff_norm = ff_angle / max(self.max_steer_angle_rad, 1e-3)
        cte_feedback = (
            self.cte_kp * cte
            + self.cte_ki * self._cte_integral
            + self.cte_kd * cte_d
        )
        steer = (
            self.curvature_ff * ff_norm
            + self.heading_kp * heading_error
            - cte_feedback
        )
        steer = float(np.clip(steer, -1.0, 1.0))

        # Speed is authored as v(s), not v(t).  Sample it from its own spatial
        # preview position instead of reusing the lateral-lookahead waypoint.
        # This makes the executed speed curve match the GUI-authored profile.
        speed_preview_s = min(
            self.path_length_m,
            float(self.s[nearest]) + max(0.0, speed * self.speed_preview_seconds),
        )
        desired_speed = max(
            0.0,
            float(np.interp(speed_preview_s, self.s, self.target_speed)),
        )
        speed_error = desired_speed - speed
        speed_d = 0.0 if self._last_speed_error is None else (speed_error - self._last_speed_error) / dt
        self._last_speed_error = speed_error
        self._speed_integral = float(np.clip(self._speed_integral + speed_error * dt, -12.0, 12.0))
        accel_cmd = (
            self.speed_kp * speed_error
            + self.speed_ki * self._speed_integral
            + self.speed_kd * speed_d
        )

        if desired_speed <= 0.10 and speed > 0.15:
            throttle, brake = 0.0, min(self.max_brake, 0.45 + 0.20 * speed)
        elif accel_cmd >= 0.0:
            throttle, brake = min(self.max_throttle, accel_cmd), 0.0
        else:
            throttle, brake = 0.0, min(self.max_brake, -accel_cmd)

        self.last_status = {
            "progress_s": float(self.s[nearest]),
            "path_length_m": self.path_length_m,
            "cross_track_error_m": float(cte),
            "heading_error_deg": math.degrees(heading_error),
            "target_speed_mps": desired_speed,
            "speed_preview_s": float(speed_preview_s),
            "speed_mps": speed,
            "steer": steer,
            "throttle": float(throttle),
            "brake": float(brake),
        }
        return carla.VehicleControl(
            throttle=float(throttle),
            steer=steer,
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
