from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _wrap_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


class _Pid:
    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float = 10.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = abs(float(integral_limit))
        self.integral = 0.0
        self.previous: Optional[float] = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous = None

    def step(self, error: float, dt: float) -> float:
        dt = _clamp(dt, 0.02, 0.25)
        err = float(error)
        self.integral = _clamp(
            self.integral + err * dt,
            -self.integral_limit,
            self.integral_limit,
        )
        derivative = 0.0 if self.previous is None else (err - self.previous) / dt
        self.previous = err
        return self.kp * err + self.ki * self.integral + self.kd * derivative


class ManualTrajectoryFollower:
    """Strict spatial path follower for user-authored P(s) + v(s).

    The plan carries no execution timestamps. Each CARLA tick projects the ego
    state onto the dense spatial path, reads target speed from position s, then
    uses feedback PID plus curvature feed-forward to produce VehicleControl.
    """

    def __init__(self, plan_path: str):
        self.plan_path = Path(plan_path).expanduser().resolve()
        with self.plan_path.open("r", encoding="utf-8") as handle:
            self.document = json.load(handle)

        schema = str(self.document.get("schema", ""))
        if schema != "b2d-manual-plan-v2":
            raise ValueError(
                "manual PID requires b2d-manual-plan-v2, got %r" % schema
            )

        generated = self.document.get("generated") or {}
        rel = generated.get("trajectory_file")
        if not rel:
            raise ValueError("manual plan has no generated.trajectory_file")
        trajectory_path = Path(rel)
        if not trajectory_path.is_absolute():
            trajectory_path = self.plan_path.parent / trajectory_path
        self.trajectory_path = trajectory_path.resolve()
        if not self.trajectory_path.is_file():
            raise FileNotFoundError(str(self.trajectory_path))

        data = np.load(str(self.trajectory_path))
        self.s = np.asarray(data["s"], dtype=float)
        self.x = np.asarray(data["x"], dtype=float)
        self.y = np.asarray(data["y"], dtype=float)
        self.yaw = np.asarray(data["yaw"], dtype=float)
        self.curvature = np.asarray(data["curvature"], dtype=float)
        self.target_speed = np.asarray(data["target_speed"], dtype=float)
        n = len(self.s)
        if n < 2 or not all(len(a) == n for a in (
            self.x, self.y, self.yaw, self.curvature, self.target_speed
        )):
            raise ValueError("invalid manual trajectory array lengths")
        if np.any(np.diff(self.s) < -1e-9):
            raise ValueError("manual trajectory s must be monotonic")

        self.xy = np.column_stack([self.x, self.y])
        self.length_m = float(self.s[-1])
        self.end_xy = np.array([self.x[-1], self.y[-1]], dtype=float)
        self.controller: Dict[str, float] = {
            str(k): float(v) for k, v in (self.document.get("controller") or {}).items()
            if isinstance(v, (int, float))
        }

        self.lateral_pid = _Pid(
            self.controller.get("lateral_kp", 0.45),
            self.controller.get("lateral_ki", 0.015),
            self.controller.get("lateral_kd", 0.10),
            integral_limit=3.0,
        )
        self.speed_pid = _Pid(
            self.controller.get("speed_kp", 0.38),
            self.controller.get("speed_ki", 0.055),
            self.controller.get("speed_kd", 0.06),
            integral_limit=8.0,
        )
        self.heading_kp = self.controller.get("heading_kp", 1.10)
        self.curvature_ff = self.controller.get("curvature_feedforward", 0.85)
        self.wheelbase_m = self.controller.get("wheelbase_m", 2.85)
        self.max_steer_angle_rad = self.controller.get("max_steer_angle_rad", 1.22)
        self.max_throttle = self.controller.get("max_throttle", 0.80)
        self.max_brake = self.controller.get("max_brake", 0.90)
        self.lookahead_base_m = self.controller.get("lookahead_base_m", 0.8)
        self.lookahead_speed_gain = self.controller.get("lookahead_speed_gain", 0.18)
        self.speed_preview_seconds = self.controller.get("speed_preview_seconds", 0.55)

        self._last_index = 0
        self._last_time: Optional[float] = None
        self._last_diag: Dict[str, float] = {}

    def reset(self, world_time: Optional[float] = None) -> None:
        self._last_index = 0
        self._last_time = None if world_time is None else float(world_time)
        self._last_diag = {}
        self.lateral_pid.reset()
        self.speed_pid.reset()

    @property
    def progress_s(self) -> float:
        return float(self.s[min(self._last_index, len(self.s) - 1)])

    @property
    def progress_ratio(self) -> float:
        if self.length_m <= 1e-6:
            return 1.0
        return _clamp(self.progress_s / self.length_m, 0.0, 1.0)

    def diagnostics(self) -> Dict[str, float]:
        return dict(self._last_diag)

    def _nearest_index(self, x: float, y: float) -> int:
        # Local forward-biased search prevents jumps to a crossing branch while
        # still allowing the controller to recover if the vehicle deviates.
        start = max(0, self._last_index - 8)
        stop = min(len(self.s), self._last_index + 160)
        pts = self.xy[start:stop]
        if len(pts) == 0:
            return self._last_index
        d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
        idx = start + int(np.argmin(d2))
        # Spatial progress is monotonic for this correction segment.
        idx = max(self._last_index, idx)
        self._last_index = min(idx, len(self.s) - 1)
        return self._last_index

    def _index_at_s(self, query_s: float) -> int:
        idx = int(np.searchsorted(self.s, float(query_s), side="left"))
        return min(max(idx, 0), len(self.s) - 1)

    @staticmethod
    def _ego_speed(hero) -> float:
        vel = hero.get_velocity()
        return math.sqrt(float(vel.x) ** 2 + float(vel.y) ** 2 + float(vel.z) ** 2)

    def compute(self, x: float, y: float, yaw_rad: float, speed_mps: float,
                world_time: float) -> Tuple[float, float, float, Dict[str, float]]:
        """Pure numeric control calculation; useful for unit tests."""
        nearest = self._nearest_index(x, y)
        current_s = float(self.s[nearest])

        dt = 0.1 if self._last_time is None else float(world_time) - self._last_time
        if not math.isfinite(dt) or dt <= 0.0:
            dt = 0.1
        dt = _clamp(dt, 0.02, 0.25)
        self._last_time = float(world_time)

        lookahead = self.lookahead_base_m + self.lookahead_speed_gain * max(speed_mps, 0.0)
        target_idx = self._index_at_s(current_s + lookahead)
        tx = float(self.x[target_idx])
        ty = float(self.y[target_idx])
        target_yaw = float(self.yaw[target_idx])

        dx = tx - float(x)
        dy = ty - float(y)
        c = math.cos(float(yaw_rad))
        sn = math.sin(float(yaw_rad))
        # CARLA local +y points to vehicle right; positive steer is right.
        lateral_error = -sn * dx + c * dy
        heading_error = _wrap_pi(target_yaw - float(yaw_rad))

        lateral_feedback = self.lateral_pid.step(lateral_error, dt)
        heading_feedback = self.heading_kp * heading_error
        kappa = float(self.curvature[target_idx])
        ff_angle = math.atan(self.wheelbase_m * kappa)
        steer_ff = ff_angle / max(abs(self.max_steer_angle_rad), 1e-3)
        steer = _clamp(
            lateral_feedback + heading_feedback + self.curvature_ff * steer_ff,
            -1.0,
            1.0,
        )

        preview_s = current_s + max(0.5, max(speed_mps, 0.0) * self.speed_preview_seconds)
        speed_idx = self._index_at_s(preview_s)
        target_speed = float(self.target_speed[speed_idx])
        speed_error = target_speed - float(speed_mps)
        accel_cmd = self.speed_pid.step(speed_error, dt)

        if target_speed <= 0.05 and speed_mps <= 0.2:
            throttle, brake = 0.0, 0.35
        elif accel_cmd >= 0.0:
            throttle = _clamp(accel_cmd, 0.0, self.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = _clamp(-accel_cmd, 0.0, self.max_brake)

        diag = {
            "path_s_m": current_s,
            "path_progress": self.progress_ratio,
            "lateral_error_m": float(lateral_error),
            "heading_error_rad": float(heading_error),
            "target_speed_mps": target_speed,
            "actual_speed_mps": float(speed_mps),
            "target_index": float(target_idx),
            "steer": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
        }
        self._last_diag = diag
        return throttle, steer, brake, diag

    def run_step(self, hero, world_time: float):
        import carla

        transform = hero.get_transform()
        speed = self._ego_speed(hero)
        throttle, steer, brake, _diag = self.compute(
            float(transform.location.x),
            float(transform.location.y),
            math.radians(float(transform.rotation.yaw)),
            speed,
            float(world_time),
        )
        return carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
