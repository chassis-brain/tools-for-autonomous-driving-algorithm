from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .tape import BehaviorTape


@dataclass(frozen=True)
class InterventionSpec:
    source_run: str
    case_id: str
    intervention_type: str
    description: str
    record_start_time: float
    handoff_time: float
    handoff_x: Optional[float] = None
    handoff_y: Optional[float] = None

    # Legacy v1 cases used a time-based record end.  The v2 three-point schema
    # intentionally leaves this unset and terminates on a free spatial End XY.
    record_end_time: Optional[float] = None

    end_x: Optional[float] = None
    end_y: Optional[float] = None
    end_radius_m: float = 1.5
    end_confirm_frames: int = 3

    replacement_mode: str = "pdm_expert"
    replacement_plan: Optional[str] = None

    time_offset_seconds: float = 0.0
    max_position_error_m: float = 2.0
    max_yaw_error_deg: float = 10.0
    divergence_grace_seconds: float = 1.0
    abort_on_divergence: bool = True

    def validate(self) -> None:
        if self.record_start_time > self.handoff_time:
            raise ValueError("record_start must be <= handoff")
        if self.record_end_time is not None and self.handoff_time > self.record_end_time:
            raise ValueError("handoff must be <= record_end")
        if self.max_position_error_m <= 0:
            raise ValueError("max_position_error_m must be > 0")
        if self.max_yaw_error_deg <= 0:
            raise ValueError("max_yaw_error_deg must be > 0")

        mode = normalize_replacement_mode(self.replacement_mode)
        if mode not in {"pdm_expert", "manual_pid"}:
            raise ValueError("unsupported replacement.mode: %s" % self.replacement_mode)

        # Three-point v2 cases MUST have a spatial End.  Legacy cases can keep
        # their time end for backwards compatibility.
        has_spatial_end = self.end_x is not None and self.end_y is not None
        if self.record_end_time is None and not has_spatial_end:
            raise ValueError("v2 intervention requires end.x/end.y")
        if has_spatial_end:
            if not (math.isfinite(float(self.end_x)) and math.isfinite(float(self.end_y))):
                raise ValueError("end.x/end.y must be finite")
            if self.end_radius_m <= 0:
                raise ValueError("end.radius_m must be > 0")
            if int(self.end_confirm_frames) < 1:
                raise ValueError("end.confirm_frames must be >= 1")

        if mode == "manual_pid" and not self.replacement_plan:
            raise ValueError("manual_pid requires replacement.plan")

    @property
    def has_spatial_end(self) -> bool:
        return self.end_x is not None and self.end_y is not None

    @property
    def end_xy(self):
        """Backward-compatible spatial End accessor used by tests/tools."""
        if not self.has_spatial_end:
            return None
        return (float(self.end_x), float(self.end_y))

    @property
    def manual_plan_path(self) -> Optional[str]:
        """Backward-compatible alias for the resolved Manual PID plan path."""
        return self.replacement_plan


def normalize_replacement_mode(value: Any) -> str:
    text = str(value or "pdm_expert").strip().lower()
    aliases = {
        "pdm": "pdm_expert",
        "expert": "pdm_expert",
        "pdm-expert": "pdm_expert",
        "manual": "manual_pid",
        "manual_path": "manual_pid",
        "manual-path": "manual_pid",
        "manual_path_pid": "manual_pid",
    }
    return aliases.get(text, text)


def _point_time(value: Any, tape: BehaviorTape, name: str) -> float:
    if isinstance(value, (int, float)):
        # Bare numeric values are relative seconds for human-facing specs.
        return tape.first_time + float(value)
    if not isinstance(value, dict):
        raise ValueError("%s must be a number or mapping" % name)
    if value.get("sim_time_s") is not None:
        return float(value["sim_time_s"])
    if value.get("relative_time_s") is not None:
        return tape.first_time + float(value["relative_time_s"])
    if value.get("sample_index") is not None:
        return tape.sim_time_for_index(int(value["sample_index"]))
    raise ValueError("%s needs sim_time_s, relative_time_s or sample_index" % name)


def _legacy_step(raw: Dict[str, Any], key: str, tape: BehaviorTape) -> Optional[float]:
    value = raw.get(key)
    if value is None:
        return None
    return tape.sim_time_for_index(int(value))


def _resolve_plan_path(case_path: Path, value: Any) -> Optional[str]:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    # IMPORTANT: replacement.plan is relative to the case JSON directory.
    # case=/.../cases/case_*.json + plans/foo.plan.json -> /.../cases/plans/...
    if not path.is_absolute():
        path = case_path.parent / path
    return str(path.resolve())


def load_intervention_spec(path: str, tape: BehaviorTape) -> InterventionSpec:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    schema = str(raw.get("schema", ""))
    is_v2 = schema == "b2d-intervention-v2" or (
        isinstance(raw.get("end"), dict) and isinstance(raw.get("replacement"), dict)
    )

    if is_v2:
        start = _point_time(raw.get("record_start"), tape, "record_start")
        handoff = _point_time(raw.get("handoff"), tape, "handoff")

        end_obj = raw.get("end") or {}
        if end_obj.get("x") is None or end_obj.get("y") is None:
            raise ValueError("v2 intervention end needs x and y")

        replacement = raw.get("replacement") or {}
        replay = raw.get("replay") or {}
        spec = InterventionSpec(
            source_run=str(raw.get("source_run", raw.get("source_run_id", tape.run_dir))),
            case_id=str(raw.get("case_id", source.stem)),
            intervention_type=str(raw.get("intervention_type", raw.get("type", "failure"))),
            description=str(raw.get("description", raw.get("note", ""))),
            record_start_time=float(start),
            handoff_time=float(handoff),
            handoff_x=(float(raw.get("handoff", {}).get("x")) if isinstance(raw.get("handoff"), dict) and raw.get("handoff", {}).get("x") is not None else None),
            handoff_y=(float(raw.get("handoff", {}).get("y")) if isinstance(raw.get("handoff"), dict) and raw.get("handoff", {}).get("y") is not None else None),
            record_end_time=None,
            end_x=float(end_obj["x"]),
            end_y=float(end_obj["y"]),
            end_radius_m=float(end_obj.get("radius_m", 1.5)),
            end_confirm_frames=int(end_obj.get("confirm_frames", 3)),
            replacement_mode=normalize_replacement_mode(replacement.get("mode", "pdm_expert")),
            replacement_plan=_resolve_plan_path(source, replacement.get("plan")),
            time_offset_seconds=float(replay.get("time_offset_seconds", raw.get("time_offset_seconds", 0.0))),
            max_position_error_m=float(replay.get("max_position_error_m", 2.0)),
            max_yaw_error_deg=float(replay.get("max_yaw_error_deg", 10.0)),
            divergence_grace_seconds=float(replay.get("divergence_grace_seconds", 1.0)),
            abort_on_divergence=bool(replay.get("abort_on_divergence", True)),
        )
        spec.validate()
        return spec

    # Current/legacy time-window schemas.
    if any(name in raw for name in ("record_start", "handoff", "record_end")):
        start = _point_time(raw.get("record_start"), tape, "record_start")
        handoff = _point_time(raw.get("handoff"), tape, "handoff")
        end = _point_time(raw.get("record_end"), tape, "record_end")
    else:
        start = _legacy_step(raw, "record_start_step", tape)
        handoff = _legacy_step(raw, "handoff_step", tape)
        end = _legacy_step(raw, "record_end_step", tape)
        if end is None and raw.get("record_duration_steps") is not None and handoff is not None:
            end_index = int(raw.get("record_start_step", 0)) + int(raw["record_duration_steps"])
            end = tape.sim_time_for_index(min(end_index, len(tape.frames) - 1))
        if start is None or handoff is None or end is None:
            raise ValueError(
                "legacy intervention spec needs record_start_step, handoff_step and record_end_step"
            )

    replay = raw.get("replay") or {}
    spec = InterventionSpec(
        source_run=str(raw.get("source_run", raw.get("source_run_id", tape.run_dir))),
        case_id=str(raw.get("case_id", source.stem)),
        intervention_type=str(raw.get("intervention_type", raw.get("type", "failure"))),
        description=str(raw.get("description", raw.get("note", ""))),
        record_start_time=float(start),
        handoff_time=float(handoff),
        record_end_time=float(end),
        replacement_mode="pdm_expert",
        time_offset_seconds=float(replay.get("time_offset_seconds", raw.get("time_offset_seconds", 0.0))),
        max_position_error_m=float(replay.get("max_position_error_m", 2.0)),
        max_yaw_error_deg=float(replay.get("max_yaw_error_deg", 10.0)),
        divergence_grace_seconds=float(replay.get("divergence_grace_seconds", 1.0)),
        abort_on_divergence=bool(replay.get("abort_on_divergence", True)),
    )
    spec.validate()
    return spec


def build_intervention_spec(
    tape: BehaviorTape,
    source_run: str,
    case_id: str,
    intervention_type: str,
    description: str,
    record_start_index: int,
    handoff_index: int,
    record_end_index: int,
) -> Dict[str, Any]:
    """Legacy helper retained for existing tests/tools."""
    def point(index: int) -> Dict[str, Any]:
        frame = tape.by_index(index)
        return {
            "sample_index": int(index),
            "relative_time_s": float(frame.sim_time - tape.first_time),
            "sim_time_s": float(frame.sim_time),
        }

    return {
        "schema": "b2d-intervention-v1",
        "source_run": str(source_run),
        "case_id": str(case_id),
        "intervention_type": str(intervention_type),
        "description": str(description),
        "record_start": point(record_start_index),
        "handoff": point(handoff_index),
        "record_end": point(record_end_index),
        "replay": {
            "time_offset_seconds": 0.0,
            "max_position_error_m": 2.0,
            "max_yaw_error_deg": 10.0,
            "divergence_grace_seconds": 1.0,
            "abort_on_divergence": True,
        },
    }
