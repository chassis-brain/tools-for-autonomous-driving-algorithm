from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

from b2d_collector.config import DisplayConfig, ExpertConfig


@dataclass
class ReplaySettings:
    tape_dir: str = ""
    intervention_spec: str = ""
    shadow_pdm: bool = True
    sensor_warmup_seconds: float = 1.0
    abort_after_tape: bool = True
    require_complete_tape: bool = True
    terminate_on_end: bool = True


@dataclass
class ReplayCollectorConfig:
    route_xml: str = ""
    route_id: str = ""
    weather_id: int = 0
    output_root: str = "outputs_recovery"
    frequency_hz: float = 10.0
    sensor_profile: str = "full"
    jpeg_quality: int = 20
    display: DisplayConfig = field(default_factory=lambda: DisplayConfig(enabled=False))
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    replay: ReplaySettings = field(default_factory=ReplaySettings)
    source_path: Path = field(default=Path(), repr=False)

    @classmethod
    def load(cls, path: str) -> "ReplayCollectorConfig":
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            raw: Dict[str, Any] = yaml.safe_load(handle) or {}

        raw["display"] = DisplayConfig(**(raw.get("display") or {"enabled": False}))
        raw["expert"] = ExpertConfig(**(raw.get("expert") or {}))
        raw["replay"] = ReplaySettings(**(raw.get("replay") or {}))
        cfg = cls(**raw)
        cfg.source_path = source

        project_root = source.parent.parent
        for attr in ("route_xml", "output_root"):
            value = getattr(cfg, attr)
            if value and not Path(value).is_absolute():
                setattr(cfg, attr, str((project_root / value).resolve()))
        if cfg.expert.config and not Path(cfg.expert.config).is_absolute():
            cfg.expert.config = str((source.parent / cfg.expert.config).resolve())
        if cfg.replay.tape_dir and not Path(cfg.replay.tape_dir).is_absolute():
            cfg.replay.tape_dir = str((project_root / cfg.replay.tape_dir).resolve())
        if cfg.replay.intervention_spec and not Path(cfg.replay.intervention_spec).is_absolute():
            cfg.replay.intervention_spec = str((project_root / cfg.replay.intervention_spec).resolve())
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.sensor_profile not in {"full", "camera_only"}:
            raise ValueError("sensor_profile must be full or camera_only")
        if not 0 < self.frequency_hz <= 20:
            raise ValueError("frequency_hz must be in (0, 20]")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if not self.expert.module:
            raise ValueError("expert.module is required")
        if not self.replay.tape_dir:
            raise ValueError("replay.tape_dir is required")
        if not self.replay.intervention_spec:
            raise ValueError("replay.intervention_spec is required")
        if self.replay.sensor_warmup_seconds < 0:
            raise ValueError("sensor_warmup_seconds must be >= 0")
