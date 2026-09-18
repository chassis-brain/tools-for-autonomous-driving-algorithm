from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DisplayConfig:
    enabled: bool = True
    camera_id: str = "CAM_FRONT"
    width: int = 960
    height: int = 540


@dataclass
class ExpertConfig:
    module: str = ""
    entry_point: str = ""
    config: str = ""


@dataclass
class CollectorConfig:
    mode: str = "manual"
    route_xml: str = ""
    route_id: str = ""
    weather_id: int = 0
    output_root: str = "outputs"
    frequency_hz: float = 10.0
    sensor_profile: str = "full"
    jpeg_quality: int = 20
    display: DisplayConfig = field(default_factory=DisplayConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    source_path: Path = field(default=Path(), repr=False)

    @classmethod
    def load(cls, path: str) -> "CollectorConfig":
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            raw: Dict[str, Any] = yaml.safe_load(handle) or {}
        raw["display"] = DisplayConfig(**(raw.get("display") or {}))
        raw["expert"] = ExpertConfig(**(raw.get("expert") or {}))
        cfg = cls(**raw)
        cfg.source_path = source
        if cfg.route_xml and not Path(cfg.route_xml).is_absolute():
            cfg.route_xml = str((source.parent.parent / cfg.route_xml).resolve())
        if cfg.output_root and not Path(cfg.output_root).is_absolute():
            cfg.output_root = str((source.parent.parent / cfg.output_root).resolve())
        if cfg.expert.config and not Path(cfg.expert.config).is_absolute():
            cfg.expert.config = str((source.parent / cfg.expert.config).resolve())
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode not in {"manual", "expert", "hybrid"}:
            raise ValueError("mode 必须是 manual、expert 或 hybrid")
        if self.sensor_profile not in {"full", "camera_only"}:
            raise ValueError("sensor_profile 必须是 full 或 camera_only")
        if not 0 < self.frequency_hz <= 20:
            raise ValueError("frequency_hz 必须在 (0, 20] 范围内")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality 必须在 [1, 100] 范围内")
        if self.mode in {"expert", "hybrid"} and not self.expert.module:
            raise ValueError("expert/hybrid 模式必须配置 expert.module")

