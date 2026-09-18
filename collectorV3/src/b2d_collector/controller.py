from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Tuple

import cv2
import numpy as np


def _load_module(value: str) -> ModuleType:
    path = Path(value).expanduser()
    if path.suffix == ".py" or path.exists():
        path = path.resolve()
        spec = importlib.util.spec_from_file_location("b2d_user_expert", str(path))
        if spec is None or spec.loader is None:
            raise ImportError("无法加载专家模块: %s" % path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(value)


class ExpertAdapter:
    def __init__(self, module: str, entry_point: str, config: str):
        loaded = _load_module(module)
        class_name = entry_point or loaded.get_entry_point()
        self.agent = getattr(loaded, class_name)()
        self.agent.setup(config)

    def sensors(self):
        return self.agent.sensors()

    def set_global_plan(self, gps_plan, world_plan):
        self.agent.set_global_plan(gps_plan, world_plan)

    def run_step(self, input_data, timestamp):
        return self.agent.run_step(input_data, timestamp)

    def set_shadow_mode(self, enabled: bool) -> None:
        setter = getattr(self.agent, "set_shadow_mode", None)
        if setter is not None:
            setter(bool(enabled))

    def assessment(self) -> Optional[np.ndarray]:
        getter = getattr(self.agent, "get_expert_assessment", None)
        if getter is None:
            return None
        value = getter()
        return None if value is None else np.asarray(value, dtype=np.float32)

    def destroy(self):
        destroy = getattr(self.agent, "destroy", None)
        if destroy:
            destroy()


class KeyboardController:
    """Pygame keyboard controller. Controls: WASD/arrows, space, R, P, T, F, ESC."""

    def __init__(self, display_cfg: Any):
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.font.init()
        self.enabled = bool(display_cfg.enabled)
        self.camera_id = display_cfg.camera_id
        self.window = None
        if self.enabled:
            self.window = pygame.display.set_mode((display_cfg.width, display_cfg.height))
            pygame.display.set_caption("Bench2Drive custom collector")
        self.clock = pygame.time.Clock()
        self.reverse = False
        self.paused = False
        self.takeover = False
        self.failure_active = False
        self.quit_requested = False
        self.events = []

    def _handle_events(self, timestamp: float) -> None:
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit_requested = True
            if event.type != pg.KEYUP:
                continue
            if event.key == pg.K_ESCAPE:
                self.quit_requested = True
            elif event.key == pg.K_r:
                self.reverse = not self.reverse
                self.events.append((timestamp, "reverse", self.reverse))
            elif event.key == pg.K_p:
                self.paused = not self.paused
                self.events.append((timestamp, "recording_paused", self.paused))
            elif event.key == pg.K_t:
                self.takeover = not self.takeover
                self.events.append((timestamp, "manual_takeover", self.takeover))
            elif event.key == pg.K_f:
                self.failure_active = not self.failure_active
                self.events.append((timestamp, "failure_segment", self.failure_active))

    def control(self, timestamp: float):
        import carla

        self._handle_events(timestamp)
        keys = self.pygame.key.get_pressed()
        throttle = 0.7 if (keys[self.pygame.K_w] or keys[self.pygame.K_UP]) else 0.0
        brake = 1.0 if (keys[self.pygame.K_s] or keys[self.pygame.K_DOWN]) else 0.0
        left = keys[self.pygame.K_a] or keys[self.pygame.K_LEFT]
        right = keys[self.pygame.K_d] or keys[self.pygame.K_RIGHT]
        steer = -0.65 if left else (0.65 if right else 0.0)
        return carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=bool(keys[self.pygame.K_SPACE]),
            reverse=self.reverse,
        )

    def render(self, input_data: dict, mode: str, timestamp: float) -> None:
        if not self.enabled or self.window is None or self.camera_id not in input_data:
            return
        image = input_data[self.camera_id][1]
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        surface = self.pygame.surfarray.make_surface(np.swapaxes(image, 0, 1))
        surface = self.pygame.transform.smoothscale(surface, self.window.get_size())
        self.window.blit(surface, (0, 0))
        font = self.pygame.font.Font(None, 28)
        state = "%s | rec:%s | failure:%s | T takeover, F failure, P pause" % (
            mode, "OFF" if self.paused else "ON", self.failure_active
        )
        self.window.blit(font.render(state, True, (255, 230, 30)), (16, 16))
        self.pygame.display.flip()
        self.clock.tick(60)

    def pop_events(self):
        events, self.events = self.events, []
        return events

    def destroy(self):
        self.pygame.quit()


class ControlMux:
    def __init__(self, mode: str, keyboard: KeyboardController, expert: Optional[ExpertAdapter]):
        self.mode = mode
        self.keyboard = keyboard
        self.expert = expert

    def step(self, input_data: dict, timestamp: float) -> Tuple[Any, str]:
        manual = self.keyboard.control(timestamp)
        if self.mode == "manual":
            return manual, "manual"
        if self.mode == "expert":
            return self.expert.run_step(input_data, timestamp), "expert"  # type: ignore[union-attr]
        if self.keyboard.takeover:
            return manual, "manual_takeover"
        return self.expert.run_step(input_data, timestamp), "expert"  # type: ignore[union-attr]

