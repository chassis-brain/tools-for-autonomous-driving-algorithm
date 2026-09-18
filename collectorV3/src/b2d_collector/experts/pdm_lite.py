from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GARAGE_TEAM_CODE = PROJECT_ROOT / "third_party" / "carla_garage" / "team_code"

if str(GARAGE_TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(GARAGE_TEAM_CODE))

from autopilot import AutoPilot


def _clone_scenario_value(value):
    """Copy Python containers while keeping CARLA actor proxies by reference."""
    if isinstance(value, list):
        return [_clone_scenario_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_scenario_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_scenario_value(item) for key, item in value.items()}
    return value


def _scenario_key(item):
    try:
        scenario_type, data = item
    except Exception:
        return (repr(item), ())

    actor_ids = []

    def visit(value):
        if isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif hasattr(value, "id"):
            try:
                actor_ids.append(int(value.id))
            except Exception:
                pass

    visit(data)
    return (str(scenario_type), tuple(actor_ids))


def get_entry_point():
    return "PdmLiteExpert"


class PdmLiteExpert(AutoPilot):
    """CARLA Garage PDM-Lite expert with isolated replay-time shadow warm-up.

    During replay, PDM updates its own route planner/controller and a private
    copy of CarlaDataProvider.active_scenarios. ScenarioRunner's real list is
    restored after every shadow step.

    At handoff, scenarios activated on that exact tick are first merged into the
    private view, then the warmed private bookkeeping is published once. That
    keeps PDM from re-processing route shifts it already prepared in shadow mode.

    Garage AutoPilot._init must retain the project's _b2d_shadow_safe guard
    around the bugged-two-wheeler actor.destroy() block.
    """

    def __init__(self):
        super().__init__("")
        self._b2d_shadow_mode = False
        self._b2d_shadow_scenarios = None
        self._b2d_shadow_seen_keys = set()
        self._b2d_shadow_steps = 0
        # Read by the tiny guard patched into Garage AutoPilot._init().
        self._b2d_shadow_safe = True

    def setup(self, config_path):
        # Do not let Garage claim/overwrite the collector's SAVE_PATH.
        saved = os.environ.pop("SAVE_PATH", None)
        try:
            super().setup(config_path or "")
        finally:
            if saved is not None:
                os.environ["SAVE_PATH"] = saved

    def sensors(self):
        # The collector already supplies IMU/SPEED. PDM additionally needs map.
        return [
            {
                "type": "sensor.opendrive_map",
                "reading_frequency": 1e-6,
                "id": "hd_map",
            }
        ]

    @staticmethod
    def _assert_shadow_safe_autopilot() -> None:
        try:
            source = inspect.getsource(AutoPilot._init)
        except Exception:
            # Source inspection can fail in unusual packaging layouts. The
            # runtime flag is still enabled through _b2d_shadow_safe.
            return
        if "_b2d_shadow_safe" not in source:
            raise RuntimeError(
                "PDM shadow mode requires the existing Garage AutoPilot._init "
                "guard that skips actor.destroy() when _b2d_shadow_safe is true"
            )

    def _import_real_scenarios(self, real_scenarios) -> None:
        if self._b2d_shadow_scenarios is None:
            self._b2d_shadow_scenarios = []

        for item in real_scenarios:
            key = _scenario_key(item)
            if key in self._b2d_shadow_seen_keys:
                continue
            self._b2d_shadow_scenarios.append(_clone_scenario_value(item))
            self._b2d_shadow_seen_keys.add(key)

    def set_shadow_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._b2d_shadow_mode:
            return

        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        real_scenarios = getattr(CarlaDataProvider, "active_scenarios", [])

        if enabled:
            self._assert_shadow_safe_autopilot()
            # Seed immediately; later shadow steps import newly activated cases.
            self._import_real_scenarios(real_scenarios)
            self._b2d_shadow_mode = True
            return

        # Handoff: also import scenarios activated on the handoff tick before
        # publishing the warmed private bookkeeping.
        self._import_real_scenarios(real_scenarios)
        if self._b2d_shadow_scenarios is not None:
            CarlaDataProvider.active_scenarios = _clone_scenario_value(
                self._b2d_shadow_scenarios
            )

        self._b2d_shadow_mode = False
        print(
            "[PDM-Shadow] handoff after %d shadow step(s); active_scenarios=%d"
            % (
                self._b2d_shadow_steps,
                len(getattr(CarlaDataProvider, "active_scenarios", [])),
            ),
            flush=True,
        )

    def run_step(self, input_data, timestamp):
        adapted = dict(input_data)
        if "imu" not in adapted and "IMU" in adapted:
            adapted["imu"] = adapted["IMU"]
        if "speed" not in adapted and "SPEED" in adapted:
            adapted["speed"] = adapted["SPEED"]

        if not self._b2d_shadow_mode:
            return super().run_step(adapted, timestamp)

        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        # Preserve the exact ScenarioRunner-owned object. PDM receives a private
        # container, so sorting/removal/flag mutations cannot leak during replay.
        real_scenarios = getattr(CarlaDataProvider, "active_scenarios", [])
        self._import_real_scenarios(real_scenarios)

        try:
            CarlaDataProvider.active_scenarios = self._b2d_shadow_scenarios
            control = super().run_step(adapted, timestamp)
            # AutoPilot may replace the list object while processing scenarios.
            self._b2d_shadow_scenarios = CarlaDataProvider.active_scenarios
            self._b2d_shadow_steps += 1
            return control
        finally:
            CarlaDataProvider.active_scenarios = real_scenarios
