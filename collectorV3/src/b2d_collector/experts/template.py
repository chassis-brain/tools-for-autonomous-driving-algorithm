from __future__ import annotations

from typing import Any


def get_entry_point():
    return "ExpertTemplate"


class ExpertTemplate:
    """Minimal contract for a new expert-driving implementation.

    Implement this class (or replace the module entirely) with your planner/controller.
    The collector's ExpertAdapter expects:
      - setup(config_path)
      - sensors()
      - set_global_plan(gps_plan, world_plan)
      - run_step(input_data, timestamp)
      - destroy() [optional]

    get_expert_assessment() is optional and may return a NumPy-compatible vector.
    """

    def __init__(self):
        self.config_path = ""
        self.global_plan_gps = None
        self.global_plan_world = None

    def setup(self, config_path: str):
        self.config_path = config_path or ""

    def sensors(self):
        # Reuse collector sensors whenever possible. Add only sensors that the
        # expert genuinely requires.
        return []

    def set_global_plan(self, gps_plan, world_plan):
        self.global_plan_gps = gps_plan
        self.global_plan_world = world_plan

    def run_step(self, input_data: dict, timestamp: float) -> Any:
        raise NotImplementedError(
            "ExpertTemplate is only a scaffold. Implement the new expert planner/controller "
            "before running expert or hybrid collection."
        )

    def get_expert_assessment(self):
        return None

    def destroy(self):
        pass
