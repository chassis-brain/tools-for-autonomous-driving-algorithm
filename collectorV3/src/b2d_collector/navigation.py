from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple


@dataclass
class NavigationSignal:
    near_xy: Tuple[float, float] = (0.0, 0.0)
    far_xy: Tuple[float, float] = (0.0, 0.0)
    near_command: int = 4
    far_command: int = 4


class RouteProgress:
    def __init__(self) -> None:
        self.plan: List[Tuple[Any, Any]] = []
        self.index = 0

    def set_plan(self, global_plan_world_coord: Sequence[Tuple[Any, Any]]) -> None:
        self.plan = list(global_plan_world_coord or [])
        self.index = 0

    @staticmethod
    def _location(transform: Any) -> Any:
        return transform.location if hasattr(transform, "location") else transform

    @staticmethod
    def _command_value(command: Any) -> int:
        return int(getattr(command, "value", command if command is not None else 4))

    def update(self, ego_location: Any) -> NavigationSignal:
        if not self.plan or ego_location is None:
            return NavigationSignal()
        search_end = min(len(self.plan), self.index + 30)
        distances = []
        for idx in range(self.index, search_end):
            loc = self._location(self.plan[idx][0])
            distances.append(((loc.x - ego_location.x) ** 2 + (loc.y - ego_location.y) ** 2, idx))
        if distances:
            self.index = min(distances)[1]
        near_idx = min(self.index + 2, len(self.plan) - 1)
        far_idx = min(self.index + 12, len(self.plan) - 1)
        near_loc = self._location(self.plan[near_idx][0])
        far_loc = self._location(self.plan[far_idx][0])
        return NavigationSignal(
            near_xy=(float(near_loc.x), float(near_loc.y)),
            far_xy=(float(far_loc.x), float(far_loc.y)),
            near_command=self._command_value(self.plan[near_idx][1]),
            far_command=self._command_value(self.plan[far_idx][1]),
        )

