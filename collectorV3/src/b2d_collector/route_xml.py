from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float
    yaw: Optional[float] = None


@dataclass(frozen=True)
class Scenario:
    name: str
    type: str
    trigger: Point
    parameters: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Route:
    id: str
    town: str
    road_id: str
    waypoints: List[Point]
    scenarios: List[Scenario]
    weathers: List[Dict[str, str]]

    @property
    def scenario_name(self) -> str:
        if not self.scenarios:
            return "RouteOnly"
        unique = list(dict.fromkeys(item.type for item in self.scenarios))
        return unique[0] if len(unique) == 1 else "MixedScenario"


class RouteCatalog:
    def __init__(self, path: str):
        self.path = Path(path).expanduser().resolve()
        self.routes = self._parse()

    @staticmethod
    def _point(node: ET.Element) -> Point:
        return Point(
            x=float(node.attrib["x"]),
            y=float(node.attrib["y"]),
            z=float(node.attrib.get("z", 0.0)),
            yaw=float(node.attrib["yaw"]) if "yaw" in node.attrib else None,
        )

    def _parse(self) -> List[Route]:
        root = ET.parse(str(self.path)).getroot()
        if root.tag != "routes":
            raise ValueError("XML 根节点必须是 <routes>")
        routes: List[Route] = []
        for node in root.findall("route"):
            scenarios: List[Scenario] = []
            for sc in node.findall("./scenarios/scenario"):
                trigger = sc.find("trigger_point")
                if trigger is None:
                    raise ValueError("场景 %s 缺少 <trigger_point>" % sc.attrib.get("name", "<unnamed>"))
                parameters = {}
                for child in sc:
                    if child.tag != "trigger_point":
                        parameters[child.tag] = child.attrib.get("value", (child.text or "").strip())
                scenarios.append(
                    Scenario(
                        name=sc.attrib.get("name", ""),
                        type=sc.attrib.get("type", ""),
                        trigger=self._point(trigger),
                        parameters=parameters,
                    )
                )
            routes.append(
                Route(
                    id=node.attrib.get("id", ""),
                    town=node.attrib.get("town", ""),
                    road_id=node.attrib.get("road_id", ""),
                    waypoints=[self._point(p) for p in node.findall("./waypoints/position")],
                    scenarios=scenarios,
                    weathers=[dict(w.attrib) for w in node.findall("./weathers/weather")],
                )
            )
        return routes

    def get(self, route_id: str = "") -> Route:
        if route_id:
            matches = [route for route in self.routes if route.id == route_id]
            if len(matches) != 1:
                raise ValueError("route_id=%s 匹配到 %d 条路线" % (route_id, len(matches)))
            return matches[0]
        if len(self.routes) != 1:
            raise ValueError("XML 含多条路线时必须显式指定 route_id")
        return self.routes[0]

    def validate(self) -> List[str]:
        warnings: List[str] = []
        ids = [r.id for r in self.routes]
        if not self.routes:
            raise ValueError("XML 中没有 <route>")
        if len(ids) != len(set(ids)):
            raise ValueError("route id 必须唯一")
        for route in self.routes:
            if not route.id or not route.town:
                raise ValueError("每条 route 必须有 id 与 town")
            if len(route.waypoints) < 2:
                raise ValueError("route %s 至少需要两个 waypoint" % route.id)
            for scenario in route.scenarios:
                if not scenario.name or not scenario.type:
                    raise ValueError("route %s 的 scenario 必须有 name 与 type" % route.id)
            percentages = {w.get("route_percentage") for w in route.weathers}
            if route.weathers and not {"0", "100"}.issubset(percentages):
                warnings.append("route %s 的天气建议覆盖 route_percentage=0 和 100" % route.id)
            if not route.scenarios:
                warnings.append("route %s 没有交互场景，将作为纯路线工况" % route.id)
        return warnings

    def summary(self) -> List[dict]:
        return [
            {
                "id": route.id,
                "town": route.town,
                "road_id": route.road_id,
                "waypoints": len(route.waypoints),
                "scenarios": [s.type for s in route.scenarios],
                "weather_keyframes": len(route.weathers),
            }
            for route in self.routes
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="校验/查看 Bench2Drive route XML")
    parser.add_argument("xml")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出摘要")
    args = parser.parse_args()
    catalog = RouteCatalog(args.xml)
    warnings = catalog.validate()
    if args.json:
        print(json.dumps({"routes": catalog.summary(), "warnings": warnings}, ensure_ascii=False, indent=2))
    else:
        for item in catalog.summary():
            print("{id}: {town}, {waypoints} waypoints, scenarios={scenarios}".format(**item))
        for warning in warnings:
            print("WARNING:", warning)


if __name__ == "__main__":
    main()

