# -*- coding: utf-8 -*-
from __future__ import print_function

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUI_PATH = ROOT / "tools" / "failure_replay_gui.py"

spec = importlib.util.spec_from_file_location(
    "failure_replay_gui_for_test",
    str(GUI_PATH),
)
gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gui)


def test_route_reference_line_uses_waypoints_only(tmp_path):
    xml = tmp_path / "routes.xml"
    xml.write_text(
        """<routes>
          <route id="17" town="Town12">
            <waypoints>
              <position x="0" y="0" z="0"/>
              <position x="3" y="4" z="0"/>
              <position x="6" y="4" z="0"/>
            </waypoints>
            <scenarios>
              <scenario name="DoNotUseMe" type="Dummy">
                <trigger_point x="999" y="999" z="0"/>
              </scenario>
            </scenarios>
          </route>
        </routes>""",
        encoding="utf-8",
    )

    points = gui.load_route_reference_line(xml, "17")
    assert len(points) == 3
    assert [(p["x"], p["y"]) for p in points] == [
        (0.0, 0.0),
        (3.0, 4.0),
        (6.0, 4.0),
    ]

    s = gui.cumulative_xy_distance(points)
    assert s == [0.0, 5.0, 8.0]

    idx = gui.nearest_xy_index(points, 3.1, 3.9)
    assert idx == 1


def test_missing_route_returns_empty_reference(tmp_path):
    xml = tmp_path / "routes.xml"
    xml.write_text(
        """<routes>
          <route id="1" town="Town01">
            <waypoints>
              <position x="0" y="0" z="0"/>
              <position x="1" y="0" z="0"/>
            </waypoints>
          </route>
        </routes>""",
        encoding="utf-8",
    )
    assert gui.load_route_reference_line(xml, "999") == []
