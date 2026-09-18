from __future__ import division

import json
import math
import os
import tempfile

import numpy as np

from b2d_collector.failure_replay.manual_plan import (
    build_dense_trajectory,
    cubic_bspline_dense,
    generate_reference_anchors,
    parse_route_reference_line,
    pchip_interpolate,
    polyline_cumulative_s,
)


def test_anchor_count_depends_on_segment_length():
    ref = np.array([[0.0, 0.0], [40.0, 0.0]])
    anchors = generate_reference_anchors(ref, 4.0)
    assert len(anchors) == 11
    assert np.allclose(anchors[0], [0.0, 0.0])
    assert np.allclose(anchors[-1], [40.0, 0.0])


def test_clamped_bspline_keeps_endpoints_and_dense_spacing():
    ctrl = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [10.0, 3.0],
        [15.0, 3.0],
        [20.0, 0.0],
    ])
    dense = cubic_bspline_dense(ctrl, 0.2)
    assert np.allclose(dense[0], ctrl[0], atol=1e-8)
    assert np.allclose(dense[-1], ctrl[-1], atol=1e-8)
    s = polyline_cumulative_s(dense)
    assert len(dense) > 50
    assert s[-1] > 20.0


def test_pchip_monotone_profile_has_no_large_overshoot():
    s = np.array([0.0, 10.0, 20.0, 40.0])
    v = np.array([4.0, 6.0, 8.0, 10.0])
    q = np.linspace(0.0, 40.0, 401)
    out = pchip_interpolate(s, v, q)
    assert np.min(out) >= 4.0 - 1e-9
    assert np.max(out) <= 10.0 + 1e-9
    assert np.all(np.diff(out) >= -1e-8)


def test_dense_trajectory_metrics():
    path = np.column_stack([np.linspace(0.0, 10.0, 51), np.zeros(51)])
    speed = np.array([[0.0, 5.0], [10.0, 5.0]])
    traj = build_dense_trajectory(path, speed)
    assert abs(traj["metrics"]["path_length_m"] - 10.0) < 1e-6
    assert abs(traj["metrics"]["estimated_duration_s"] - 2.0) < 1e-6
    assert np.max(np.abs(traj["curvature"])) < 1e-8


def test_route_parser_only_uses_waypoints_positions():
    xml = """<routes>
      <route id="2091" town="Town12">
        <waypoints>
          <position x="0" y="0" z="0"/>
          <position x="10" y="0" z="0"/>
          <position x="20" y="2" z="0"/>
        </waypoints>
        <scenarios>
          <scenario type="Dummy">
            <trigger_point x="999" y="999" z="0"/>
          </scenario>
        </scenarios>
      </route>
    </routes>"""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "routes.xml")
        with open(path, "w") as f:
            f.write(xml)
        pts = parse_route_reference_line(path, "2091")
    assert pts.shape == (3, 2)
    assert np.max(pts[:, 0]) == 20.0
    assert np.max(pts[:, 1]) == 2.0


def test_free_anchors_are_sparse_and_endpoints_are_exact():
    from b2d_collector.failure_replay.manual_plan import generate_free_anchors
    anchors = generate_free_anchors([0.0, 0.0], [31.0, 8.0], anchor_spacing_m=6.0)
    assert 4 <= len(anchors) <= 12
    assert np.allclose(anchors[0], [0.0, 0.0])
    assert np.allclose(anchors[-1], [31.0, 8.0])


def test_v2_plan_is_spatial_only():
    from b2d_collector.failure_replay.manual_plan import make_plan_document_v2
    path = np.column_stack([np.linspace(0.0, 10.0, 51), np.zeros(51)])
    speed = np.array([[0.0, 3.0], [5.0, 2.0], [10.0, 5.0]])
    dense = build_dense_trajectory(path, speed)
    doc = make_plan_document_v2(
        "case_x", "/tmp/probe", "", "1711",
        {"sample_index": 10, "sim_time_s": 1.7, "x": 0.0, "y": 0.0, "speed_mps": 2.0},
        {"x": 10.0, "y": 0.0, "radius_m": 1.5, "confirm_frames": 3},
        None,
        np.array([[0.0, 0.0], [3.0, 0.0], [7.0, 0.0], [10.0, 0.0]]),
        speed,
        dense,
        6.0,
        0.2,
    )
    assert doc["schema"] == "b2d-manual-plan-v2"
    assert doc["settings"]["time_parameterization"] == "none-spatial-only"
    assert "record_end_time" not in doc
    assert doc["end"]["x"] == 10.0
