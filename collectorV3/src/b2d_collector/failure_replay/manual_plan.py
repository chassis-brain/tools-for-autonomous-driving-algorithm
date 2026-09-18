from __future__ import division

import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np


SCHEMA = "b2d-manual-plan-v1"


def polyline_cumulative_s(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros((0,), dtype=float)
    if len(pts) == 1:
        return np.zeros((1,), dtype=float)
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def point_at_s(points, s_query):
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        raise ValueError("empty polyline")
    if len(pts) == 1:
        return pts[0, :2].copy()
    cs = polyline_cumulative_s(pts)
    s = float(np.clip(s_query, cs[0], cs[-1]))
    idx = int(np.searchsorted(cs, s, side="right") - 1)
    idx = min(max(idx, 0), len(pts) - 2)
    span = cs[idx + 1] - cs[idx]
    if span <= 1e-9:
        return pts[idx, :2].copy()
    a = (s - cs[idx]) / span
    return (1.0 - a) * pts[idx, :2] + a * pts[idx + 1, :2]


def project_point_to_polyline(point_xy, points):
    p = np.asarray(point_xy, dtype=float)[:2]
    pts = np.asarray(points, dtype=float)[:, :2]
    if len(pts) == 0:
        raise ValueError("empty polyline")
    if len(pts) == 1:
        d = float(np.linalg.norm(p - pts[0]))
        return {
            "point": pts[0].copy(),
            "s": 0.0,
            "distance": d,
            "segment_index": 0,
            "segment_t": 0.0,
        }

    a = pts[:-1]
    b = pts[1:]
    ab = b - a
    denom = np.sum(ab * ab, axis=1)
    denom_safe = np.where(denom > 1e-12, denom, 1.0)
    t = np.sum((p - a) * ab, axis=1) / denom_safe
    t = np.clip(t, 0.0, 1.0)
    proj = a + ab * t[:, None]
    d2 = np.sum((proj - p) ** 2, axis=1)
    idx = int(np.argmin(d2))
    cs = polyline_cumulative_s(pts)
    seg_len = math.sqrt(float(denom[idx])) if denom[idx] > 0 else 0.0
    s = float(cs[idx] + t[idx] * seg_len)
    return {
        "point": proj[idx].copy(),
        "s": s,
        "distance": math.sqrt(float(d2[idx])),
        "segment_index": idx,
        "segment_t": float(t[idx]),
    }


def slice_polyline(points, s0, s1):
    pts = np.asarray(points, dtype=float)[:, :2]
    if len(pts) < 2:
        return pts.copy()
    cs = polyline_cumulative_s(pts)
    lo = float(np.clip(min(s0, s1), cs[0], cs[-1]))
    hi = float(np.clip(max(s0, s1), cs[0], cs[-1]))
    out = [point_at_s(pts, lo)]
    for i in range(1, len(pts) - 1):
        if lo < cs[i] < hi:
            out.append(pts[i].copy())
    out.append(point_at_s(pts, hi))
    arr = np.asarray(out, dtype=float)
    if s1 < s0:
        arr = arr[::-1]
    return arr


def sample_polyline(points, count):
    pts = np.asarray(points, dtype=float)[:, :2]
    if count < 2:
        raise ValueError("count must be >= 2")
    cs = polyline_cumulative_s(pts)
    if len(cs) == 0 or cs[-1] <= 1e-9:
        return np.repeat(pts[:1], count, axis=0)
    s = np.linspace(0.0, cs[-1], int(count))
    return np.vstack([point_at_s(pts, q) for q in s])


def generate_reference_anchors(reference_segment, anchor_spacing_m=4.0):
    pts = np.asarray(reference_segment, dtype=float)[:, :2]
    if len(pts) < 2:
        raise ValueError("reference segment needs at least 2 points")
    length = float(polyline_cumulative_s(pts)[-1])
    if length <= 1e-9:
        return np.vstack([pts[0], pts[-1]])
    spacing = max(float(anchor_spacing_m), 0.2)
    count = int(math.ceil(length / spacing)) + 1
    count = max(2, count)
    return sample_polyline(pts, count)


def generate_free_anchors(start_xy, end_xy=None, anchor_spacing_m=6.0, **kwargs):
    """Generate sparse editable anchors between Handoff and the free spatial End.

    This is deliberately a *spatial* initializer: no timestamps and no route/E2E
    snapping are introduced.  The first and last anchors are exactly the supplied
    endpoints and intermediate anchors are evenly spaced on the straight chord;
    the GUI can then move those intermediate anchors freely before the B-spline
    is generated.

    Compatibility forms accepted by the integrated/manual planner UIs:
      generate_free_anchors(start_xy, end_xy, spacing)
      generate_free_anchors([start_xy, end_xy], spacing)
      generate_free_anchors([start_xy, ..., end_xy], anchor_spacing_m=spacing)

    Optional keyword aliases ``spacing_m`` and ``spacing`` are accepted so old
    hotfix panels do not fail when mixed with the cumulative runtime patch.
    """
    if "spacing_m" in kwargs and kwargs["spacing_m"] is not None:
        anchor_spacing_m = kwargs["spacing_m"]
    if "spacing" in kwargs and kwargs["spacing"] is not None:
        anchor_spacing_m = kwargs["spacing"]

    # Older panel variant: generate_free_anchors(points, spacing).
    arr0 = np.asarray(start_xy, dtype=float)
    if end_xy is not None and np.isscalar(end_xy) and arr0.ndim >= 2 and arr0.shape[0] >= 2:
        anchor_spacing_m = float(end_xy)
        start = arr0[0, :2]
        end = arr0[-1, :2]
    elif end_xy is None:
        if arr0.ndim < 2 or arr0.shape[0] < 2:
            raise ValueError("generate_free_anchors needs start and end points")
        start = arr0[0, :2]
        end = arr0[-1, :2]
    else:
        start = arr0.reshape(-1)[:2]
        end = np.asarray(end_xy, dtype=float).reshape(-1)[:2]

    if start.size < 2 or end.size < 2:
        raise ValueError("start/end must contain x and y")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise ValueError("start/end must be finite")

    length = float(np.linalg.norm(end - start))
    spacing = max(float(anchor_spacing_m), 0.2)
    if length <= 1e-9:
        return np.vstack([start.copy(), end.copy()])

    # ceil keeps segment spacing <= requested spacing.  Endpoints are fixed; only
    # interior anchors are intended to be draggable by the GUI.
    segments = max(1, int(math.ceil(length / spacing)))
    count = segments + 1
    anchors = np.linspace(start, end, count)
    anchors[0] = start
    anchors[-1] = end
    return anchors


def _open_uniform_knots(n_ctrl, degree):
    p = int(degree)
    n = int(n_ctrl)
    if n <= p:
        raise ValueError("n_ctrl must be greater than degree")
    internal_count = n - p - 1
    if internal_count > 0:
        internal = np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
        return np.concatenate((np.zeros(p + 1), internal, np.ones(p + 1)))
    return np.concatenate((np.zeros(p + 1), np.ones(p + 1)))


def _de_boor(ctrl, degree, knots, u):
    ctrl = np.asarray(ctrl, dtype=float)
    p = int(degree)
    n_ctrl = len(ctrl)
    if u >= 1.0:
        return ctrl[-1].copy()
    if u <= 0.0:
        return ctrl[0].copy()

    k = int(np.searchsorted(knots, u, side="right") - 1)
    k = min(max(k, p), n_ctrl - 1)
    d = [ctrl[j + k - p].copy() for j in range(0, p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            i = j + k - p
            den = knots[i + p + 1 - r] - knots[i]
            alpha = 0.0 if abs(den) < 1e-12 else (u - knots[i]) / den
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[p]


def cubic_bspline_dense(control_points, sample_spacing_m=0.2):
    ctrl = np.asarray(control_points, dtype=float)[:, :2]
    if len(ctrl) < 2:
        raise ValueError("need at least two control points")
    if len(ctrl) == 2:
        length = float(np.linalg.norm(ctrl[1] - ctrl[0]))
        count = max(2, int(math.ceil(length / max(sample_spacing_m, 0.02))) + 1)
        return np.linspace(ctrl[0], ctrl[1], count)

    degree = min(3, len(ctrl) - 1)
    knots = _open_uniform_knots(len(ctrl), degree)
    rough_len = float(np.sum(np.linalg.norm(np.diff(ctrl, axis=0), axis=1)))
    pre_count = max(300, int(math.ceil(max(rough_len, 1.0) / 0.03)) + 1)
    u_grid = np.linspace(0.0, 1.0, pre_count)
    raw = np.vstack([_de_boor(ctrl, degree, knots, float(u)) for u in u_grid])

    cs = polyline_cumulative_s(raw)
    total = float(cs[-1])
    if total <= 1e-9:
        return np.repeat(raw[:1], 2, axis=0)
    spacing = max(float(sample_spacing_m), 0.02)
    count = max(2, int(math.ceil(total / spacing)) + 1)
    s_out = np.linspace(0.0, total, count)
    x = np.interp(s_out, cs, raw[:, 0])
    y = np.interp(s_out, cs, raw[:, 1])
    out = np.column_stack([x, y])
    out[0] = ctrl[0]
    out[-1] = ctrl[-1]
    return out


def path_geometry(path_xy):
    pts = np.asarray(path_xy, dtype=float)[:, :2]
    if len(pts) < 2:
        raise ValueError("path needs at least 2 points")
    s = polyline_cumulative_s(pts)
    if s[-1] <= 1e-9:
        yaw = np.zeros(len(pts), dtype=float)
        curvature = np.zeros(len(pts), dtype=float)
        return s, yaw, curvature

    x = pts[:, 0]
    y = pts[:, 1]
    edge_order = 2 if len(pts) >= 3 else 1
    dx = np.gradient(x, s, edge_order=edge_order)
    dy = np.gradient(y, s, edge_order=edge_order)
    ddx = np.gradient(dx, s, edge_order=edge_order)
    ddy = np.gradient(dy, s, edge_order=edge_order)
    yaw = np.arctan2(dy, dx)
    denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    curvature = (dx * ddy - dy * ddx) / denom
    return s, yaw, curvature


def _pchip_slopes(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least two PCHIP points")
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("PCHIP x must be strictly increasing")
    delta = np.diff(y) / h
    m = np.zeros(n, dtype=float)

    if n == 2:
        m[:] = delta[0]
        return m

    for k in range(1, n - 1):
        if delta[k - 1] == 0.0 or delta[k] == 0.0 or np.sign(delta[k - 1]) != np.sign(delta[k]):
            m[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    m0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(m0) != np.sign(delta[0]):
        m0 = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(m0) > abs(3.0 * delta[0]):
        m0 = 3.0 * delta[0]
    m[0] = m0

    mn = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(mn) != np.sign(delta[-1]):
        mn = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(mn) > abs(3.0 * delta[-1]):
        mn = 3.0 * delta[-1]
    m[-1] = mn
    return m


def pchip_interpolate(x_ctrl, y_ctrl, x_query):
    x = np.asarray(x_ctrl, dtype=float)
    y = np.asarray(y_ctrl, dtype=float)
    q = np.asarray(x_query, dtype=float)
    if len(x) != len(y):
        raise ValueError("x/y control lengths differ")
    if len(x) < 2:
        raise ValueError("need at least two speed control points")
    m = _pchip_slopes(x, y)

    out = np.empty_like(q, dtype=float)
    for idx, xv in np.ndenumerate(q):
        if xv <= x[0]:
            out[idx] = y[0]
            continue
        if xv >= x[-1]:
            out[idx] = y[-1]
            continue
        i = int(np.searchsorted(x, xv, side="right") - 1)
        h = x[i + 1] - x[i]
        t = (xv - x[i]) / h
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        out[idx] = (
            h00 * y[i]
            + h10 * h * m[i]
            + h01 * y[i + 1]
            + h11 * h * m[i + 1]
        )
    return out


def default_speed_control_points(path_length_m, default_speed_mps=5.0, count=6):
    length = max(float(path_length_m), 0.01)
    n = max(2, int(count))
    s = np.linspace(0.0, length, n)
    v = np.full(n, max(0.0, float(default_speed_mps)), dtype=float)
    return np.column_stack([s, v])


def build_dense_trajectory(path_xy, speed_control_points):
    pts = np.asarray(path_xy, dtype=float)[:, :2]
    speed_ctrl = np.asarray(speed_control_points, dtype=float)
    s, yaw, curvature = path_geometry(pts)
    target_speed = pchip_interpolate(speed_ctrl[:, 0], speed_ctrl[:, 1], s)
    target_speed = np.maximum(target_speed, 0.0)

    duration = 0.0
    if len(s) > 1:
        ds = np.diff(s)
        mid_v = 0.5 * (target_speed[:-1] + target_speed[1:])
        if np.any(mid_v <= 1e-4):
            duration = float("inf")
        else:
            duration = float(np.sum(ds / mid_v))

    return {
        "s": s,
        "x": pts[:, 0],
        "y": pts[:, 1],
        "yaw": yaw,
        "curvature": curvature,
        "target_speed": target_speed,
        "metrics": {
            "path_length_m": float(s[-1]) if len(s) else 0.0,
            "max_abs_curvature_1pm": float(np.max(np.abs(curvature))) if len(curvature) else 0.0,
            "estimated_duration_s": duration,
            "min_target_speed_mps": float(np.min(target_speed)) if len(target_speed) else 0.0,
            "max_target_speed_mps": float(np.max(target_speed)) if len(target_speed) else 0.0,
        },
    }


def parse_route_reference_line(route_xml, route_id):
    root = ET.parse(str(route_xml)).getroot()
    wanted = str(route_id)
    route = None
    for elem in root.iter():
        if elem.tag.split("}")[-1] == "route" and str(elem.attrib.get("id", "")) == wanted:
            route = elem
            break
    if route is None:
        raise KeyError("route id %s not found in %s" % (wanted, route_xml))

    waypoints = None
    for child in list(route):
        if child.tag.split("}")[-1] == "waypoints":
            waypoints = child
            break
    if waypoints is None:
        raise ValueError("route %s has no <waypoints>" % wanted)

    pts = []
    for p in list(waypoints):
        if p.tag.split("}")[-1] != "position":
            continue
        pts.append([float(p.attrib["x"]), float(p.attrib["y"])])
    if len(pts) < 2:
        raise ValueError("route %s has fewer than 2 waypoint positions" % wanted)
    return np.asarray(pts, dtype=float)


def _vec2_from_obj(obj, default=(0.0, 0.0)):
    if not isinstance(obj, dict):
        return float(default[0]), float(default[1])
    return float(obj.get("x", default[0])), float(obj.get("y", default[1]))


def extract_frame_xy(frame):
    ego = frame.get("ego", {})
    state = ego.get("state", {}) if isinstance(ego, dict) else {}

    for holder in (state, ego, frame):
        if not isinstance(holder, dict):
            continue
        for key in ("location", "position", "transform"):
            val = holder.get(key)
            if isinstance(val, dict):
                if key == "transform" and isinstance(val.get("location"), dict):
                    return _vec2_from_obj(val["location"])
                if "x" in val and "y" in val:
                    return _vec2_from_obj(val)

    transform = state.get("transform") if isinstance(state, dict) else None
    if isinstance(transform, dict) and isinstance(transform.get("location"), dict):
        return _vec2_from_obj(transform["location"])
    raise KeyError("could not find ego x/y in frame")


def extract_frame_speed(frame):
    ego = frame.get("ego", {})
    state = ego.get("state", {}) if isinstance(ego, dict) else {}
    vel = state.get("velocity", {}) if isinstance(state, dict) else {}
    if isinstance(vel, dict) and "x" in vel and "y" in vel:
        vx = float(vel.get("x", 0.0))
        vy = float(vel.get("y", 0.0))
        vz = float(vel.get("z", 0.0))
        return math.sqrt(vx * vx + vy * vy + vz * vz)
    for holder in (state, ego, frame):
        if isinstance(holder, dict):
            for key in ("speed", "speed_mps"):
                if key in holder:
                    return float(holder[key])
    return 0.0


def extract_frame_time(frame):
    world = frame.get("world", {})
    if isinstance(world, dict):
        for key in ("elapsed_seconds", "sim_time_s", "time"):
            if key in world:
                return float(world[key])
    for key in ("sim_time_s", "elapsed_seconds", "time"):
        if key in frame:
            return float(frame[key])
    raise KeyError("could not find simulation time in frame")


def load_probe_frames(probe_dir):
    path = os.path.join(str(probe_dir), "frames.jsonl")
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("no frames in %s" % path)
    return rows


def nearest_frame_index_by_time(frames, sim_time_s):
    times = np.asarray([extract_frame_time(f) for f in frames], dtype=float)
    return int(np.argmin(np.abs(times - float(sim_time_s))))


def case_time(case_obj, key):
    obj = case_obj.get(key, {})
    if not isinstance(obj, dict):
        raise KeyError(key)
    if "sim_time_s" in obj:
        return float(obj["sim_time_s"])
    if "world_time_s" in obj:
        return float(obj["world_time_s"])
    raise KeyError("%s.sim_time_s" % key)


def make_plan_document(case_id, source_run, route_xml, route_id, handoff, planning_end,
                       reference_line, path_control_points, speed_control_points,
                       dense_trajectory, anchor_spacing_m, sample_spacing_m):
    metrics = dense_trajectory["metrics"]
    return {
        "schema": SCHEMA,
        "status": "edited_v1",
        "case_id": str(case_id),
        "source_run": os.path.abspath(str(source_run)),
        "route_xml": os.path.abspath(str(route_xml)),
        "route_id": str(route_id),
        "planning_start": {
            "source": "handoff",
            "sim_time_s": float(handoff["sim_time_s"]),
            "x": float(handoff["x"]),
            "y": float(handoff["y"]),
        },
        "planning_end": {
            "x": float(planning_end["x"]),
            "y": float(planning_end["y"]),
            "snap_to_reference": bool(planning_end.get("snap_to_reference", True)),
            "reference_s_m": float(planning_end.get("reference_s_m", 0.0)),
            "from_handoff_m": float(planning_end.get("from_handoff_m", 0.0)),
        },
        "settings": {
            "anchor_spacing_m": float(anchor_spacing_m),
            "dense_sample_spacing_m": float(sample_spacing_m),
            "path_interpolation": "clamped-cubic-b-spline",
            "speed_interpolation": "pchip",
        },
        "reference_line": np.asarray(reference_line, dtype=float).tolist(),
        "path_control_points": np.asarray(path_control_points, dtype=float).tolist(),
        "speed_control_points": [
            {"s": float(row[0]), "v": float(row[1])}
            for row in np.asarray(speed_control_points, dtype=float)
        ],
        "metrics": metrics,
        "generated": {
            "samples": int(len(dense_trajectory["s"])),
            "trajectory_file": None,
        },
    }



def _manual_plan_v2_point(obj, default=None):
    """Return a small JSON-safe spatial point dict from common GUI/case forms."""
    if obj is None:
        obj = default
    if obj is None:
        return None

    if isinstance(obj, dict):
        # Some panels pass a nested case point object directly; others use a
        # location/position holder.
        holder = obj
        for key in ("location", "position", "point"):
            value = holder.get(key) if isinstance(holder, dict) else None
            if isinstance(value, dict) and value.get("x") is not None and value.get("y") is not None:
                holder = value
                break
        if holder.get("x") is None or holder.get("y") is None:
            return None
        out = {"x": float(holder["x"]), "y": float(holder["y"])}
        for key in ("sim_time_s", "world_time_s", "sample_index", "speed_mps", "yaw"):
            if obj.get(key) is not None:
                value = obj[key]
                if key == "sample_index":
                    out[key] = int(value)
                else:
                    out[key] = float(value)
        return out

    arr = np.asarray(obj, dtype=float).reshape(-1)
    if arr.size < 2:
        return None
    return {"x": float(arr[0]), "y": float(arr[1])}


def _manual_plan_v2_pick(mapping, names, default=None):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def make_plan_document_v2(*args, **kwargs):
    """Build the three-point/manual-PID plan document used by the integrated GUI.

    The ManualPlanPanel went through a few hotfix iterations.  Some revisions
    call this helper with keyword arguments, while older ones kept the positional
    ordering of ``make_plan_document``.  This adapter intentionally accepts both
    forms and normalizes them to one runtime-friendly document.

    Semantics are the final three-point design:
      * planning start == the E2E-aligned Handoff (time + XY);
      * plan end == the case's free spatial End (XY only; no end timestamp);
      * path and speed are P(s) + v(s), with no planned time axis.
    """
    # Positional compatibility with the old v1 constructor:
    #   case_id, source_run, route_xml, route_id, handoff, end,
    #   reference_line, path_control_points, speed_control_points,
    #   dense_trajectory, anchor_spacing_m, sample_spacing_m
    positional_names = (
        "case_id", "source_run", "route_xml", "route_id", "handoff", "end",
        "reference_line", "path_control_points", "speed_control_points",
        "dense_trajectory", "anchor_spacing_m", "sample_spacing_m",
    )
    values = dict(kwargs)
    for idx, value in enumerate(args):
        if idx < len(positional_names):
            values.setdefault(positional_names[idx], value)
        else:
            # Extra positional values from an experimental panel are retained as
            # metadata rather than causing a TypeError at save time.
            values.setdefault("_extra_positional", []).append(value)

    case_obj = _manual_plan_v2_pick(values, ("case", "case_obj", "intervention", "case_document"))
    if isinstance(values.get("case_id"), dict) and case_obj is None:
        case_obj = values.get("case_id")
        values["case_id"] = case_obj.get("case_id")

    case_obj = case_obj if isinstance(case_obj, dict) else {}

    case_id = _manual_plan_v2_pick(values, ("case_id", "id"), case_obj.get("case_id", "manual_case"))
    source_run = _manual_plan_v2_pick(
        values, ("source_run", "probe_run", "probe_dir", "run_dir"), case_obj.get("source_run")
    )
    route_xml = _manual_plan_v2_pick(values, ("route_xml", "routes_xml"), case_obj.get("route_xml"))
    route_id = _manual_plan_v2_pick(values, ("route_id", "route"), case_obj.get("route_id"))

    handoff = _manual_plan_v2_pick(
        values, ("handoff", "planning_start", "start", "start_point"), case_obj.get("handoff")
    )
    end_obj = _manual_plan_v2_pick(
        values,
        ("end", "end_point", "spatial_end", "record_end", "planning_end"),
        case_obj.get("end"),
    )
    handoff_point = _manual_plan_v2_point(handoff)
    end_point = _manual_plan_v2_point(end_obj)

    path_control_points = _manual_plan_v2_pick(
        values,
        ("path_control_points", "anchors", "path_anchors", "control_points"),
    )
    speed_control_points = _manual_plan_v2_pick(
        values,
        ("speed_control_points", "speed_points", "speed_ctrl", "speed_profile"),
    )
    dense_trajectory = _manual_plan_v2_pick(
        values, ("dense_trajectory", "trajectory", "dense", "traj")
    )
    reference_line = _manual_plan_v2_pick(
        values, ("reference_line", "reference", "route_reference")
    )

    # If the panel hands us only the editable anchors and speed points, derive
    # the dense P(s)+v(s) trajectory here.  This also guarantees that the JSON
    # and NPZ saved by save_plan() describe exactly the same path.
    if path_control_points is None and isinstance(dense_trajectory, dict):
        if dense_trajectory.get("x") is not None and dense_trajectory.get("y") is not None:
            path_control_points = np.column_stack(
                [np.asarray(dense_trajectory["x"], dtype=float), np.asarray(dense_trajectory["y"], dtype=float)]
            )

    if path_control_points is None:
        if handoff_point is None or end_point is None:
            raise ValueError("make_plan_document_v2 needs path_control_points or Handoff+End")
        path_control_points = generate_free_anchors(
            [handoff_point["x"], handoff_point["y"]],
            [end_point["x"], end_point["y"]],
            anchor_spacing_m=float(_manual_plan_v2_pick(values, ("anchor_spacing_m", "anchor_spacing"), 6.0)),
        )

    path_control_points = np.asarray(path_control_points, dtype=float)
    if path_control_points.ndim != 2 or path_control_points.shape[0] < 2 or path_control_points.shape[1] < 2:
        raise ValueError("path_control_points must be Nx2 with N>=2")
    path_control_points = path_control_points[:, :2]

    # Enforce the final semantics at serialization time.  Whatever a stale panel
    # calls its endpoint, the saved executable plan starts exactly at Handoff and
    # ends exactly at the case spatial End when those are available.
    if handoff_point is not None:
        path_control_points[0] = [handoff_point["x"], handoff_point["y"]]
    if end_point is not None:
        path_control_points[-1] = [end_point["x"], end_point["y"]]

    sample_spacing_m = float(_manual_plan_v2_pick(
        values, ("sample_spacing_m", "dense_sample_spacing_m", "path_sample_spacing_m"), 0.2
    ))
    anchor_spacing_m = float(_manual_plan_v2_pick(
        values, ("anchor_spacing_m", "anchor_spacing"), 6.0
    ))

    if dense_trajectory is None:
        dense_path = cubic_bspline_dense(path_control_points, sample_spacing_m)
        if speed_control_points is None:
            speed_control_points = default_speed_control_points(
                float(polyline_cumulative_s(dense_path)[-1]),
                default_speed_mps=float(_manual_plan_v2_pick(values, ("default_speed_mps", "speed_mps"), 5.0)),
            )
        dense_trajectory = build_dense_trajectory(dense_path, speed_control_points)
    elif not isinstance(dense_trajectory, dict):
        dense_path = np.asarray(dense_trajectory, dtype=float)
        if dense_path.ndim != 2 or dense_path.shape[1] < 2:
            raise ValueError("dense_trajectory array must be Nx2")
        if speed_control_points is None:
            speed_control_points = default_speed_control_points(
                float(polyline_cumulative_s(dense_path[:, :2])[-1]), 5.0
            )
        dense_trajectory = build_dense_trajectory(dense_path[:, :2], speed_control_points)

    # Some GUI revisions compute dense_trajectory before the user drags the last
    # anchor.  Rebuild if its endpoints do not match the authoritative anchors.
    try:
        dx = np.asarray(dense_trajectory["x"], dtype=float)
        dy = np.asarray(dense_trajectory["y"], dtype=float)
        dense_endpoints_ok = (
            len(dx) >= 2
            and math.hypot(dx[0] - path_control_points[0, 0], dy[0] - path_control_points[0, 1]) <= 1e-4
            and math.hypot(dx[-1] - path_control_points[-1, 0], dy[-1] - path_control_points[-1, 1]) <= 1e-4
        )
    except Exception:
        dense_endpoints_ok = False
    if not dense_endpoints_ok:
        dense_path = cubic_bspline_dense(path_control_points, sample_spacing_m)
        if speed_control_points is None:
            speed_control_points = default_speed_control_points(float(polyline_cumulative_s(dense_path)[-1]), 5.0)
        dense_trajectory = build_dense_trajectory(dense_path, speed_control_points)

    if speed_control_points is None:
        # Fall back to a compact representation sampled from the dense speed
        # field, so runtime regeneration remains possible even without the NPZ.
        s_arr = np.asarray(dense_trajectory["s"], dtype=float)
        v_arr = np.asarray(dense_trajectory["target_speed"], dtype=float)
        count = min(6, max(2, len(s_arr)))
        sample_s = np.linspace(float(s_arr[0]), float(s_arr[-1]), count)
        sample_v = np.interp(sample_s, s_arr, v_arr)
        speed_control_points = np.column_stack([sample_s, sample_v])

    speed_rows = []
    for row in np.asarray(speed_control_points, dtype=float):
        speed_rows.append({"s": float(row[0]), "v": float(row[1])})

    metrics = dict(dense_trajectory.get("metrics", {}))
    if not metrics:
        metrics = build_dense_trajectory(
            np.column_stack([dense_trajectory["x"], dense_trajectory["y"]]),
            np.asarray([[row["s"], row["v"]] for row in speed_rows], dtype=float),
        )["metrics"]

    end_meta = dict(end_point or {})
    if isinstance(end_obj, dict):
        if end_obj.get("radius_m") is not None:
            end_meta["radius_m"] = float(end_obj["radius_m"])
        if end_obj.get("confirm_frames") is not None:
            end_meta["confirm_frames"] = int(end_obj["confirm_frames"])
    if "radius_m" not in end_meta:
        end_meta["radius_m"] = float(_manual_plan_v2_pick(values, ("end_radius_m", "radius_m"), 1.5))
    if "confirm_frames" not in end_meta:
        end_meta["confirm_frames"] = int(_manual_plan_v2_pick(values, ("end_confirm_frames", "confirm_frames"), 3))

    doc = {
        "schema": "b2d-manual-plan-v2",
        "status": "edited_v2",
        "case_id": str(case_id),
        "route_id": None if route_id is None else str(route_id),
        "planning_start": dict(handoff_point or {}),
        "end": end_meta,
        "settings": {
            "anchor_spacing_m": anchor_spacing_m,
            "dense_sample_spacing_m": sample_spacing_m,
            "path_interpolation": "clamped-cubic-b-spline",
            "speed_interpolation": "pchip",
            "time_parameterization": "none-spatial-only",
        },
        "path_control_points": path_control_points.tolist(),
        "speed_control_points": speed_rows,
        "metrics": metrics,
        "generated": {
            "samples": int(len(np.asarray(dense_trajectory["s"]))),
            "trajectory_file": None,
        },
    }
    if source_run:
        doc["source_run"] = os.path.abspath(str(source_run))
    if route_xml:
        doc["route_xml"] = os.path.abspath(str(route_xml))
    if reference_line is not None:
        ref = np.asarray(reference_line, dtype=float)
        if ref.ndim == 2 and ref.shape[1] >= 2:
            doc["reference_line"] = ref[:, :2].tolist()

    # A few panel versions pass the complete case path for traceability.
    case_path = _manual_plan_v2_pick(values, ("case_path", "case_json", "case_file"))
    if case_path:
        doc["case_file"] = os.path.abspath(str(case_path))

    # Runtime loader intentionally ignores this embedded copy when an NPZ exists,
    # but keeping it here makes the plan self-describing and easier to inspect.
    doc["dense_trajectory"] = {
        key: np.asarray(dense_trajectory[key], dtype=float).tolist()
        for key in ("s", "x", "y", "yaw", "curvature", "target_speed")
    }
    return doc

def save_plan(plan_path, plan_doc, dense_trajectory, npz_path=None):
    plan_path = os.path.abspath(str(plan_path))
    parent = os.path.dirname(plan_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if npz_path is None:
        stem, _ = os.path.splitext(plan_path)
        npz_path = stem + ".trajectory.npz"
    npz_path = os.path.abspath(str(npz_path))
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)

    np.savez_compressed(
        npz_path,
        s=np.asarray(dense_trajectory["s"], dtype=float),
        x=np.asarray(dense_trajectory["x"], dtype=float),
        y=np.asarray(dense_trajectory["y"], dtype=float),
        yaw=np.asarray(dense_trajectory["yaw"], dtype=float),
        curvature=np.asarray(dense_trajectory["curvature"], dtype=float),
        target_speed=np.asarray(dense_trajectory["target_speed"], dtype=float),
    )

    doc = dict(plan_doc)
    doc["generated"] = dict(doc.get("generated", {}))
    doc["generated"]["samples"] = int(len(dense_trajectory["s"]))
    doc["generated"]["trajectory_file"] = os.path.relpath(npz_path, os.path.dirname(plan_path))

    with open(plan_path, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
        f.write("\n")
    return plan_path, npz_path
