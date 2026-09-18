#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import json
import math
import os
import sys
import traceback

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:
    import Tkinter as tk
    import ttk
    import tkFileDialog as filedialog
    import tkMessageBox as messagebox

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(PROJECT_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from b2d_collector.failure_replay.manual_plan import (
    build_dense_trajectory,
    case_time,
    cubic_bspline_dense,
    default_speed_control_points,
    extract_frame_speed,
    extract_frame_time,
    extract_frame_xy,
    generate_reference_anchors,
    load_probe_frames,
    make_plan_document,
    nearest_frame_index_by_time,
    parse_route_reference_line,
    point_at_s,
    polyline_cumulative_s,
    project_point_to_polyline,
    save_plan,
    slice_polyline,
)


def _read_json(path, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def _guess_workbench_config():
    return os.path.expanduser("~/.failure_replay_workbench.json")


def _cfg_lookup(cfg, names):
    if not isinstance(cfg, dict):
        return ""
    for name in names:
        if name in cfg and cfg[name]:
            return str(cfg[name])
    for value in cfg.values():
        if isinstance(value, dict):
            found = _cfg_lookup(value, names)
            if found:
                return found
    return ""


def _infer_route_id(case_id):
    text = str(case_id)
    parts = text.replace("-", "_").split("_")
    for part in parts:
        if part.isdigit() and len(part) >= 3:
            return part
    return ""


class ManualPlanEditor(tk.Tk):
    def __init__(self, args):
        tk.Tk.__init__(self)
        self.title("Failure Replay - Manual Planning V1")
        self.geometry("1480x920")
        self.minsize(1180, 760)

        self.project_root = os.path.abspath(args.project_root or PROJECT_ROOT)
        wb = _read_json(_guess_workbench_config(), {}) or {}

        default_route_xml = (
            args.route_xml
            or _cfg_lookup(wb, ["route_xml", "routes_xml", "bench2drive_route_xml"])
            or os.path.join(
                os.environ.get("B2D_ROOT", ""),
                "leaderboard", "data", "bench2drive220.xml"
            )
        )

        self.var_case = tk.StringVar(value=args.case or "")
        self.var_probe = tk.StringVar(value=args.probe or "")
        self.var_route_xml = tk.StringVar(value=default_route_xml if os.path.exists(default_route_xml) else "")
        self.var_route_id = tk.StringVar(value=args.route_id or "")
        self.var_snap = tk.BooleanVar(value=True)
        self.var_anchor_spacing = tk.DoubleVar(value=4.0)
        self.var_dense_spacing = tk.DoubleVar(value=0.2)
        self.var_default_speed = tk.DoubleVar(value=5.0)
        self.var_status = tk.StringVar(value="Load a case to begin.")
        self.var_metrics = tk.StringVar(value="Path not generated.")

        self.case_obj = None
        self.frames = None
        self.e2e_xy = None
        self.e2e_t = None
        self.ref_full = None
        self.ref_segment = None
        self.handoff = None
        self.planning_end = None
        self.path_ctrl = None
        self.dense_path = None
        self.speed_ctrl = None
        self.dense_traj = None

        self.pick_end_mode = False
        self.drag_anchor_idx = None
        self.drag_speed_idx = None
        self._speed_selected_idx = None

        self._build_ui()
        self._refresh_case_choices()

        if args.case:
            self.after(100, self.load_case)

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Case JSON").grid(row=0, column=0, sticky="w")
        self.case_combo = ttk.Combobox(top, textvariable=self.var_case, width=58)
        self.case_combo.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="Browse", command=self._browse_case).grid(row=0, column=2, padx=2)
        ttk.Button(top, text="Load Case", command=self.load_case).grid(row=0, column=3, padx=2)

        ttk.Label(top, text="Probe directory").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.var_probe, width=58).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="Browse", command=self._browse_probe).grid(row=1, column=2, padx=2)

        ttk.Label(top, text="Route XML").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.var_route_xml, width=58).grid(row=2, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="Browse", command=self._browse_route_xml).grid(row=2, column=2, padx=2)

        ttk.Label(top, text="Route ID").grid(row=2, column=3, sticky="e")
        ttk.Entry(top, textvariable=self.var_route_id, width=12).grid(row=2, column=4, sticky="w", padx=4)
        top.columnconfigure(1, weight=1)

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(body)
        right = ttk.Frame(body, padding=(8, 0, 0, 0))
        body.add(left, weight=4)
        body.add(right, weight=1)

        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.ax_path = self.fig.add_subplot(211)
        self.ax_speed = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=2.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, left)
        toolbar.update()

        self.cid_press = self.canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_motion = self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.cid_release = self.canvas.mpl_connect("button_release_event", self._on_release)

        seg = ttk.LabelFrame(right, text="Planning Segment", padding=8)
        seg.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(seg, text="Snap Planning End to reference line", variable=self.var_snap).pack(anchor="w")
        ttk.Button(seg, text="Pick Planning End on Map", command=self._arm_pick_end).pack(fill=tk.X, pady=3)
        ttk.Label(seg, text="Anchor spacing (m)").pack(anchor="w", pady=(6, 0))
        ttk.Spinbox(seg, from_=0.5, to=20.0, increment=0.5, textvariable=self.var_anchor_spacing).pack(fill=tk.X)
        ttk.Button(seg, text="Generate Reference Anchors", command=self.generate_anchors).pack(fill=tk.X, pady=3)
        ttk.Button(seg, text="Reset Anchors to Reference", command=self.generate_anchors).pack(fill=tk.X, pady=3)

        path = ttk.LabelFrame(right, text="Path Editor", padding=8)
        path.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(path, text="Drag interior anchor points in the BEV plot.\nP0 and PN stay fixed.").pack(anchor="w")
        ttk.Label(path, text="Dense sample spacing (m)").pack(anchor="w", pady=(6, 0))
        ttk.Spinbox(path, from_=0.05, to=1.0, increment=0.05, textvariable=self.var_dense_spacing).pack(fill=tk.X)
        ttk.Button(path, text="Smooth / Rebuild B-Spline", command=self.rebuild_path).pack(fill=tk.X, pady=3)

        speed = ttk.LabelFrame(right, text="Speed Profile v(s)", padding=8)
        speed.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(speed, text="Default speed (m/s)").pack(anchor="w")
        ttk.Spinbox(speed, from_=0.0, to=30.0, increment=0.5, textvariable=self.var_default_speed).pack(fill=tk.X)
        ttk.Button(speed, text="Reset Speed Profile", command=self.reset_speed).pack(fill=tk.X, pady=3)
        row = ttk.Frame(speed)
        row.pack(fill=tk.X)
        ttk.Button(row, text="+ Speed Point", command=self.add_speed_point).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(row, text="- Speed Point", command=self.remove_speed_point).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        ttk.Label(speed, text="Drag speed points vertically in the lower plot.").pack(anchor="w", pady=(4, 0))

        out = ttk.LabelFrame(right, text="Output", padding=8)
        out.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(out, textvariable=self.var_metrics, justify=tk.LEFT, wraplength=300).pack(anchor="w")
        ttk.Button(out, text="Save Manual Plan", command=self.save_current_plan).pack(fill=tk.X, pady=(8, 2))

        status = ttk.LabelFrame(right, text="Status", padding=8)
        status.pack(fill=tk.BOTH, expand=True)
        ttk.Label(status, textvariable=self.var_status, justify=tk.LEFT, wraplength=310).pack(anchor="nw")

        self._redraw()

    def _refresh_case_choices(self):
        case_dir = os.path.join(self.project_root, "cases")
        items = []
        if os.path.isdir(case_dir):
            for name in sorted(os.listdir(case_dir)):
                if name.startswith("case_") and name.endswith(".json"):
                    items.append(os.path.join(case_dir, name))
        self.case_combo["values"] = items

    def _browse_case(self):
        path = filedialog.askopenfilename(
            title="Select case JSON",
            initialdir=os.path.join(self.project_root, "cases"),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.var_case.set(path)

    def _browse_probe(self):
        path = filedialog.askdirectory(
            title="Select probe directory",
            initialdir=os.path.join(self.project_root, "probe_runs")
        )
        if path:
            self.var_probe.set(path)

    def _browse_route_xml(self):
        path = filedialog.askopenfilename(
            title="Select Bench2Drive route XML",
            filetypes=[("XML", "*.xml"), ("All files", "*.*")]
        )
        if path:
            self.var_route_xml.set(path)

    def load_case(self):
        try:
            case_path = os.path.abspath(os.path.expanduser(self.var_case.get().strip()))
            if not os.path.isfile(case_path):
                raise IOError("Case JSON does not exist: %s" % case_path)
            with open(case_path, "r") as f:
                self.case_obj = json.load(f)

            source_run = self.var_probe.get().strip() or self.case_obj.get("source_run", "")
            if source_run and not os.path.isabs(source_run):
                source_run = os.path.abspath(os.path.join(self.project_root, source_run))
            if not source_run or not os.path.isdir(source_run):
                raise IOError("Probe directory not found. Set Probe directory first.")
            self.var_probe.set(source_run)

            route_id = self.var_route_id.get().strip() or str(self.case_obj.get("route_id", "")) or _infer_route_id(self.case_obj.get("case_id", ""))
            if not route_id:
                raise ValueError("Could not infer Route ID; enter it manually.")
            self.var_route_id.set(route_id)

            route_xml = os.path.abspath(os.path.expanduser(self.var_route_xml.get().strip()))
            if not os.path.isfile(route_xml):
                raise IOError("Route XML does not exist: %s" % route_xml)

            self.frames = load_probe_frames(source_run)
            self.e2e_xy = np.asarray([extract_frame_xy(f) for f in self.frames], dtype=float)
            self.e2e_t = np.asarray([extract_frame_time(f) for f in self.frames], dtype=float)

            handoff_t = case_time(self.case_obj, "handoff")
            idx = nearest_frame_index_by_time(self.frames, handoff_t)
            hx, hy = extract_frame_xy(self.frames[idx])
            hs = extract_frame_speed(self.frames[idx])
            self.handoff = {
                "sim_time_s": float(self.e2e_t[idx]),
                "frame_index": int(idx),
                "x": float(hx),
                "y": float(hy),
                "speed_mps": float(hs),
            }

            self.ref_full = parse_route_reference_line(route_xml, route_id)
            self.planning_end = None
            self.ref_segment = None
            self.path_ctrl = None
            self.dense_path = None
            self.speed_ctrl = None
            self.dense_traj = None
            self.var_default_speed.set(max(0.5, float(hs)))
            self.var_status.set(
                "Loaded case: %s\nHandoff t=%.3f s, position=(%.2f, %.2f), speed=%.2f m/s\nPick a Planning End on the BEV map."
                % (self.case_obj.get("case_id", os.path.basename(case_path)),
                   self.handoff["sim_time_s"], hx, hy, hs)
            )
            self._redraw()
        except Exception as exc:
            self._show_error("Load Case failed", exc)

    def _arm_pick_end(self):
        if self.handoff is None or self.ref_full is None:
            messagebox.showwarning("Planning End", "Load a case first.")
            return
        self.pick_end_mode = True
        self.var_status.set("Planning End picking is armed. Click any point in the BEV plot.")
        self.canvas.get_tk_widget().configure(cursor="crosshair")

    def _set_planning_end_from_click(self, x, y):
        ref_proj = project_point_to_polyline([x, y], self.ref_full)
        handoff_proj = project_point_to_polyline([self.handoff["x"], self.handoff["y"]], self.ref_full)
        end_s = float(ref_proj["s"])
        start_s = float(handoff_proj["s"])

        if end_s <= start_s + 0.2:
            raise ValueError(
                "Planning End must be ahead of Handoff on the route reference line. "
                "Choose a point farther forward."
            )

        if self.var_snap.get():
            px, py = ref_proj["point"]
        else:
            px, py = float(x), float(y)

        self.planning_end = {
            "x": float(px),
            "y": float(py),
            "snap_to_reference": bool(self.var_snap.get()),
            "reference_s_m": end_s,
            "from_handoff_m": end_s - start_s,
        }

        self.ref_segment = slice_polyline(self.ref_full, start_s, end_s)
        if not self.var_snap.get():
            self.ref_segment = np.vstack([self.ref_segment, [px, py]])

        self.path_ctrl = None
        self.dense_path = None
        self.speed_ctrl = None
        self.dense_traj = None
        self.var_status.set(
            "Planning End set: x=%.2f y=%.2f, route s=%.2f m, from Handoff=%.2f m.\n"
            "Set anchor spacing, then click Generate Reference Anchors."
            % (px, py, end_s, end_s - start_s)
        )
        self._redraw()

    def generate_anchors(self):
        try:
            if self.ref_segment is None or self.planning_end is None:
                raise ValueError("Pick Planning End first.")
            self.path_ctrl = generate_reference_anchors(
                self.ref_segment, self.var_anchor_spacing.get()
            )
            self.path_ctrl[0] = [self.handoff["x"], self.handoff["y"]]
            self.path_ctrl[-1] = [self.planning_end["x"], self.planning_end["y"]]
            self.rebuild_path(reset_speed=True)
            self.var_status.set(
                "Generated %d path control points. Drag interior points to edit the path."
                % len(self.path_ctrl)
            )
        except Exception as exc:
            self._show_error("Generate Anchors failed", exc)

    def rebuild_path(self, reset_speed=False):
        try:
            if self.path_ctrl is None:
                raise ValueError("Generate path anchors first.")
            self.dense_path = cubic_bspline_dense(
                self.path_ctrl, self.var_dense_spacing.get()
            )
            if reset_speed or self.speed_ctrl is None:
                length = float(polyline_cumulative_s(self.dense_path)[-1])
                self.speed_ctrl = default_speed_control_points(
                    length, self.var_default_speed.get(), count=6
                )
            else:
                old = self.speed_ctrl.copy()
                length = float(polyline_cumulative_s(self.dense_path)[-1])
                old_len = max(float(old[-1, 0]), 1e-6)
                old[:, 0] = old[:, 0] / old_len * length
                self.speed_ctrl = old
            self._recompute_dense_trajectory()
            self._redraw()
        except Exception as exc:
            self._show_error("Smooth path failed", exc)

    def _recompute_dense_trajectory(self):
        if self.dense_path is None or self.speed_ctrl is None:
            self.dense_traj = None
            self.var_metrics.set("Path not generated.")
            return
        self.dense_traj = build_dense_trajectory(self.dense_path, self.speed_ctrl)
        m = self.dense_traj["metrics"]
        dur = m["estimated_duration_s"]
        dur_text = "INF (contains zero speed)" if not math.isfinite(dur) else "%.2f s" % dur
        self.var_metrics.set(
            "Path Length: %.2f m\n"
            "Max Curvature: %.4f 1/m\n"
            "Estimated Duration: %s\n"
            "Target Speed: %.2f ~ %.2f m/s"
            % (
                m["path_length_m"],
                m["max_abs_curvature_1pm"],
                dur_text,
                m["min_target_speed_mps"],
                m["max_target_speed_mps"],
            )
        )

    def reset_speed(self):
        try:
            if self.dense_path is None:
                raise ValueError("Generate and smooth the path first.")
            length = float(polyline_cumulative_s(self.dense_path)[-1])
            self.speed_ctrl = default_speed_control_points(
                length, self.var_default_speed.get(), count=6
            )
            self._recompute_dense_trajectory()
            self._redraw()
        except Exception as exc:
            self._show_error("Reset Speed failed", exc)

    def add_speed_point(self):
        if self.speed_ctrl is None or len(self.speed_ctrl) < 2:
            return
        gaps = np.diff(self.speed_ctrl[:, 0])
        i = int(np.argmax(gaps))
        s = 0.5 * (self.speed_ctrl[i, 0] + self.speed_ctrl[i + 1, 0])
        v = 0.5 * (self.speed_ctrl[i, 1] + self.speed_ctrl[i + 1, 1])
        self.speed_ctrl = np.insert(self.speed_ctrl, i + 1, [s, v], axis=0)
        self._speed_selected_idx = i + 1
        self._recompute_dense_trajectory()
        self._redraw()

    def remove_speed_point(self):
        if self.speed_ctrl is None or len(self.speed_ctrl) <= 2:
            return
        idx = self._speed_selected_idx
        if idx is None or idx <= 0 or idx >= len(self.speed_ctrl) - 1:
            idx = len(self.speed_ctrl) // 2
        self.speed_ctrl = np.delete(self.speed_ctrl, idx, axis=0)
        self._speed_selected_idx = None
        self._recompute_dense_trajectory()
        self._redraw()

    def _nearest_screen_point(self, event, points, ax, max_px=14.0):
        if event.x is None or event.y is None:
            return None
        if points is None or len(points) == 0:
            return None
        disp = ax.transData.transform(np.asarray(points, dtype=float))
        target = np.array([event.x, event.y], dtype=float)
        dist = np.linalg.norm(disp - target, axis=1)
        idx = int(np.argmin(dist))
        if dist[idx] <= max_px:
            return idx
        return None

    def _on_press(self, event):
        try:
            if event.button != 1:
                return
            if event.inaxes == self.ax_path:
                if self.pick_end_mode and event.xdata is not None and event.ydata is not None:
                    self._set_planning_end_from_click(event.xdata, event.ydata)
                    self.pick_end_mode = False
                    self.canvas.get_tk_widget().configure(cursor="")
                    return

                if self.path_ctrl is not None:
                    idx = self._nearest_screen_point(event, self.path_ctrl, self.ax_path)
                    if idx is not None and 0 < idx < len(self.path_ctrl) - 1:
                        self.drag_anchor_idx = idx
                        return

            if event.inaxes == self.ax_speed and self.speed_ctrl is not None:
                idx = self._nearest_screen_point(event, self.speed_ctrl, self.ax_speed)
                if idx is not None:
                    self.drag_speed_idx = idx
                    self._speed_selected_idx = idx
        except Exception as exc:
            self._show_error("Mouse action failed", exc)

    def _on_motion(self, event):
        if self.drag_anchor_idx is not None and event.inaxes == self.ax_path:
            if event.xdata is None or event.ydata is None:
                return
            self.path_ctrl[self.drag_anchor_idx] = [float(event.xdata), float(event.ydata)]
            try:
                self.dense_path = cubic_bspline_dense(
                    self.path_ctrl, self.var_dense_spacing.get()
                )
                if self.speed_ctrl is not None:
                    old_len = max(float(self.speed_ctrl[-1, 0]), 1e-6)
                    new_len = float(polyline_cumulative_s(self.dense_path)[-1])
                    self.speed_ctrl[:, 0] *= new_len / old_len
                    self.speed_ctrl[0, 0] = 0.0
                    self.speed_ctrl[-1, 0] = new_len
                self._recompute_dense_trajectory()
                self._redraw()
            except Exception:
                pass

        if self.drag_speed_idx is not None and event.inaxes == self.ax_speed:
            if event.ydata is None:
                return
            self.speed_ctrl[self.drag_speed_idx, 1] = max(0.0, float(event.ydata))
            self._recompute_dense_trajectory()
            self._redraw()

    def _on_release(self, event):
        self.drag_anchor_idx = None
        self.drag_speed_idx = None

    def _redraw(self):
        self.ax_path.clear()
        self.ax_speed.clear()

        self.ax_path.set_title("BEV Path Editor")
        self.ax_path.set_xlabel("World X (m)")
        self.ax_path.set_ylabel("World Y (m)")
        self.ax_path.grid(True, alpha=0.25)
        self.ax_path.set_aspect("equal", adjustable="datalim")

        if self.e2e_xy is not None:
            self.ax_path.plot(
                self.e2e_xy[:, 0], self.e2e_xy[:, 1],
                linestyle="-", linewidth=1.4, alpha=0.65,
                label="Recorded E2E trajectory"
            )
        if self.ref_full is not None:
            self.ax_path.plot(
                self.ref_full[:, 0], self.ref_full[:, 1],
                linestyle="--", linewidth=1.2, alpha=0.8,
                label="Route reference line"
            )
        if self.handoff is not None:
            self.ax_path.plot(
                [self.handoff["x"]], [self.handoff["y"]],
                marker="o", markersize=7, linestyle="None",
                label="Handoff"
            )
        if self.ref_segment is not None:
            self.ax_path.plot(
                self.ref_segment[:, 0], self.ref_segment[:, 1],
                linestyle="--", linewidth=2.0,
                label="Planning reference segment"
            )
        if self.planning_end is not None:
            self.ax_path.plot(
                [self.planning_end["x"]], [self.planning_end["y"]],
                marker="D", markersize=7, linestyle="None",
                label="Planning End"
            )
        if self.dense_path is not None:
            self.ax_path.plot(
                self.dense_path[:, 0], self.dense_path[:, 1],
                linewidth=2.0,
                label="Smoothed manual path"
            )
        if self.path_ctrl is not None:
            self.ax_path.plot(
                self.path_ctrl[:, 0], self.path_ctrl[:, 1],
                marker="o", markersize=6, linestyle=":",
                label="Path control points"
            )
        if self.ax_path.lines:
            self.ax_path.legend(loc="best", fontsize=8)

        self.ax_speed.set_title("Speed Profile v(s)")
        self.ax_speed.set_xlabel("Path distance s (m)")
        self.ax_speed.set_ylabel("Target speed (m/s)")
        self.ax_speed.grid(True, alpha=0.25)

        if self.dense_traj is not None:
            self.ax_speed.plot(
                self.dense_traj["s"], self.dense_traj["target_speed"],
                linewidth=2.0, label="PCHIP speed profile"
            )
        if self.speed_ctrl is not None:
            self.ax_speed.plot(
                self.speed_ctrl[:, 0], self.speed_ctrl[:, 1],
                marker="o", markersize=7, linestyle="--",
                label="Speed control points"
            )
            vmax = max(float(np.max(self.speed_ctrl[:, 1])) * 1.25, 5.0)
            self.ax_speed.set_ylim(bottom=0.0, top=vmax)
        if self.ax_speed.lines:
            self.ax_speed.legend(loc="best", fontsize=8)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def save_current_plan(self):
        try:
            if self.case_obj is None:
                raise ValueError("Load a case first.")
            if self.planning_end is None or self.ref_segment is None:
                raise ValueError("Pick Planning End first.")
            if self.path_ctrl is None or self.dense_traj is None or self.speed_ctrl is None:
                raise ValueError("Generate anchors and build the path/speed profile first.")

            case_id = str(self.case_obj.get("case_id") or os.path.splitext(os.path.basename(self.var_case.get()))[0])
            out_dir = os.path.join(self.project_root, "cases", "plans")
            gen_dir = os.path.join(out_dir, "generated")
            os.makedirs(gen_dir, exist_ok=True)
            plan_path = os.path.join(out_dir, case_id + ".plan.json")
            npz_path = os.path.join(gen_dir, case_id + ".trajectory.npz")

            doc = make_plan_document(
                case_id=case_id,
                source_run=self.var_probe.get(),
                route_xml=self.var_route_xml.get(),
                route_id=self.var_route_id.get(),
                handoff=self.handoff,
                planning_end=self.planning_end,
                reference_line=self.ref_segment,
                path_control_points=self.path_ctrl,
                speed_control_points=self.speed_ctrl,
                dense_trajectory=self.dense_traj,
                anchor_spacing_m=self.var_anchor_spacing.get(),
                sample_spacing_m=self.var_dense_spacing.get(),
            )
            plan_path, npz_path = save_plan(plan_path, doc, self.dense_traj, npz_path)
            self.var_status.set(
                "Saved manual plan:\n%s\n\nGenerated dense trajectory:\n%s"
                % (plan_path, npz_path)
            )
            messagebox.showinfo(
                "Manual Plan saved",
                "Plan JSON:\n%s\n\nTrajectory NPZ:\n%s" % (plan_path, npz_path)
            )
        except Exception as exc:
            self._show_error("Save Manual Plan failed", exc)

    def _show_error(self, title, exc):
        self.var_status.set("%s: %s" % (title, exc))
        messagebox.showerror(title, "%s\n\n%s" % (exc, traceback.format_exc(limit=4)))


def parse_args():
    p = argparse.ArgumentParser(description="Manual path + speed planner for Failure Replay cases")
    p.add_argument("--project-root", default=PROJECT_ROOT)
    p.add_argument("--case", default="")
    p.add_argument("--probe", default="")
    p.add_argument("--route-xml", default="")
    p.add_argument("--route-id", default="")
    return p.parse_args()


def main():
    args = parse_args()
    app = ManualPlanEditor(args)
    app.mainloop()


if __name__ == "__main__":
    main()
