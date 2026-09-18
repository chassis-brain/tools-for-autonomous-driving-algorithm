#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from b2d_collector.failure_replay.manual_plan import (
    build_dense_trajectory,
    cubic_bspline_dense,
    default_speed_control_points,
    generate_free_anchors,
    make_plan_document_v2,
    polyline_cumulative_s,
    save_plan,
)


class ManualPlanPanel(ttk.Frame):
    """Embedded manual spatial-path + speed-profile editor.

    Geometry is intentionally time-free: P0 is the Tape-aligned Handoff state,
    PN is the user's free spatial End point, and the vehicle later follows P(s)
    with target speed v(s) through ManualTrajectoryFollower.
    """

    def __init__(self, master, workbench):
        super().__init__(master)
        self.workbench = workbench

        self.e2e_xy = None
        self.ref_xy = None
        self.handoff = None
        self.end = None
        self.path_ctrl = None
        self.dense_path = None
        self.speed_ctrl = None
        self.dense_traj = None
        self.drag_anchor_idx = None
        self.drag_speed_idx = None
        self.speed_selected_idx = None
        self.saved_plan_path: Optional[Path] = None

        self.anchor_spacing = tk.DoubleVar(value=6.0)
        self.dense_spacing = tk.DoubleVar(value=0.20)
        self.default_speed = tk.DoubleVar(value=5.0)
        self.metrics_var = tk.StringVar(value="Load the current case to start manual planning.")
        self.status_var = tk.StringVar(value="Manual path is spatial-only; execution time is not preassigned.")

        self.lat_kp = tk.DoubleVar(value=0.45)
        self.lat_ki = tk.DoubleVar(value=0.015)
        self.lat_kd = tk.DoubleVar(value=0.10)
        self.heading_kp = tk.DoubleVar(value=1.10)
        self.speed_kp = tk.DoubleVar(value=0.38)
        self.speed_ki = tk.DoubleVar(value=0.055)
        self.speed_kd = tk.DoubleVar(value=0.06)

        self._build_ui()

    def _build_ui(self):
        # Keep the primary save action permanently visible.  The original V2
        # placed it at the bottom of a long fixed-height right column, so on
        # 720/900 px desktops it could be clipped below the window.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        main = ttk.Frame(self)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=4)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        plot_card = ttk.Frame(main, style="Card.TFrame")
        plot_card.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=(10, 5))

        # Right-hand controls are scrollable, but the Save button is NOT in
        # this scroll region; it lives in the fixed footer below.
        controls_card = ttk.Frame(main, style="Card.TFrame")
        controls_card.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=(10, 5))
        controls_card.columnconfigure(0, weight=1)
        controls_card.rowconfigure(0, weight=1)

        self.controls_canvas = tk.Canvas(
            controls_card,
            highlightthickness=0,
            borderwidth=0,
            background="#20252B",
        )
        controls_scroll = ttk.Scrollbar(
            controls_card,
            orient="vertical",
            command=self.controls_canvas.yview,
        )
        self.controls_canvas.configure(yscrollcommand=controls_scroll.set)
        self.controls_canvas.grid(row=0, column=0, sticky="nsew")
        controls_scroll.grid(row=0, column=1, sticky="ns")

        controls = ttk.Frame(self.controls_canvas, style="Card.TFrame", padding=12)
        self._controls_window = self.controls_canvas.create_window(
            (0, 0), window=controls, anchor="nw"
        )

        def _sync_scrollregion(_event=None):
            self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

        def _sync_controls_width(event):
            self.controls_canvas.itemconfigure(self._controls_window, width=event.width)

        controls.bind("<Configure>", _sync_scrollregion)
        self.controls_canvas.bind("<Configure>", _sync_controls_width)

        def _mousewheel(event):
            # Linux/X11 uses Button-4/5; Windows/macOS generally use MouseWheel.
            if getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            else:
                raw = getattr(event, "delta", 0)
                delta = -1 if raw > 0 else (1 if raw < 0 else 0)
            if delta:
                self.controls_canvas.yview_scroll(delta * 3, "units")

        self.controls_canvas.bind("<Enter>", lambda _e: (
            self.controls_canvas.bind_all("<MouseWheel>", _mousewheel),
            self.controls_canvas.bind_all("<Button-4>", _mousewheel),
            self.controls_canvas.bind_all("<Button-5>", _mousewheel),
        ))
        self.controls_canvas.bind("<Leave>", lambda _e: (
            self.controls_canvas.unbind_all("<MouseWheel>"),
            self.controls_canvas.unbind_all("<Button-4>"),
            self.controls_canvas.unbind_all("<Button-5>"),
        ))

        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor="#20252B")
        self.ax_path = self.fig.add_subplot(211)
        self.ax_speed = self.fig.add_subplot(212)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08, hspace=0.34)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_card, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x", padx=4, pady=(0, 4))

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        head = ttk.Label(controls, text="Manual Path + PID", style="Title.TLabel")
        head.pack(anchor="w", pady=(0, 4))
        ttk.Label(
            controls,
            text="Handoff → free spatial End. Drag only a few anchors; the dense path is rebuilt automatically.",
            style="Muted.TLabel",
            wraplength=300,
        ).pack(anchor="w", pady=(0, 12))

        ttk.Button(
            controls,
            text="Load current Case / End",
            style="Accent.TButton",
            command=self.load_from_workbench,
        ).pack(fill="x", pady=(0, 10))

        path_box = ttk.LabelFrame(controls, text="Path P(s)", padding=10)
        path_box.pack(fill="x", pady=(0, 10))
        self._labeled_spin(path_box, "Anchor spacing (m)", self.anchor_spacing, 2.0, 15.0, 0.5)
        self._labeled_spin(path_box, "Dense spacing (m)", self.dense_spacing, 0.05, 0.5, 0.05)
        ttk.Button(path_box, text="Generate / Reset Anchors", command=self.generate_anchors).pack(fill="x", pady=(6, 2))
        ttk.Button(path_box, text="Smooth B-Spline", command=self.rebuild_path).pack(fill="x", pady=2)
        ttk.Label(
            path_box,
            text="P0=Handoff and PN=End are locked. Drag interior points freely in the upper plot.",
            style="Muted.TLabel",
            wraplength=270,
        ).pack(anchor="w", pady=(5, 0))

        speed_box = ttk.LabelFrame(controls, text="Speed v(s)", padding=10)
        speed_box.pack(fill="x", pady=(0, 10))
        self._labeled_spin(speed_box, "Uniform speed for reset (m/s)", self.default_speed, 0.0, 25.0, 0.5)
        ttk.Button(speed_box, text="Apply Uniform Speed to Entire Curve", command=self.reset_speed).pack(fill="x", pady=(6, 2))
        row = ttk.Frame(speed_box)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="+ Speed Point", command=self.add_speed_point).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(row, text="− Speed Point", command=self.remove_speed_point).pack(side="left", expand=True, fill="x", padx=(3, 0))
        ttk.Label(
            speed_box,
            text="The field above only sets the value used by Apply. The lower plot is the exact v(s) that Recovery will execute and save.",
            style="Muted.TLabel",
            wraplength=270,
        ).pack(anchor="w", pady=(5, 0))

        pid_box = ttk.LabelFrame(controls, text="PID tracking", padding=10)
        pid_box.pack(fill="x", pady=(0, 10))
        grid = ttk.Frame(pid_box)
        grid.pack(fill="x")
        fields = [
            ("Lat Kp", self.lat_kp), ("Lat Ki", self.lat_ki), ("Lat Kd", self.lat_kd),
            ("Heading Kp", self.heading_kp),
            ("Speed Kp", self.speed_kp), ("Speed Ki", self.speed_ki), ("Speed Kd", self.speed_kd),
        ]
        for i, (label, var) in enumerate(fields):
            ttk.Label(grid, text=label).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(grid, textvariable=var, width=9).grid(row=i, column=1, sticky="e", pady=2)
        grid.columnconfigure(0, weight=1)

        out = ttk.LabelFrame(controls, text="Plan status", padding=10)
        out.pack(fill="x", pady=(0, 10))
        ttk.Label(out, textvariable=self.metrics_var, justify="left", wraplength=280).pack(anchor="w")
        ttk.Label(out, textvariable=self.status_var, style="Muted.TLabel", justify="left", wraplength=280).pack(anchor="w", pady=(6, 0))

        ttk.Label(
            controls,
            text="The accented Save bar below is fixed and remains visible even when this parameter column is scrolled.",
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor="w", pady=(2, 12))

        # Fixed bottom action bar: always visible on 720p/900p desktops.
        footer = ttk.Frame(self, style="Card.TFrame", padding=(14, 10))
        footer.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        footer.columnconfigure(0, weight=1)
        ttk.Label(
            footer,
            text="Ready to execute after saving: case JSON + plan.json + trajectory.npz",
            style="Muted.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Button(
            footer,
            text="Save Manual Plan for Recovery",
            style="Accent.TButton",
            command=self.save_plan,
        ).grid(row=0, column=1, sticky="e", ipadx=12, ipady=4)

        self._redraw()

    @staticmethod
    def _labeled_spin(parent, label, var, lo, hi, step):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(3, 0))
        ttk.Spinbox(parent, from_=lo, to=hi, increment=step, textvariable=var).pack(fill="x")

    def load_from_workbench(self):
        wb = self.workbench
        if not wb.frames or wb.loaded_probe is None:
            return self._error("Load a Probe and select the case first.")
        hidx = wb.marker_indices.get("handoff")
        if hidx is None:
            return self._error("Set Handoff in Review / Case first.")
        if wb.planning_end_point is None:
            return self._error("Pick the free spatial End point in Review / Case first.")

        h = wb.frames[hidx]
        self.handoff = {
            "sample_index": int(h["sample_index"]),
            "sim_time_s": float(h["t"]),
            "x": float(h["x"]), "y": float(h["y"]), "z": float(h["z"]),
            "yaw": float(h["yaw"]), "speed_mps": float(h["speed"]),
        }
        self.end = {
            "x": float(wb.planning_end_point["x"]),
            "y": float(wb.planning_end_point["y"]),
            "radius_m": float(wb.end_radius_var.get()),
            "confirm_frames": int(wb.end_confirm_var.get()),
        }
        self.e2e_xy = np.asarray([[f["x"], f["y"]] for f in wb.frames], dtype=float)
        self.ref_xy = np.asarray([[p["x"], p["y"]] for p in wb.reference_line], dtype=float) if wb.reference_line else None
        self.default_speed.set(max(1.0, float(h["speed"])))
        self.saved_plan_path = None
        self.generate_anchors()
        self.status_var.set(
            "Loaded %s. Handoff t=%.3fs; End=(%.2f, %.2f)."
            % (wb.case_id_var.get().strip() or "current case", h["t"], self.end["x"], self.end["y"])
        )

    def generate_anchors(self):
        if self.handoff is None or self.end is None:
            return self._error("Load current Case / End first.")
        self.path_ctrl = generate_free_anchors(
            [self.handoff["x"], self.handoff["y"]],
            [self.end["x"], self.end["y"]],
            anchor_spacing_m=float(self.anchor_spacing.get()),
            min_points=4,
            max_points=12,
        )
        self.rebuild_path(reset_speed=True)

    def rebuild_path(self, reset_speed=False):
        if self.path_ctrl is None:
            return self._error("Generate anchors first.")
        try:
            self.dense_path = cubic_bspline_dense(self.path_ctrl, float(self.dense_spacing.get()))
            length = float(polyline_cumulative_s(self.dense_path)[-1])
            if reset_speed or self.speed_ctrl is None:
                self.speed_ctrl = default_speed_control_points(length, float(self.default_speed.get()), count=6)
            else:
                old = max(float(self.speed_ctrl[-1, 0]), 1e-6)
                self.speed_ctrl[:, 0] *= length / old
                self.speed_ctrl[0, 0] = 0.0
                self.speed_ctrl[-1, 0] = length
            self._recompute()
        except Exception as exc:
            self._error(str(exc))

    def _recompute(self):
        if self.dense_path is None or self.speed_ctrl is None:
            return
        self.speed_ctrl = self.speed_ctrl[np.argsort(self.speed_ctrl[:, 0])]
        self.dense_traj = build_dense_trajectory(self.dense_path, self.speed_ctrl)
        m = self.dense_traj["metrics"]
        duration = m["estimated_duration_s"]
        duration_txt = "∞" if not math.isfinite(duration) else "%.2f s" % duration
        self.metrics_var.set(
            "Path length   %.2f m\nMax curvature %.4f 1/m\nSpeed range   %.2f – %.2f m/s\nEst. duration %s"
            % (
                m["path_length_m"], m["max_abs_curvature_1pm"],
                m["min_target_speed_mps"], m["max_target_speed_mps"], duration_txt,
            )
        )
        self._redraw()

    def reset_speed(self):
        if self.dense_path is None:
            return self._error("Build the path first.")
        length = float(polyline_cumulative_s(self.dense_path)[-1])
        self.speed_ctrl = default_speed_control_points(length, float(self.default_speed.get()), count=6)
        self._recompute()

    def add_speed_point(self):
        if self.speed_ctrl is None or len(self.speed_ctrl) < 2:
            return
        gaps = np.diff(self.speed_ctrl[:, 0])
        i = int(np.argmax(gaps))
        row = 0.5 * (self.speed_ctrl[i] + self.speed_ctrl[i + 1])
        self.speed_ctrl = np.insert(self.speed_ctrl, i + 1, row, axis=0)
        self.speed_selected_idx = i + 1
        self._recompute()

    def remove_speed_point(self):
        if self.speed_ctrl is None or len(self.speed_ctrl) <= 2:
            return
        idx = self.speed_selected_idx
        if idx is None or idx <= 0 or idx >= len(self.speed_ctrl) - 1:
            idx = len(self.speed_ctrl) // 2
        self.speed_ctrl = np.delete(self.speed_ctrl, idx, axis=0)
        self.speed_selected_idx = None
        self._recompute()

    @staticmethod
    def _nearest_screen(event, points, ax, max_px=15.0):
        if points is None or len(points) == 0 or event.x is None or event.y is None:
            return None
        disp = ax.transData.transform(np.asarray(points, dtype=float))
        d = np.linalg.norm(disp - np.asarray([event.x, event.y], dtype=float), axis=1)
        idx = int(np.argmin(d))
        return idx if d[idx] <= max_px else None

    def _on_press(self, event):
        if event.button != 1:
            return
        if event.inaxes == self.ax_path and self.path_ctrl is not None:
            idx = self._nearest_screen(event, self.path_ctrl, self.ax_path)
            if idx is not None and 0 < idx < len(self.path_ctrl) - 1:
                self.drag_anchor_idx = idx
        elif event.inaxes == self.ax_speed and self.speed_ctrl is not None:
            idx = self._nearest_screen(event, self.speed_ctrl, self.ax_speed)
            if idx is not None:
                self.drag_speed_idx = idx
                self.speed_selected_idx = idx

    def _on_motion(self, event):
        if self.drag_anchor_idx is not None and event.inaxes == self.ax_path:
            if event.xdata is None or event.ydata is None:
                return
            self.path_ctrl[self.drag_anchor_idx] = [float(event.xdata), float(event.ydata)]
            try:
                self.dense_path = cubic_bspline_dense(self.path_ctrl, float(self.dense_spacing.get()))
                length = float(polyline_cumulative_s(self.dense_path)[-1])
                if self.speed_ctrl is not None:
                    old = max(float(self.speed_ctrl[-1, 0]), 1e-6)
                    self.speed_ctrl[:, 0] *= length / old
                    self.speed_ctrl[0, 0] = 0.0
                    self.speed_ctrl[-1, 0] = length
                self._recompute()
            except Exception:
                pass
        elif self.drag_speed_idx is not None and event.inaxes == self.ax_speed:
            if event.xdata is None or event.ydata is None:
                return
            i = self.drag_speed_idx
            v = max(0.0, float(event.ydata))
            if i == 0:
                s = 0.0
            elif i == len(self.speed_ctrl) - 1:
                s = float(self.speed_ctrl[-1, 0])
            else:
                lo = float(self.speed_ctrl[i - 1, 0]) + 0.10
                hi = float(self.speed_ctrl[i + 1, 0]) - 0.10
                s = max(lo, min(hi, float(event.xdata)))
            self.speed_ctrl[i] = [s, v]
            self._recompute()

    def _on_release(self, _event):
        self.drag_anchor_idx = None
        self.drag_speed_idx = None

    def _axis_dark(self, ax):
        ax.set_facecolor("#191D22")
        ax.tick_params(colors="#AAB1B8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#3A4149")
        ax.xaxis.label.set_color("#BFC5CB")
        ax.yaxis.label.set_color("#BFC5CB")
        ax.title.set_color("#ECEEEF")
        ax.grid(True, color="#58616A", alpha=0.18, linewidth=0.7)

    def _redraw(self):
        self.ax_path.clear()
        self.ax_speed.clear()
        self._axis_dark(self.ax_path)
        self._axis_dark(self.ax_speed)

        self.ax_path.set_title("Spatial Path Editor — Handoff to End")
        self.ax_path.set_xlabel("World X (m)")
        self.ax_path.set_ylabel("World Y (m)")
        self.ax_path.set_aspect("equal", adjustable="datalim")
        if self.e2e_xy is not None:
            self.ax_path.plot(self.e2e_xy[:, 0], self.e2e_xy[:, 1], color="#9DA5AD", linewidth=1.3, alpha=0.58, label="Original E2E")
        if self.ref_xy is not None and len(self.ref_xy) >= 2:
            self.ax_path.plot(self.ref_xy[:, 0], self.ref_xy[:, 1], color="#78A6A1", linestyle="--", linewidth=1.3, alpha=0.72, label="Route reference")
        if self.dense_path is not None:
            self.ax_path.plot(self.dense_path[:, 0], self.dense_path[:, 1], color="#84AAA5", linewidth=2.5, label="Manual B-Spline P(s)")
        if self.path_ctrl is not None:
            self.ax_path.plot(self.path_ctrl[:, 0], self.path_ctrl[:, 1], color="#C0A36F", marker="o", markerfacecolor="#D0B47C", linestyle=":", linewidth=1.1, label="Editable anchors")
        if self.handoff is not None:
            self.ax_path.plot([self.handoff["x"]], [self.handoff["y"]], color="#D2AE72", marker="o", markersize=8, linestyle="None", label="Handoff")
        if self.end is not None:
            self.ax_path.plot([self.end["x"]], [self.end["y"]], color="#A68BB3", marker="D", markersize=8, linestyle="None", label="Free End")
        if self.ax_path.lines:
            leg = self.ax_path.legend(loc="best", fontsize=8)
            if leg:
                leg.get_frame().set_alpha(0.88)
                leg.get_frame().set_facecolor("#23282E")
                leg.get_frame().set_edgecolor("#3A4149")
                for txt in leg.get_texts():
                    txt.set_color("#D8DCE1")

        self.ax_speed.set_title("Speed vs Position — v(s)")
        self.ax_speed.set_xlabel("Path position s (m)")
        self.ax_speed.set_ylabel("Target speed (m/s)")
        if self.dense_traj is not None:
            self.ax_speed.plot(self.dense_traj["s"], self.dense_traj["target_speed"], color="#84AAA5", linewidth=2.4, label="PCHIP v(s)")
        if self.speed_ctrl is not None:
            self.ax_speed.plot(self.speed_ctrl[:, 0], self.speed_ctrl[:, 1], color="#C0A36F", marker="o", markerfacecolor="#D0B47C", linestyle="--", linewidth=1.0, label="Editable speed points")
            vmax = max(5.0, float(np.max(self.speed_ctrl[:, 1])) * 1.25)
            self.ax_speed.set_ylim(0.0, vmax)
        if self.ax_speed.lines:
            leg = self.ax_speed.legend(loc="best", fontsize=8)
            if leg:
                leg.get_frame().set_alpha(0.88)
                leg.get_frame().set_facecolor("#23282E")
                leg.get_frame().set_edgecolor("#3A4149")
                for txt in leg.get_texts():
                    txt.set_color("#D8DCE1")
        self.canvas.draw_idle()

    def _controller_doc(self):
        return {
            "lateral_kp": float(self.lat_kp.get()),
            "lateral_ki": float(self.lat_ki.get()),
            "lateral_kd": float(self.lat_kd.get()),
            "heading_kp": float(self.heading_kp.get()),
            "curvature_feedforward": 0.85,
            "speed_kp": float(self.speed_kp.get()),
            "speed_ki": float(self.speed_ki.get()),
            "speed_kd": float(self.speed_kd.get()),
            "wheelbase_m": 2.85,
            "max_steer_angle_rad": 1.22,
            "max_throttle": 0.80,
            "max_brake": 0.90,
            "lookahead_base_m": 0.8,
            "lookahead_speed_gain": 0.18,
            "speed_preview_seconds": 0.55,
        }

    def save_plan(self):
        try:
            wb = self.workbench
            if self.handoff is None or self.end is None or self.dense_traj is None:
                raise RuntimeError("Load the case and generate the manual path first.")
            case_id = wb.case_id_var.get().strip()
            if not case_id:
                raise RuntimeError("Case ID is empty.")
            if wb.loaded_probe is None:
                raise RuntimeError("No Probe is loaded.")

            out = wb.project_root() / "cases" / "plans"
            gen = out / "generated"
            out.mkdir(parents=True, exist_ok=True)
            gen.mkdir(parents=True, exist_ok=True)
            plan_path = out / (case_id + ".plan.json")
            npz_path = gen / (case_id + ".trajectory.npz")
            ref = self.ref_xy if self.ref_xy is not None else np.empty((0, 2), dtype=float)
            doc = make_plan_document_v2(
                case_id=case_id,
                source_run=str(wb.loaded_probe),
                route_xml=str(wb.route_xml()),
                route_id=wb.route_var.get().strip(),
                handoff=self.handoff,
                end_point=self.end,
                reference_line=ref,
                path_control_points=self.path_ctrl,
                speed_control_points=self.speed_ctrl,
                dense_trajectory=self.dense_traj,
                anchor_spacing_m=float(self.anchor_spacing.get()),
                sample_spacing_m=float(self.dense_spacing.get()),
                controller=self._controller_doc(),
            )
            save_plan(str(plan_path), doc, self.dense_traj, str(npz_path))
            self.saved_plan_path = plan_path.resolve()

            case_path = wb.project_root() / "cases" / (case_id + ".json")
            if case_path.is_file():
                case = json.loads(case_path.read_text(encoding="utf-8"))
                case["replacement"] = {
                    "mode": "manual_pid",
                    "plan": "plans/%s.plan.json" % case_id,
                }
                case_path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                if hasattr(wb, "recovery_case_var"):
                    wb.recovery_case_var.set(str(case_path.resolve()))
            saved_s = np.asarray(self.dense_traj["s"], dtype=float)
            saved_v = np.asarray(self.dense_traj["target_speed"], dtype=float)
            checkpoints = []
            for frac in (0.0, 0.25, 0.50, 0.75, 1.0):
                qs = float(saved_s[-1]) * frac
                qv = float(np.interp(qs, saved_s, saved_v))
                checkpoints.append((qs, qv))
            speed_summary = ", ".join(
                "%.1fm=%.2fm/s" % (qs, qv) for qs, qv in checkpoints
            )

            self.status_var.set(
                "Saved %s | executable speed %.2f–%.2f m/s"
                % (plan_path.name, float(saved_v.min()), float(saved_v.max()))
            )
            wb._log(
                "Saved Manual PID plan: %s\n"
                "Dense trajectory: %s\n"
                "EXECUTABLE v(s): %s\n"
                "Speed range: %.2f .. %.2f m/s\n"
                % (
                    plan_path,
                    npz_path,
                    speed_summary,
                    float(saved_v.min()),
                    float(saved_v.max()),
                )
            )
            messagebox.showinfo(
                "Manual plan saved",
                "Plan and dense trajectory are ready for Recovery.\n\n"
                "Actual executable v(s):\n%s\n\nRange: %.2f .. %.2f m/s"
                % (
                    speed_summary,
                    float(saved_v.min()),
                    float(saved_v.max()),
                ),
            )
        except Exception as exc:
            self._error(str(exc), popup=True)

    def _error(self, text, popup=False):
        self.status_var.set(str(text))
        if popup:
            messagebox.showerror("Manual Planner", str(text))
        return None
