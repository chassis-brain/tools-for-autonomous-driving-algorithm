#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Failure Replay Workbench
========================

Single-window GUI for the current Bench2Drive failure-replay workflow:

    CARLA
      -> E2E + Probe
      -> Load/Review Behavior Tape
      -> Select Tape-aligned Record Start / Handoff + free spatial End
      -> Write intervention case
      -> Update recovery YAML
      -> Preflight
      -> Replay -> PDM Expert or Manual PID -> Base dataset collection
      -> Validate / export latest dataset

Target runtime: Python 3.7+ (b2d-collector37).
GUI dependency: Tkinter (standard library).
Optional dependency used by the existing project: PyYAML.

This GUI deliberately keeps the already-validated replay contract frozen:
    case time_offset_seconds = 0.0
    causal replay shift       = +1 tick (implemented in failure_replay/agent.py)

Put this file at:
    <project_root>/tools/failure_replay_gui.py

Then launch from the collector environment:
    python tools/failure_replay_gui.py
"""

from __future__ import print_function

import gzip
import json
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    # Python 2 fallback is intentionally unsupported, but this makes the error clear.
    import Tkinter as tk
    import tkFileDialog as filedialog
    import tkMessageBox as messagebox
    import ttk

try:
    import yaml
except Exception:
    yaml = None


APP_NAME = "Failure Replay Workbench"
SETTINGS_PATH = Path.home() / ".failure_replay_workbench.json"

DEFAULT_TYPES = [
    "slow_decision",
    "blocked",
    "collision_risk",
    "route_deviation",
    "unsafe_behavior",
    "other",
]


def now_stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def shell_join(parts):
    """Readable shell-like representation for logs only."""
    out = []
    for part in parts:
        s = str(part)
        if not s:
            out.append("''")
        elif all(c.isalnum() or c in "._-/=:+," for c in s):
            out.append(s)
        else:
            out.append("'" + s.replace("'", "'\\''") + "'")
    return " ".join(out)


def read_json(path, default=None):
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def nested_get(d, path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def as_float(v, default=0.0):
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def infer_project_root():
    here = Path(__file__).resolve()
    # Expected installation: <project>/tools/failure_replay_gui.py
    if here.parent.name == "tools":
        return here.parent.parent
    cwd = Path.cwd()
    if (cwd / "src" / "b2d_collector").is_dir():
        return cwd
    return cwd


def infer_sparsedrive_python():
    exe = Path(sys.executable).resolve()
    # Typical:
    # ~/miniconda3/envs/b2d-collector37/bin/python
    try:
        if exe.parent.name == "bin" and exe.parent.parent.parent.name == "envs":
            envs = exe.parent.parent.parent
            candidate = envs / "sparsedrive-b2d" / "bin" / "python"
            if candidate.is_file():
                return str(candidate)
    except Exception:
        pass
    return ""


def find_carla_egg(carla_root):
    root = Path(carla_root).expanduser()
    dist = root / "PythonAPI" / "carla" / "dist"
    preferred = dist / "carla-0.9.15-py3.7-linux-x86_64.egg"
    if preferred.is_file():
        return str(preferred)

    # Prefer Python 3.7 egg, then cp37 wheel. Never choose cp27.
    candidates = list(dist.glob("carla-*-py3.7-*.egg"))
    if candidates:
        return str(sorted(candidates)[0])

    candidates = list(dist.glob("carla-*-cp37-*.whl"))
    if candidates:
        return str(sorted(candidates)[0])

    return ""


def load_route_reference_line(route_xml_path, route_id):
    """Load the sparse route reference polyline from Bench2Drive route XML.

    Only ./waypoints/position is treated as the route reference line, so
    scenario trigger points are intentionally excluded.
    """
    path = Path(route_xml_path).expanduser()
    if not path.is_file():
        return []

    root = ET.parse(str(path)).getroot()
    target = None
    for route in root.findall("route"):
        if str(route.attrib.get("id", "")).strip() == str(route_id).strip():
            target = route
            break
    if target is None:
        return []

    points = []
    for node in target.findall("./waypoints/position"):
        try:
            point = {
                "x": float(node.attrib["x"]),
                "y": float(node.attrib["y"]),
                "z": float(node.attrib.get("z", 0.0)),
                "yaw": (
                    float(node.attrib["yaw"])
                    if "yaw" in node.attrib else None
                ),
            }
        except (KeyError, TypeError, ValueError):
            continue

        if points:
            prev = points[-1]
            if (
                abs(prev["x"] - point["x"]) < 1e-9
                and abs(prev["y"] - point["y"]) < 1e-9
                and abs(prev["z"] - point["z"]) < 1e-9
            ):
                continue
        points.append(point)
    return points


def cumulative_xy_distance(points):
    """Return cumulative XY arc length for a list of {x, y, ...} points."""
    if not points:
        return []
    out = [0.0]
    for i in range(1, len(points)):
        dx = float(points[i]["x"]) - float(points[i - 1]["x"])
        dy = float(points[i]["y"]) - float(points[i - 1]["y"])
        out.append(out[-1] + math.sqrt(dx * dx + dy * dy))
    return out


def nearest_xy_index(points, x, y):
    """Return index of the closest XY point, or None for an empty sequence."""
    if not points:
        return None
    best_idx = None
    best_d2 = None
    for i, point in enumerate(points):
        dx = float(point["x"]) - float(x)
        dy = float(point["y"]) - float(y)
        d2 = dx * dx + dy * dy
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_idx = i
    return best_idx


class ProcessRunner(object):
    def __init__(self, event_queue):
        self.event_queue = event_queue
        self.processes = {}
        self.callbacks = {}

    def running(self, name):
        p = self.processes.get(name)
        return p is not None and p.poll() is None

    def start(self, name, cmd, cwd=None, env=None, callback=None):
        if self.running(name):
            raise RuntimeError("%s is already running" % name)

        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            start_new_session=True,
        )
        self.processes[name] = proc
        if callback is not None:
            self.callbacks[name] = callback

        self.event_queue.put(("log", "\n[%s] START pid=%d\n$ %s\n" % (
            name, proc.pid, shell_join(cmd)
        )))

        thread = threading.Thread(
            target=self._reader,
            args=(name, proc),
            daemon=True,
        )
        thread.start()
        return proc

    def _reader(self, name, proc):
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                self.event_queue.put(("log", "[%s] %s" % (name, line)))
        except Exception as exc:
            self.event_queue.put(("log", "[%s] reader error: %s\n" % (name, exc)))
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            rc = proc.wait()
            self.event_queue.put(("done", name, rc))

    def stop(self, name):
        p = self.processes.get(name)
        if p is None or p.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            try:
                p.terminate()
            except Exception:
                pass

    def stop_all(self):
        for name in list(self.processes):
            self.stop(name)


class FailureReplayWorkbench(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title(APP_NAME)
        self.geometry("1420x920")
        self.minsize(1120, 720)

        self._configure_fonts()

        self.events = queue.Queue()
        self.runner = ProcessRunner(self.events)

        self.frames = []
        self.canvas_points = []
        self.reference_line = []
        self.reference_canvas_points = []
        self.reference_s = []
        self.planning_end_point = None
        self.pick_planning_end_active = False
        self.loaded_probe = None
        self.marker_indices = {
            "record_start": None,
            "handoff": None,
            "record_end": None,
        }
        self.route_records = {}
        self.latest_clip = None

        self.settings_vars = {}
        self._build_ui()
        self._load_settings()
        self._apply_derived_defaults(only_empty=True)
        self.refresh_everything(silent=True)

        self.after(100, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_fonts(self):
        """Soft dark workbench theme with modern Linux-safe sans fonts."""
        try:
            import tkinter.font as tkfont
        except ImportError:
            import tkFont as tkfont
        available = set(tkfont.families(self))
        preferred = [
            "Inter", "Noto Sans", "Noto Sans CJK SC", "Source Sans 3",
            "IBM Plex Sans", "Ubuntu", "DejaVu Sans", "Liberation Sans", "Arial"
        ]
        family = next((name for name in preferred if name in available), "DejaVu Sans")
        mono_pref = [
            "JetBrains Mono", "Cascadia Mono", "Noto Sans Mono",
            "DejaVu Sans Mono", "Liberation Mono"
        ]
        fixed = next((name for name in mono_pref if name in available), family)

        for named in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            try:
                tkfont.nametofont(named).configure(family=family, size=10)
            except Exception:
                pass
        try:
            tkfont.nametofont("TkHeadingFont").configure(family=family, size=10, weight="bold")
        except Exception:
            pass
        try:
            tkfont.nametofont("TkFixedFont").configure(family=fixed, size=10)
        except Exception:
            pass

        self._ui_font_family = family
        self._ui_fixed_font_family = fixed
        self._ui_heading_font = tkfont.Font(self, family=family, size=12, weight="bold")

        # Intentionally low-saturation: the GUI is often kept open beside CARLA
        # for long sessions, so contrast is clear without a bright blue/black UI.
        self._ui_colors = {
            "bg": "#171A1F",
            "surface": "#20252B",
            "surface_alt": "#292F36",
            "surface_hover": "#333A42",
            "text": "#E7E9EC",
            "muted": "#9EA7B1",
            "accent": "#6F9E98",
            "accent_hover": "#7FAFA8",
            "accent_pressed": "#5E8883",
            "border": "#353C44",
            "canvas": "#191D22",
            "selection": "#405A57",
        }
        palette = self._ui_colors
        self.configure(background=palette["bg"])

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        family10 = (family, 10)
        family10b = (family, 10, "bold")
        style.configure(".", font=family10, background=palette["bg"], foreground=palette["text"])
        style.configure("TFrame", background=palette["bg"])
        style.configure("Card.TFrame", background=palette["surface"])
        style.configure("TLabel", background=palette["bg"], foreground=palette["text"])
        style.configure("Card.TLabel", background=palette["surface"], foreground=palette["text"])
        style.configure("Muted.TLabel", background=palette["surface"], foreground=palette["muted"])
        style.configure(
            "Title.TLabel",
            background=palette["surface"], foreground="#F2F3F4",
            font=(family, 15, "bold"),
        )
        style.configure(
            "TLabelframe",
            background=palette["surface"], foreground=palette["text"],
            bordercolor=palette["border"], lightcolor=palette["border"],
            darkcolor=palette["border"], relief="solid", borderwidth=1,
        )
        style.configure(
            "TLabelframe.Label",
            background=palette["surface"], foreground="#ECEEEF",
            font=family10b,
        )
        style.configure(
            "TButton", background=palette["surface_alt"], foreground=palette["text"],
            borderwidth=0, padding=(11, 7), focusthickness=0,
        )
        style.map(
            "TButton",
            background=[("active", palette["surface_hover"]), ("pressed", "#2B3138")],
            foreground=[("disabled", "#717981")],
        )
        style.configure(
            "Accent.TButton", background=palette["accent"], foreground="#F7FAF9",
            borderwidth=0, padding=(13, 8), font=family10b, focusthickness=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", palette["accent_hover"]), ("pressed", palette["accent_pressed"])],
        )
        style.configure(
            "TEntry", fieldbackground=palette["surface_alt"], foreground=palette["text"],
            insertcolor=palette["text"], bordercolor=palette["border"],
            lightcolor=palette["border"], darkcolor=palette["border"], padding=5,
        )
        style.configure(
            "TCombobox", fieldbackground=palette["surface_alt"], background=palette["surface_alt"],
            foreground=palette["text"], arrowcolor=palette["muted"], bordercolor=palette["border"],
            lightcolor=palette["border"], darkcolor=palette["border"], padding=4,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", palette["surface_alt"])],
            selectbackground=[("readonly", palette["surface_alt"])],
            selectforeground=[("readonly", palette["text"])],
        )
        style.configure(
            "TSpinbox", fieldbackground=palette["surface_alt"], background=palette["surface_alt"],
            foreground=palette["text"], arrowcolor=palette["muted"], bordercolor=palette["border"],
            lightcolor=palette["border"], darkcolor=palette["border"], padding=4,
        )
        style.configure("TCheckbutton", background=palette["surface"], foreground=palette["text"], padding=2)
        style.configure("TRadiobutton", background=palette["surface"], foreground=palette["text"], padding=2)
        style.configure("TNotebook", background=palette["bg"], borderwidth=0, tabmargins=(2, 4, 2, 0))
        style.configure(
            "TNotebook.Tab", background=palette["surface"], foreground=palette["muted"],
            padding=(15, 9), font=family10b, borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", palette["surface_alt"]), ("active", "#252B31")],
            foreground=[("selected", "#F0F2F3"), ("active", palette["text"])],
        )
        style.configure(
            "Treeview", background=palette["surface"], fieldbackground=palette["surface"],
            foreground=palette["text"], rowheight=27, bordercolor=palette["border"],
        )
        style.map(
            "Treeview",
            background=[("selected", palette["selection"])],
            foreground=[("selected", "#F5F7F6")],
        )
        style.configure(
            "Treeview.Heading", background=palette["surface_alt"], foreground=palette["text"],
            font=family10b, relief="flat", padding=(7, 6),
        )
        style.configure(
            "TScrollbar", background=palette["surface_alt"], troughcolor=palette["surface"],
            bordercolor=palette["surface"], arrowcolor=palette["muted"],
        )

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="both", expand=True)

        hero = ttk.Frame(top, style="Card.TFrame", padding=(14, 10))
        hero.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(hero, text="Failure Replay Workbench", style="Title.TLabel").pack(side="left")
        ttk.Label(
            hero,
            text="Deterministic E2E replay  →  Expert / Manual PID  →  correction dataset",
            style="Muted.TLabel",
        ).pack(side="left", padx=16)

        self.notebook = ttk.Notebook(top)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(4, 5))

        self.tab_setup = ttk.Frame(self.notebook)
        self.tab_run = ttk.Frame(self.notebook)
        self.tab_review = ttk.Frame(self.notebook)
        self.tab_manual = ttk.Frame(self.notebook)
        self.tab_recovery = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_setup, text="1  Setup")
        self.notebook.add(self.tab_run, text="2  E2E Probe")
        self.notebook.add(self.tab_review, text="3  Three-Point Case")
        self.notebook.add(self.tab_manual, text="4  Manual Path + PID")
        self.notebook.add(self.tab_recovery, text="5  Recovery / Dataset")

        self._build_setup_tab()
        self._build_run_tab()
        self._build_review_tab()
        from manual_plan_panel import ManualPlanPanel
        self.manual_panel = ManualPlanPanel(self.tab_manual, self)
        self.manual_panel.pack(fill="both", expand=True)
        self._build_recovery_tab()

        log_frame = ttk.LabelFrame(self, text="Process log")
        log_frame.pack(fill="both", expand=False, padx=10, pady=(4, 10))
        toolbar = ttk.Frame(log_frame, style="Card.TFrame")
        toolbar.pack(fill="x", padx=5, pady=4)
        ttk.Button(toolbar, text="Clear", command=self._clear_log).pack(side="left")
        ttk.Button(toolbar, text="Save Log…", command=self._save_log).pack(side="left", padx=5)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(toolbar, textvariable=self.status_var, style="Muted.TLabel").pack(side="right")
        self.log = tk.Text(
            log_frame,
            height=11,
            wrap="word",
            font=(self._ui_fixed_font_family, 10),
            background=self._ui_colors["canvas"],
            foreground="#D8DCE1",
            insertbackground=self._ui_colors["text"],
            selectbackground=self._ui_colors["selection"],
            selectforeground="#F5F7F6",
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=9,
        )
        self.log.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        self.log.configure(state="disabled")

    def _build_setup_tab(self):
        frame = self.tab_setup
        frame.columnconfigure(1, weight=1)

        fields = [
            ("project_root", "Collector project root", "dir"),
            ("carla_root", "CARLA 0.9.15 root", "dir"),
            ("b2d_root", "SparseDriveV2 / Bench2Drive root", "dir"),
            ("e2e_python", "SparseDrive Python executable", "file"),
            ("route_xml", "Bench2Drive route XML", "file"),
            ("e2e_agent", "E2E agent", "file"),
            ("e2e_config", "SparseDrive config", "file"),
            ("e2e_checkpoint", "SparseDrive checkpoint", "file"),
            ("recovery_yaml", "Recovery YAML", "file"),
            ("carla_port", "CARLA port", None),
            ("tm_port", "Traffic Manager port", None),
            ("gpu_rank", "GPU rank", None),
        ]

        for row, (key, label, browse) in enumerate(fields):
            ttk.Label(frame, text=label).grid(
                row=row, column=0, sticky="w", padx=8, pady=5
            )
            var = tk.StringVar()
            self.settings_vars[key] = var
            entry = ttk.Entry(frame, textvariable=var)
            entry.grid(row=row, column=1, sticky="ew", padx=8, pady=5)
            if browse:
                cmd = lambda k=key, b=browse: self._browse_setting(k, b)
                ttk.Button(frame, text="Browse", command=cmd).grid(
                    row=row, column=2, sticky="ew", padx=8, pady=5
                )

        buttons = ttk.Frame(frame)
        buttons.grid(
            row=len(fields), column=0, columnspan=3,
            sticky="ew", padx=8, pady=10
        )
        ttk.Button(
            buttons, text="Auto-fill derived paths",
            command=lambda: self._apply_derived_defaults(only_empty=False)
        ).pack(side="left")
        ttk.Button(
            buttons, text="Save settings",
            command=self._save_settings
        ).pack(side="left", padx=6)
        ttk.Button(
            buttons, text="Check installation",
            command=self.check_installation
        ).pack(side="left")

        self.install_check_var = tk.StringVar(value="")
        ttk.Label(
            frame,
            textvariable=self.install_check_var,
            justify="left",
        ).grid(
            row=len(fields) + 1, column=0, columnspan=3,
            sticky="nw", padx=8, pady=8
        )

        note = (
            "Only machine-specific roots/executables live here. "
            "Route ID, probe, intervention window and output are selected in the other tabs. "
            "The GUI never changes the frozen +1-tick replay rule."
        )
        ttk.Label(
            frame, text=note, wraplength=1000, justify="left"
        ).grid(
            row=len(fields) + 2, column=0, columnspan=3,
            sticky="w", padx=8, pady=8
        )

    def _build_run_tab(self):
        frame = self.tab_run
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Route ID").grid(
            row=0, column=0, sticky="w", padx=8, pady=6
        )
        self.route_var = tk.StringVar()
        self.route_combo = ttk.Combobox(
            frame, textvariable=self.route_var, state="normal"
        )
        self.route_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        self.route_combo.bind("<<ComboboxSelected>>", self._route_changed)
        self.route_combo.bind("<FocusOut>", self._route_changed)
        ttk.Button(
            frame, text="Refresh routes", command=self.refresh_routes
        ).grid(row=0, column=2, padx=8, pady=6)

        self.route_info_var = tk.StringVar(value="")
        ttk.Label(
            frame, textvariable=self.route_info_var, justify="left"
        ).grid(
            row=1, column=0, columnspan=3,
            sticky="w", padx=8, pady=(0, 8)
        )

        carla = ttk.LabelFrame(frame, text="CARLA")
        carla.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=5)
        ttk.Button(carla, text="Start CARLA", command=self.start_carla).pack(
            side="left", padx=6, pady=6
        )
        ttk.Button(
            carla, text="Stop CARLA",
            command=lambda: self.runner.stop("CARLA")
        ).pack(side="left", padx=6, pady=6)

        probe = ttk.LabelFrame(frame, text="Behavior Probe")
        probe.grid(row=3, column=0, columnspan=3, sticky="ew", padx=8, pady=5)
        ttk.Button(
            probe, text="Start Probe Listener", command=self.start_probe
        ).pack(side="left", padx=6, pady=6)
        ttk.Button(
            probe, text="Stop Probe",
            command=lambda: self.runner.stop("PROBE")
        ).pack(side="left", padx=6, pady=6)

        e2e = ttk.LabelFrame(frame, text="Original E2E / SparseDrive")
        e2e.grid(row=4, column=0, columnspan=3, sticky="ew", padx=8, pady=5)
        ttk.Button(e2e, text="Run selected Route", command=self.run_e2e).pack(
            side="left", padx=6, pady=6
        )
        ttk.Button(
            e2e, text="Stop E2E",
            command=lambda: self.runner.stop("E2E")
        ).pack(side="left", padx=6, pady=6)

        probes = ttk.LabelFrame(frame, text="Recorded probes")
        probes.grid(row=5, column=0, columnspan=3, sticky="ew", padx=8, pady=5)
        probes.columnconfigure(1, weight=1)

        ttk.Label(probes, text="Probe").grid(
            row=0, column=0, padx=6, pady=6, sticky="w"
        )
        self.probe_var = tk.StringVar()
        self.probe_combo = ttk.Combobox(
            probes, textvariable=self.probe_var, state="normal"
        )
        self.probe_combo.grid(
            row=0, column=1, padx=6, pady=6, sticky="ew"
        )
        ttk.Button(
            probes, text="Refresh", command=self.refresh_probes
        ).grid(row=0, column=2, padx=6, pady=6)
        ttk.Button(
            probes, text="Validate", command=self.validate_probe
        ).grid(row=0, column=3, padx=6, pady=6)
        ttk.Button(
            probes, text="Load in Review", command=self.load_selected_probe
        ).grid(row=0, column=4, padx=6, pady=6)

        guidance = (
            "Normal order: Start CARLA -> Start Probe Listener -> Run selected Route. "
            "After E2E finishes, validate the new probe, then load it into Review."
        )
        ttk.Label(
            frame, text=guidance, wraplength=1000, justify="left"
        ).grid(
            row=6, column=0, columnspan=3,
            sticky="w", padx=8, pady=10
        )

    def _build_review_tab(self):
        frame = self.tab_review
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        top = ttk.Frame(frame, style="Card.TFrame", padding=8)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Loaded Probe", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self.loaded_probe_var = tk.StringVar(value="None")
        ttk.Label(top, textvariable=self.loaded_probe_var, style="Muted.TLabel").grid(row=0, column=1, sticky="w", padx=8)

        self.current_info_var = tk.StringVar(value="No probe loaded")
        ttk.Label(frame, textvariable=self.current_info_var, justify="left").grid(row=1, column=0, sticky="w", padx=12, pady=(0, 4))

        canvas_frame = ttk.Frame(frame, style="Card.TFrame")
        canvas_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=4)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.traj_canvas = tk.Canvas(canvas_frame, background=self._ui_colors["canvas"], highlightbackground=self._ui_colors["border"], highlightthickness=1)
        self.traj_canvas.grid(row=0, column=0, sticky="nsew")
        self.traj_canvas.bind("<Button-1>", self._canvas_click)
        self.traj_canvas.bind("<Configure>", lambda e: self._draw_trajectory())

        slider_frame = ttk.Frame(frame)
        slider_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=4)
        slider_frame.columnconfigure(0, weight=1)
        self.frame_index_var = tk.IntVar(value=0)
        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=0, orient="horizontal", command=self._slider_changed)
        self.frame_slider.grid(row=0, column=0, sticky="ew")
        ttk.Button(slider_frame, text="−1", command=lambda: self._step_frame(-1)).grid(row=0, column=1, padx=(6, 2))
        ttk.Button(slider_frame, text="+1", command=lambda: self._step_frame(1)).grid(row=0, column=2, padx=2)

        selection = ttk.LabelFrame(frame, text="Three points", padding=8)
        selection.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
        for col in range(4):
            selection.columnconfigure(col, weight=1 if col in (1, 3) else 0)

        ttk.Button(selection, text="Set Record Start", command=lambda: self.set_marker("record_start")).grid(row=0, column=0, padx=5, pady=5)
        self.start_var = tk.StringVar(value="not set")
        ttk.Label(selection, textvariable=self.start_var).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Button(selection, text="Set Handoff", command=lambda: self.set_marker("handoff")).grid(row=0, column=2, padx=5, pady=5)
        self.handoff_var = tk.StringVar(value="not set")
        ttk.Label(selection, textvariable=self.handoff_var).grid(row=0, column=3, sticky="w", padx=5)

        ttk.Button(selection, text="Pick free End on Map", style="Accent.TButton", command=self.begin_pick_planning_end).grid(row=1, column=0, padx=5, pady=7)
        self.planning_end_var = tk.StringVar(value="not set")
        ttk.Label(selection, textvariable=self.planning_end_var).grid(row=1, column=1, columnspan=3, sticky="w", padx=5)
        self.snap_planning_end_var = tk.BooleanVar(value=False)  # retained only for old helper compatibility

        ttk.Label(selection, text="End radius (m)").grid(row=2, column=0, sticky="e", padx=5, pady=4)
        self.end_radius_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(selection, from_=0.3, to=5.0, increment=0.1, textvariable=self.end_radius_var, width=8).grid(row=2, column=1, sticky="w", padx=5)
        ttk.Label(selection, text="Confirm frames").grid(row=2, column=2, sticky="e", padx=5, pady=4)
        self.end_confirm_var = tk.IntVar(value=3)
        ttk.Spinbox(selection, from_=1, to=20, increment=1, textvariable=self.end_confirm_var, width=8).grid(row=2, column=3, sticky="w", padx=5)
        ttk.Label(
            selection,
            text="Start/Handoff are time+space anchors on the original E2E Tape. End is free XY only; it has no E2E timestamp.",
            style="Muted.TLabel",
            wraplength=1050,
        ).grid(row=3, column=0, columnspan=4, sticky="w", padx=5, pady=(4, 2))

        self.reference_info_var = tk.StringVar(value="reference line not loaded")
        ttk.Label(selection, textvariable=self.reference_info_var, style="Muted.TLabel").grid(row=4, column=0, columnspan=4, sticky="w", padx=5, pady=(0, 4))

        case = ttk.LabelFrame(frame, text="Case / replacement controller", padding=8)
        case.grid(row=5, column=0, sticky="ew", padx=10, pady=5)
        case.columnconfigure(1, weight=1)
        ttk.Label(case, text="Case ID").grid(row=0, column=0, sticky="w", padx=5, pady=4)
        self.case_id_var = tk.StringVar(value="")
        ttk.Entry(case, textvariable=self.case_id_var).grid(row=0, column=1, sticky="ew", padx=5, pady=4)
        ttk.Label(case, text="Type").grid(row=0, column=2, sticky="w", padx=5, pady=4)
        self.case_type_var = tk.StringVar(value="slow_decision")
        ttk.Combobox(case, textvariable=self.case_type_var, values=DEFAULT_TYPES, state="normal", width=20).grid(row=0, column=3, padx=5, pady=4)
        ttk.Label(case, text="Description").grid(row=1, column=0, sticky="w", padx=5, pady=4)
        self.case_desc_var = tk.StringVar(value="E2E weakness/failure correction segment")
        ttk.Entry(case, textvariable=self.case_desc_var).grid(row=1, column=1, columnspan=3, sticky="ew", padx=5, pady=4)

        ttk.Label(case, text="After Handoff").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.replacement_mode_var = tk.StringVar(value="pdm_expert")
        modes = ttk.Frame(case, style="Card.TFrame")
        modes.grid(row=2, column=1, columnspan=3, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(modes, text="PDM Expert", value="pdm_expert", variable=self.replacement_mode_var).pack(side="left")
        ttk.Radiobutton(modes, text="Manual Path + PID", value="manual_pid", variable=self.replacement_mode_var).pack(side="left", padx=14)

        buttons = ttk.Frame(case, style="Card.TFrame")
        buttons.grid(row=3, column=0, columnspan=4, sticky="w", padx=5, pady=6)
        ttk.Button(buttons, text="Save Case", style="Accent.TButton", command=self.save_case).pack(side="left")
        ttk.Button(buttons, text="Open Manual Planner", command=self.prepare_manual_planner).pack(side="left", padx=6)
        ttk.Button(buttons, text="Clear points", command=self.clear_markers).pack(side="left", padx=6)

    def _build_recovery_tab(self):
        frame = self.tab_recovery
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="Route ID").grid(row=0, column=0, sticky="w", padx=10, pady=7)
        self.recovery_route_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.recovery_route_var).grid(row=0, column=1, sticky="ew", padx=10, pady=7)
        ttk.Label(frame, text="Reference Probe").grid(row=1, column=0, sticky="w", padx=10, pady=7)
        self.recovery_probe_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.recovery_probe_var).grid(row=1, column=1, sticky="ew", padx=10, pady=7)
        ttk.Button(frame, text="Browse", command=self._browse_recovery_probe).grid(row=1, column=2, padx=10, pady=7)
        ttk.Label(frame, text="Case JSON").grid(row=2, column=0, sticky="w", padx=10, pady=7)
        self.recovery_case_var = tk.StringVar()
        self.case_combo = ttk.Combobox(frame, textvariable=self.recovery_case_var, state="normal")
        self.case_combo.grid(row=2, column=1, sticky="ew", padx=10, pady=7)
        self.case_combo.bind("<<ComboboxSelected>>", self._case_selected)
        ttk.Button(frame, text="Refresh", command=self.refresh_cases).grid(row=2, column=2, padx=10, pady=7)

        config_box = ttk.LabelFrame(frame, text="Recovery preflight", padding=8)
        config_box.grid(row=3, column=0, columnspan=3, sticky="ew", padx=10, pady=7)
        ttk.Button(config_box, text="Write / Update YAML", command=self.update_recovery_yaml).pack(side="left", padx=5, pady=5)
        ttk.Button(config_box, text="Run Preflight", command=self.run_preflight).pack(side="left", padx=5, pady=5)

        run_box = ttk.LabelFrame(frame, text="Replay → replacement → spatial End", padding=8)
        run_box.grid(row=4, column=0, columnspan=3, sticky="ew", padx=10, pady=7)
        ttk.Button(run_box, text="Run Recovery", style="Accent.TButton", command=self.run_recovery).pack(side="left", padx=5, pady=5)
        ttk.Button(run_box, text="Stop Recovery", command=lambda: self.runner.stop("RECOVERY")).pack(side="left", padx=5, pady=5)
        ttk.Label(
            run_box,
            text="The selected Case decides PDM Expert vs Manual PID. Recording stops at the free spatial End; no End timestamp is used.",
            style="Muted.TLabel",
        ).pack(side="left", padx=12)

        data_box = ttk.LabelFrame(frame, text="Dataset", padding=8)
        data_box.grid(row=5, column=0, columnspan=3, sticky="ew", padx=10, pady=7)
        ttk.Button(data_box, text="Validate latest dataset", command=self.validate_latest_dataset).pack(side="left", padx=5, pady=5)
        ttk.Button(data_box, text="Open latest dataset folder", command=self.open_latest_dataset).pack(side="left", padx=5, pady=5)
        ttk.Button(data_box, text="Export latest clip…", command=self.export_latest_dataset).pack(side="left", padx=5, pady=5)

        self.dataset_summary_var = tk.StringVar(value="No dataset inspected yet.")
        ttk.Label(frame, textvariable=self.dataset_summary_var, justify="left", wraplength=1080).grid(row=6, column=0, columnspan=3, sticky="nw", padx=10, pady=10)

    def _default_settings(self):
        project = infer_project_root()
        carla = os.environ.get("CARLA_ROOT", "")
        b2d = os.environ.get("B2D_ROOT", "")
        e2e_python = infer_sparsedrive_python()

        return {
            "project_root": str(project),
            "carla_root": carla,
            "b2d_root": b2d,
            "e2e_python": e2e_python,
            "route_xml": "",
            "e2e_agent": "",
            "e2e_config": "",
            "e2e_checkpoint": "",
            "recovery_yaml": "",
            "carla_port": "23000",
            "tm_port": "23050",
            "gpu_rank": "0",
        }

    def _load_settings(self):
        values = self._default_settings()
        stored = read_json(SETTINGS_PATH, {}) or {}
        values.update(stored)
        for key, var in self.settings_vars.items():
            var.set(str(values.get(key, "")))

    def _save_settings(self):
        data = {
            key: var.get().strip()
            for key, var in self.settings_vars.items()
        }
        write_json(SETTINGS_PATH, data)
        self._log("Saved settings: %s\n" % SETTINGS_PATH)
        self.refresh_everything(silent=True)

    def _apply_derived_defaults(self, only_empty=True):
        def put(key, value):
            if not value:
                return
            if only_empty and self.settings_vars[key].get().strip():
                return
            self.settings_vars[key].set(str(value))

        project = Path(
            self.settings_vars["project_root"].get().strip()
            or infer_project_root()
        ).expanduser()
        put("project_root", project)

        carla = self.settings_vars["carla_root"].get().strip()
        b2d = self.settings_vars["b2d_root"].get().strip()

        if b2d:
            br = Path(b2d).expanduser()
            put("route_xml", br / "leaderboard" / "data" / "bench2drive220.xml")
            put("e2e_agent", br / "team_code" / "sparsedrive_b2d_agent.py")
            put("e2e_config", br / "projects" / "configs" / "sparsedrive_stage2.py")
            put(
                "e2e_checkpoint",
                br / "ckpt" / "sparsedrive_small_b2d_stage2.pth"
            )

        put("recovery_yaml", project / "configs" / "replay_to_pdm.yaml")

        if not self.settings_vars["e2e_python"].get().strip():
            put("e2e_python", infer_sparsedrive_python())

        self.refresh_everything(silent=True)

    def _browse_setting(self, key, mode):
        if mode == "dir":
            value = filedialog.askdirectory()
        else:
            value = filedialog.askopenfilename()
        if value:
            self.settings_vars[key].set(value)
            if key in ("project_root", "b2d_root"):
                self._apply_derived_defaults(only_empty=True)

    def project_root(self):
        return Path(self.settings_vars["project_root"].get().strip()).expanduser()

    def carla_root(self):
        return Path(self.settings_vars["carla_root"].get().strip()).expanduser()

    def b2d_root(self):
        return Path(self.settings_vars["b2d_root"].get().strip()).expanduser()

    def route_xml(self):
        return Path(self.settings_vars["route_xml"].get().strip()).expanduser()

    def recovery_yaml(self):
        return Path(self.settings_vars["recovery_yaml"].get().strip()).expanduser()

    def carla_egg(self):
        return find_carla_egg(self.settings_vars["carla_root"].get().strip())

    def collector_env(self):
        env = os.environ.copy()
        project = self.project_root()
        carla = self.carla_root()
        b2d = self.b2d_root()
        egg = self.carla_egg()

        python_parts = [
            str(project / "src"),
            str(project / "third_party" / "carla_garage" / "team_code"),
        ]
        if b2d:
            python_parts.extend([
                str(b2d),
                str(b2d / "leaderboard"),
                str(b2d / "leaderboard" / "team_code"),
                str(b2d / "scenario_runner"),
            ])
        if carla:
            python_parts.extend([
                str(carla / "PythonAPI"),
                str(carla / "PythonAPI" / "carla"),
            ])
        if egg:
            python_parts.insert(0, egg)

        existing = env.get("PYTHONPATH", "")
        if existing:
            python_parts.append(existing)

        env["PYTHONPATH"] = os.pathsep.join(python_parts)
        env["CARLA_ROOT"] = str(carla)
        env["B2D_ROOT"] = str(b2d)
        env["SCENARIO_RUNNER_ROOT"] = str(b2d / "scenario_runner")
        env["LEADERBOARD_ROOT"] = str(b2d / "leaderboard")
        env["IS_BENCH2DRIVE"] = "True"
        if egg:
            env["CARLA_EGG"] = egg
        return env

    def e2e_env(self):
        env = os.environ.copy()
        carla = self.carla_root()
        b2d = self.b2d_root()
        egg = self.carla_egg()

        parts = [
            egg,
            str(carla / "PythonAPI" / "carla"),
            str(carla / "PythonAPI"),
            str(b2d),
            str(b2d / "leaderboard"),
            str(b2d / "leaderboard" / "team_code"),
            str(b2d / "scenario_runner"),
        ]
        parts = [x for x in parts if x]
        env["PYTHONPATH"] = os.pathsep.join(parts)
        env["CARLA_ROOT"] = str(carla)
        env["SCENARIO_RUNNER_ROOT"] = str(b2d / "scenario_runner")
        env["LEADERBOARD_ROOT"] = str(b2d / "leaderboard")
        env["IS_BENCH2DRIVE"] = "True"
        return env

    def check_installation(self):
        project = self.project_root()
        carla = self.carla_root()
        b2d = self.b2d_root()

        checks = [
            ("project", project.is_dir(), project),
            ("CARLA", carla.is_dir(), carla),
            ("B2D", b2d.is_dir(), b2d),
            ("CARLA egg", bool(self.carla_egg()), self.carla_egg() or "NOT FOUND"),
            ("route XML", self.route_xml().is_file(), self.route_xml()),
            ("E2E python", Path(self.settings_vars["e2e_python"].get().strip()).is_file(),
             self.settings_vars["e2e_python"].get().strip()),
            ("probe listener", (project / "tools" / "e2e_probe_listener.py").is_file(),
             project / "tools" / "e2e_probe_listener.py"),
            ("probe validator", (project / "tools" / "validate_probe.py").is_file(),
             project / "tools" / "validate_probe.py"),
            ("recovery preflight", (project / "tools" / "preflight_recovery.py").is_file(),
             project / "tools" / "preflight_recovery.py"),
            ("recovery runner", (project / "scripts" / "run_replay_to_pdm.sh").is_file(),
             project / "scripts" / "run_replay_to_pdm.sh"),
            ("recovery agent", (project / "replay_to_pdm_agent.py").is_file(),
             project / "replay_to_pdm_agent.py"),
        ]

        lines = []
        ok_all = True
        for name, ok, value in checks:
            ok_all = ok_all and ok
            lines.append("%-20s %s  %s" % (
                name, "OK" if ok else "MISSING", value
            ))

        if yaml is None:
            ok_all = False
            lines.append("%-20s MISSING  PyYAML import failed" % "PyYAML")

        self.install_check_var.set("\n".join(lines))
        self.status_var.set("Installation OK" if ok_all else "Check missing items")
        return ok_all

    # ------------------------------------------------------------------
    # Route list / launch
    # ------------------------------------------------------------------

    def refresh_everything(self, silent=False):
        try:
            self.refresh_routes()
            self.refresh_probes()
            self.refresh_cases()
            if not silent:
                self._log("Refreshed routes, probes and cases.\n")
        except Exception as exc:
            if not silent:
                self._error("Refresh failed", exc)

    def refresh_routes(self):
        path = self.route_xml()
        records = {}
        if path.is_file():
            root = ET.parse(str(path)).getroot()
            for route in root.iter("route"):
                rid = str(route.attrib.get("id", "")).strip()
                if not rid:
                    continue
                record = dict(route.attrib)
                # Some route files hold scenario/description info in children.
                scenario_names = []
                for elem in route.iter():
                    if elem is route:
                        continue
                    for key in ("type", "name", "scenario_type"):
                        value = elem.attrib.get(key)
                        if value and value not in scenario_names:
                            scenario_names.append(value)
                if scenario_names:
                    record["_children"] = ", ".join(scenario_names[:5])
                records[rid] = record

        self.route_records = records

        def sort_key(x):
            try:
                return (0, int(x))
            except Exception:
                return (1, x)

        values = sorted(records.keys(), key=sort_key)
        self.route_combo["values"] = values
        if not self.route_var.get() and values:
            self.route_var.set(values[0])
        self._route_changed()

    def _route_changed(self, event=None):
        rid = self.route_var.get().strip()
        rec = self.route_records.get(rid, {})
        details = []
        for key in ("town", "map", "name", "weather"):
            if rec.get(key):
                details.append("%s=%s" % (key, rec[key]))
        if rec.get("_children"):
            details.append("scenario=%s" % rec["_children"])
        self.route_info_var.set("  ".join(details))

        self.recovery_route_var.set(rid)
        if rid and not self.case_id_var.get().strip():
            self.case_id_var.set(self.suggest_case_id(rid))

        if hasattr(self, "reference_info_var"):
            self._reload_reference_line()

    def start_carla(self):
        root = self.carla_root()
        script = root / "CarlaUE4.sh"
        if not script.is_file():
            return self._error("CARLA", "Missing %s" % script)

        port = self.settings_vars["carla_port"].get().strip() or "23000"
        cmd = [
            str(script),
            "-carla-port=%s" % port,
            "-quality-level=Low",
        ]
        try:
            self.runner.start("CARLA", cmd, cwd=root, env=os.environ.copy())
        except Exception as exc:
            self._error("Start CARLA failed", exc)

    def start_probe(self):
        project = self.project_root()
        script = project / "tools" / "e2e_probe_listener.py"
        if not script.is_file():
            return self._error("Probe", "Missing %s" % script)

        port = self.settings_vars["carla_port"].get().strip() or "23000"
        cmd = [
            sys.executable,
            "-u",
            str(script),
            "--host", "127.0.0.1",
            "--port", port,
            "--output-root", str(project / "probe_runs"),
            "--ego-role", "hero",
            "--no-native-recorder",
        ]
        try:
            self.runner.start(
                "PROBE", cmd, cwd=project, env=self.collector_env(),
                callback=lambda rc: self.after(300, self.refresh_probes),
            )
        except Exception as exc:
            self._error("Start Probe failed", exc)

    def run_e2e(self):
        rid = self.route_var.get().strip()
        if not rid:
            return self._error("E2E", "Select a Route ID first.")

        b2d = self.b2d_root()
        py = Path(self.settings_vars["e2e_python"].get().strip()).expanduser()
        evaluator = b2d / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"
        route_xml = self.route_xml()
        agent = Path(self.settings_vars["e2e_agent"].get().strip()).expanduser()
        config = Path(self.settings_vars["e2e_config"].get().strip()).expanduser()
        ckpt = Path(self.settings_vars["e2e_checkpoint"].get().strip()).expanduser()

        for label, path in [
            ("SparseDrive Python", py),
            ("leaderboard evaluator", evaluator),
            ("route XML", route_xml),
            ("E2E agent", agent),
            ("E2E config", config),
            ("E2E checkpoint", ckpt),
        ]:
            if not path.is_file():
                return self._error("E2E", "%s not found:\n%s" % (label, path))

        result_dir = b2d / "close_loop_log" / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        run_name = "probe_%s_%s" % (rid, now_stamp())
        checkpoint = result_dir / ("%s.json" % run_name)
        agent_config = "%s+%s+%s+0" % (config, ckpt, run_name)

        cmd = [
            str(py),
            str(evaluator),
            "--host=127.0.0.1",
            "--port=%s" % (self.settings_vars["carla_port"].get().strip() or "23000"),
            "--traffic-manager-port=%s" % (
                self.settings_vars["tm_port"].get().strip() or "23050"
            ),
            "--traffic-manager-seed=0",
            "--routes=%s" % route_xml,
            "--routes-subset=%s" % rid,
            "--repetitions=1",
            "--track=SENSORS",
            "--checkpoint=%s" % checkpoint,
            "--agent=%s" % agent,
            "--agent-config=%s" % agent_config,
            "--debug=0",
            "--gpu-rank=%s" % (
                self.settings_vars["gpu_rank"].get().strip() or "0"
            ),
        ]

        try:
            self.runner.start(
                "E2E", cmd, cwd=b2d, env=self.e2e_env(),
                callback=lambda rc: self.after(500, self.refresh_probes),
            )
        except Exception as exc:
            self._error("Run E2E failed", exc)

    # ------------------------------------------------------------------
    # Probe loading / review
    # ------------------------------------------------------------------

    def refresh_probes(self):
        root = self.project_root() / "probe_runs"
        probes = []
        if root.is_dir():
            probes = sorted(
                [p for p in root.glob("probe_*") if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        values = [str(p.resolve()) for p in probes]
        self.probe_combo["values"] = values
        if values and not self.probe_var.get().strip():
            self.probe_var.set(values[0])

    def validate_probe(self):
        probe = Path(self.probe_var.get().strip()).expanduser()
        script = self.project_root() / "tools" / "validate_probe.py"
        if not probe.is_dir():
            return self._error("Validate Probe", "Select a valid probe directory.")
        if not script.is_file():
            return self._error("Validate Probe", "Missing %s" % script)

        cmd = [sys.executable, str(script), str(probe)]
        try:
            self.runner.start(
                "VALIDATE_PROBE", cmd,
                cwd=self.project_root(),
                env=self.collector_env(),
            )
        except Exception as exc:
            self._error("Validate Probe failed", exc)

    def load_selected_probe(self):
        probe = Path(self.probe_var.get().strip()).expanduser()
        if not probe.is_dir():
            return self._error("Load Probe", "Select a valid probe directory.")
        try:
            self.load_probe(probe)
            self.notebook.select(self.tab_review)
        except Exception as exc:
            self._error("Load Probe failed", exc)

    def load_probe(self, probe):
        path = Path(probe) / "frames.jsonl"
        if not path.is_file():
            raise RuntimeError("Missing frames.jsonl: %s" % path)

        frames = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)

                t = as_float(nested_get(d, ["world", "elapsed_seconds"], i * 0.1))
                tr = nested_get(d, ["ego", "state", "transform"], {}) or {}
                loc = tr.get("location", {}) if isinstance(tr, dict) else {}
                rot = tr.get("rotation", {}) if isinstance(tr, dict) else {}

                x = as_float(loc.get("x", 0.0))
                y = as_float(loc.get("y", 0.0))
                z = as_float(loc.get("z", 0.0))
                yaw = as_float(rot.get("yaw", 0.0))

                vel = nested_get(d, ["ego", "state", "velocity"], {}) or {}
                speed = math.sqrt(
                    as_float(vel.get("x", 0.0)) ** 2
                    + as_float(vel.get("y", 0.0)) ** 2
                    + as_float(vel.get("z", 0.0)) ** 2
                )

                ctrl = nested_get(d, ["ego", "applied_control"], {}) or {}
                frames.append({
                    "row": d,
                    "sample_index": int(d.get("sample_index", i)),
                    "t": t,
                    "x": x,
                    "y": y,
                    "z": z,
                    "yaw": yaw,
                    "speed": speed,
                    "throttle": as_float(ctrl.get("throttle", 0.0)),
                    "brake": as_float(ctrl.get("brake", 0.0)),
                    "steer": as_float(ctrl.get("steer", 0.0)),
                })

        if not frames:
            raise RuntimeError("Probe contains no frames.")

        self.frames = frames
        self.loaded_probe = Path(probe).resolve()
        self.loaded_probe_var.set(str(self.loaded_probe))
        self.recovery_probe_var.set(str(self.loaded_probe))

        self.clear_markers(redraw=False)
        self.clear_planning_end(redraw=False)
        self._reload_reference_line(redraw=False)
        self.frame_slider.configure(from_=0, to=max(0, len(frames) - 1))
        self.frame_slider.set(0)
        self._update_current_info(0)

        rid = self.route_var.get().strip()
        if rid:
            self.case_id_var.set(self.suggest_case_id(rid))

        self._draw_trajectory()
        self._log(
            "Loaded probe: %s (%d frames, %.3f .. %.3f s)\n"
            % (self.loaded_probe, len(frames), frames[0]["t"], frames[-1]["t"])
        )

    def _slider_changed(self, value):
        if not self.frames:
            return
        idx = int(round(float(value)))
        idx = max(0, min(idx, len(self.frames) - 1))
        self.frame_index_var.set(idx)
        self._update_current_info(idx)
        self._draw_trajectory()

    def _step_frame(self, delta):
        if not self.frames:
            return
        idx = self.frame_index_var.get() + int(delta)
        idx = max(0, min(idx, len(self.frames) - 1))
        self.frame_slider.set(idx)
        self._slider_changed(idx)

    def _update_current_info(self, idx):
        if not self.frames:
            self.current_info_var.set("No probe loaded")
            return
        f = self.frames[idx]
        rel = f["t"] - self.frames[0]["t"]
        self.current_info_var.set(
            "frame=%d / %d    sim_time=%.3f s    relative=%.3f s    "
            "speed=%.3f m/s    throttle=%.3f    brake=%.3f    steer=%.3f    "
            "x=%.2f  y=%.2f  yaw=%.2f"
            % (
                idx,
                len(self.frames) - 1,
                f["t"],
                rel,
                f["speed"],
                f["throttle"],
                f["brake"],
                f["steer"],
                f["x"],
                f["y"],
                f["yaw"],
            )
        )

    def _trajectory_transform(self):
        if not self.frames and not self.reference_line:
            return None

        width = max(100, self.traj_canvas.winfo_width())
        height = max(100, self.traj_canvas.winfo_height())
        pad = 45

        world_points = []
        world_points.extend(
            (float(f["x"]), float(f["y"])) for f in self.frames
        )
        world_points.extend(
            (float(p["x"]), float(p["y"])) for p in self.reference_line
        )
        if self.planning_end_point is not None:
            world_points.append(
                (
                    float(self.planning_end_point["x"]),
                    float(self.planning_end_point["y"]),
                )
            )

        if not world_points:
            return None

        # CARLA/Unreal top-down convention used by the operator view:
        #   +X -> screen up (North-like), +Y -> screen right (East-like).
        # The old Review canvas used world X as horizontal and world Y as
        # vertical, which rotates a CARLA east-west road into north-south.
        # Only this Three-Point Case view is re-oriented; stored world XY and
        # every replay/control path remain untouched.
        us = [p[1] for p in world_points]  # screen horizontal <- world Y
        vs = [p[0] for p in world_points]  # screen vertical   <- world X

        umin, umax = min(us), max(us)
        vmin, vmax = min(vs), max(vs)
        du = max(umax - umin, 1e-6)
        dv = max(vmax - vmin, 1e-6)

        sx = (width - 2 * pad) / du
        sy = (height - 2 * pad) / dv
        scale = max(min(sx, sy), 1e-9)

        draw_w = du * scale
        draw_h = dv * scale
        xoff = (width - draw_w) / 2.0
        yoff = (height - draw_h) / 2.0

        def convert(x, y):
            u = float(y)
            v = float(x)
            cx = xoff + (u - umin) * scale
            cy = height - (yoff + (v - vmin) * scale)
            return cx, cy

        def invert(cx, cy):
            u = ((float(cx) - xoff) / scale) + umin
            v = (((height - float(cy)) - yoff) / scale) + vmin
            return v, u  # world x, world y

        return convert, invert

    def _draw_trajectory(self):
        c = self.traj_canvas
        c.delete("all")
        self.canvas_points = []
        self.reference_canvas_points = []
        palette = getattr(self, "_ui_colors", {})
        text_color = palette.get("text", "#E7E9EC")
        muted_color = palette.get("muted", "#9EA7B1")
        if not self.frames and not self.reference_line:
            c.create_text(max(50, c.winfo_width()/2), max(50, c.winfo_height()/2), text="Load a Probe to review the E2E trajectory.", fill=muted_color, font=(self._ui_font_family, 11))
            return
        transform = self._trajectory_transform()
        if transform is None:
            return
        convert, _invert = transform
        if self.reference_line:
            ref = [convert(p["x"], p["y"]) for p in self.reference_line]
            self.reference_canvas_points = ref
            if len(ref) >= 2:
                flat = [v for pt in ref for v in pt]
                c.create_line(*flat, fill="#78A6A1", width=2, dash=(7, 5))
        if self.frames:
            pts = [convert(f["x"], f["y"]) for f in self.frames]
            self.canvas_points = pts
            if len(pts) >= 2:
                c.create_line(*[v for pt in pts for v in pt], fill="#A1A8B0", width=3)
            styles = {
                "record_start": ("#7FA5C9", "Record Start"),
                "handoff": ("#D2AE72", "Handoff"),
            }
            for key, (color, label) in styles.items():
                idx = self.marker_indices.get(key)
                if idx is None or idx >= len(pts):
                    continue
                x, y = pts[idx]
                c.create_oval(x-7, y-7, x+7, y+7, fill=color, outline="#ECEEEF", width=1)
                c.create_text(x+10, y+10, text="%s  %.2fs" % (label, self.frames[idx]["t"]), fill=color, anchor="w", font=(self._ui_font_family, 9, "bold"))
            idx = max(0, min(self.frame_index_var.get(), len(pts)-1))
            x, y = pts[idx]
            c.create_oval(x-5, y-5, x+5, y+5, fill="#C97D7D", outline="#ECEEEF")
        if self.planning_end_point is not None:
            x, y = convert(self.planning_end_point["x"], self.planning_end_point["y"])
            c.create_oval(x-9, y-9, x+9, y+9, fill="#A68BB3", outline="#ECEEEF", width=2)
            c.create_line(x-13, y, x+13, y, fill="#ECEEEF")
            c.create_line(x, y-13, x, y+13, fill="#ECEEEF")
            c.create_text(x+12, y+14, text="Free End (no time)", fill="#BBA4C5", anchor="w", font=(self._ui_font_family, 9, "bold"))

        # Compact orientation cue matching CARLA's top-down world axes.
        compass_x = max(70, c.winfo_width() - 72)
        compass_y = 70
        c.create_line(compass_x, compass_y, compass_x, compass_y-28, fill="#AAB1B8", width=2, arrow="last")
        c.create_line(compass_x, compass_y, compass_x+28, compass_y, fill="#AAB1B8", width=2, arrow="last")
        c.create_text(compass_x, compass_y-37, text="+X", fill=muted_color, font=(self._ui_font_family, 8, "bold"))
        c.create_text(compass_x+38, compass_y, text="+Y", fill=muted_color, font=(self._ui_font_family, 8, "bold"))

        c.create_text(12, 11, text="E2E    Route ref.    Record Start    Handoff    Free End", fill=text_color, anchor="nw", font=(self._ui_font_family, 10, "bold"))
        c.create_text(12, 32, text="CARLA-aligned view: +Y → right, +X → up.  Click E2E to select a frame; Free End remains arbitrary world XY.", fill=muted_color, anchor="nw", font=(self._ui_font_family, 9))

    def _canvas_click(self, event):
        if self.pick_planning_end_active:
            self._pick_planning_end_from_canvas(event)
            return

        if not self.canvas_points:
            return
        best_idx = 0
        best_d2 = None
        for i, (x, y) in enumerate(self.canvas_points):
            d2 = (x - event.x) ** 2 + (y - event.y) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_idx = i
        self.frame_slider.set(best_idx)
        self._slider_changed(best_idx)

    def _reload_reference_line(self, redraw=True):
        rid = self.route_var.get().strip() if hasattr(self, "route_var") else ""
        path = self.route_xml()

        try:
            points = load_route_reference_line(path, rid) if rid else []
        except Exception as exc:
            points = []
            self._log("Reference line load failed: %s\\n" % exc)

        self.reference_line = points
        self.reference_s = cumulative_xy_distance(points)

        if hasattr(self, "reference_info_var"):
            if points:
                length = self.reference_s[-1] if self.reference_s else 0.0
                self.reference_info_var.set(
                    "Route %s: %d XML waypoints, polyline length %.1f m"
                    % (rid, len(points), length)
                )
            elif rid:
                self.reference_info_var.set(
                    "Route %s: no ./waypoints/position found in route XML" % rid
                )
            else:
                self.reference_info_var.set("route not selected")

        if redraw and hasattr(self, "traj_canvas"):
            self._draw_trajectory()

    def begin_pick_planning_end(self):
        if not self.frames:
            return self._error("End Point", "Load a probe first.")
        self.pick_planning_end_active = True
        try:
            self.traj_canvas.configure(cursor="crosshair")
        except Exception:
            pass
        self.planning_end_var.set("click any XY location on the map…")

    def _pick_planning_end_from_canvas(self, event):
        transform = self._trajectory_transform()
        if transform is None:
            return
        _convert, invert = transform
        x, y = invert(event.x, event.y)
        self.planning_end_point = {"x": float(x), "y": float(y)}
        self.pick_planning_end_active = False
        try:
            self.traj_canvas.configure(cursor="")
        except Exception:
            pass
        self._update_planning_end_label()
        self._draw_trajectory()

    def _update_planning_end_label(self):
        if not hasattr(self, "planning_end_var"):
            return
        point = self.planning_end_point
        if point is None:
            self.planning_end_var.set("not set")
            return
        text = "End XY = (%.2f, %.2f)" % (float(point["x"]), float(point["y"]))
        hidx = self.marker_indices.get("handoff")
        if hidx is not None and self.frames:
            h = self.frames[hidx]
            straight = math.hypot(float(point["x"]) - h["x"], float(point["y"]) - h["y"])
            text += "    straight-line from Handoff = %.1f m" % straight
        self.planning_end_var.set(text)

    def clear_planning_end(self, redraw=True):
        self.planning_end_point = None
        self.pick_planning_end_active = False
        try:
            if hasattr(self, "traj_canvas"):
                self.traj_canvas.configure(cursor="")
        except Exception:
            pass
        self._update_planning_end_label()
        if redraw and hasattr(self, "traj_canvas"):
            self._draw_trajectory()

    def _planning_payload(self):
        if self.planning_end_point is None:
            raise RuntimeError("Planning End is not set.")

        handoff_idx = self.marker_indices.get("handoff")
        if handoff_idx is None:
            raise RuntimeError(
                "Set Handoff first. The spatial planning segment starts at Handoff."
            )

        handoff = self.frames[handoff_idx]
        payload = {
            "schema": "b2d-manual-plan-v1",
            "status": "endpoint_only",
            "case_id": self.case_id_var.get().strip(),
            "route_id": self.route_var.get().strip(),
            "source_run": (
                str(self.loaded_probe)
                if self.loaded_probe is not None else ""
            ),
            "reference_line_source": {
                "type": "bench2drive_route_xml_waypoints",
                "route_xml": str(self.route_xml().resolve()),
                "route_id": self.route_var.get().strip(),
            },
            "planning_start": {
                "source": "handoff",
                "sample_index": int(handoff["sample_index"]),
                "sim_time_s": float(handoff["t"]),
                "x": float(handoff["x"]),
                "y": float(handoff["y"]),
                "z": float(handoff["z"]),
                "yaw": float(handoff["yaw"]),
            },
            "planning_end": dict(self.planning_end_point),
            "reference_line": [
                {
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "z": float(p.get("z", 0.0)),
                    "yaw": p.get("yaw"),
                    "s_m": (
                        float(self.reference_s[i])
                        if i < len(self.reference_s) else None
                    ),
                }
                for i, p in enumerate(self.reference_line)
            ],
        }

        end_idx = self.marker_indices.get("record_end")
        if end_idx is not None:
            payload["record_end_time_s"] = float(
                self.frames[end_idx]["t"]
            )
        return payload

    def _save_planning_end_file(self, silent=False):
        case_id = self.case_id_var.get().strip()
        if not case_id:
            raise RuntimeError("Case ID is empty.")

        payload = self._planning_payload()
        root = self.project_root() / "cases" / "plans"
        root.mkdir(parents=True, exist_ok=True)
        path = root / ("%s.plan.json" % case_id)
        write_json(path, payload)

        if not silent:
            self._log(
                "Saved planning endpoint: %s\\n"
                "  start=handoff %.3f s\\n"
                "  end=(%.2f, %.2f)\\n"
                % (
                    path,
                    payload["planning_start"]["sim_time_s"],
                    payload["planning_end"]["x"],
                    payload["planning_end"]["y"],
                )
            )
        return path

    def save_planning_end(self):
        try:
            path = self._save_planning_end_file(silent=False)
            self.status_var.set(
                "Planning endpoint saved: %s" % path.name
            )
        except Exception as exc:
            self._error("Save Planning End", exc)

    def set_marker(self, key):
        if not self.frames:
            return self._error("Marker", "Load a probe first.")
        idx = self.frame_index_var.get()
        self.marker_indices[key] = idx
        self._update_marker_labels()
        self._update_planning_end_label()
        self._draw_trajectory()

    def clear_markers(self, redraw=True):
        self.marker_indices = {"record_start": None, "handoff": None, "record_end": None}
        self.planning_end_point = None
        self.pick_planning_end_active = False
        self._update_marker_labels()
        self._update_planning_end_label()
        if redraw:
            self._draw_trajectory()

    def _update_marker_labels(self):
        for key, var in (("record_start", self.start_var), ("handoff", self.handoff_var)):
            idx = self.marker_indices.get(key)
            if idx is None or not self.frames:
                var.set("not set")
            else:
                f = self.frames[idx]
                var.set("frame %d  |  t=%.3f s  |  (%.2f, %.2f)" % (idx, f["t"], f["x"], f["y"]))

    def suggest_case_id(self, route_id):
        cases = self.project_root() / "cases"
        cases.mkdir(parents=True, exist_ok=True)
        prefix = "case_%s_" % route_id
        used = []
        for path in cases.glob(prefix + "*.json"):
            stem = path.stem
            tail = stem[len(prefix):]
            try:
                used.append(int(tail))
            except Exception:
                pass
        next_id = max(used or [0]) + 1
        return "%s%03d" % (prefix, next_id)

    def save_case(self):
        if not self.frames or self.loaded_probe is None:
            return self._error("Save Case", "Load a probe first.")
        a = self.marker_indices.get("record_start")
        b = self.marker_indices.get("handoff")
        if a is None or b is None:
            return self._error("Save Case", "Set Record Start and Handoff first.")
        if a > b:
            return self._error("Save Case", "Required order: Record Start <= Handoff.")
        if self.planning_end_point is None:
            return self._error("Save Case", "Pick the free spatial End point first.")
        case_id = self.case_id_var.get().strip() or self.suggest_case_id(self.route_var.get().strip())
        self.case_id_var.set(case_id)
        def point(i):
            fr=self.frames[i]
            return {
                "sample_index": int(fr.get("sample_index", i)),
                "relative_time_s": float(fr.get("relative_time_s", fr["t"] - self.frames[0]["t"])),
                "sim_time_s": float(fr["t"]),
                "x": float(fr["x"]), "y": float(fr["y"]), "z": float(fr.get("z", 0.0)),
                "yaw": float(fr.get("yaw", 0.0)), "speed_mps": float(fr.get("speed", 0.0)),
            }
        mode = self.replacement_mode_var.get().strip() or "pdm_expert"
        replacement = {"mode": mode}
        if mode == "manual_pid":
            replacement["plan"] = "plans/%s.plan.json" % case_id
        payload = {
            "schema": "b2d-intervention-v2",
            "case_id": case_id,
            "source_run": str(self.loaded_probe.resolve()),
            "route_id": self.route_var.get().strip(),
            "intervention_type": self.case_type_var.get().strip() or "manual_intervention",
            "description": self.case_desc_var.get().strip(),
            "record_start": point(a),
            "handoff": point(b),
            "end": {
                "x": float(self.planning_end_point["x"]),
                "y": float(self.planning_end_point["y"]),
                "radius_m": float(self.end_radius_var.get()),
                "confirm_frames": int(self.end_confirm_var.get()),
            },
            "replacement": replacement,
            "replay": {
                "time_offset_seconds": 0.0,
                "max_position_error_m": 2.0,
                "max_yaw_error_deg": 10.0,
                "divergence_grace_seconds": 1.0,
                "abort_on_divergence": True,
            },
        }
        path = self.project_root() / "cases" / (case_id + ".json")
        write_json(path, payload)
        self.recovery_case_var.set(str(path.resolve()))
        self.recovery_probe_var.set(str(self.loaded_probe))
        self.recovery_route_var.set(self.route_var.get().strip())
        self._log(
            "Saved v2 case: %s\n  record_start=%.3f  handoff=%.3f\n"
            "  End=(%.2f, %.2f), r=%.2fm, confirm=%d\n  replacement=%s\n"
            % (path, payload["record_start"]["sim_time_s"], payload["handoff"]["sim_time_s"],
               payload["end"]["x"], payload["end"]["y"], payload["end"]["radius_m"],
               payload["end"]["confirm_frames"], mode)
        )
        self.refresh_cases()
        if mode == "manual_pid":
            self.prepare_manual_planner()
        else:
            self.notebook.select(self.tab_recovery)

    def prepare_manual_planner(self):
        if self.replacement_mode_var.get().strip() != "manual_pid":
            self.replacement_mode_var.set("manual_pid")
        if not self.frames or self.marker_indices.get("handoff") is None or self.planning_end_point is None:
            return self._error("Manual Planner", "Load Probe, set Handoff and pick End first.")
        try:
            self.manual_panel.load_from_workbench()
            self.notebook.select(self.tab_manual)
        except Exception as exc:
            self._error("Manual Planner", exc)

    def refresh_cases(self):
        root = self.project_root() / "cases"
        cases = []
        if root.is_dir():
            rid = self.route_var.get().strip()
            pattern = "case_%s_*.json" % rid if rid else "case_*.json"
            cases = sorted(
                root.glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            # If filtering yields nothing, still show all cases.
            if not cases:
                cases = sorted(
                    root.glob("case_*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )

        values = [str(p.resolve()) for p in cases]
        self.case_combo["values"] = values

    def _case_selected(self, event=None):
        path = Path(self.recovery_case_var.get().strip()).expanduser()
        data = read_json(path, {})
        if not data:
            return
        source = data.get("source_run")
        if source:
            self.recovery_probe_var.set(str(source))
        case_id = str(data.get("case_id", ""))
        parts = case_id.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            self.recovery_route_var.set(parts[1])
            self.route_var.set(parts[1])
            self._route_changed()

    def _browse_recovery_probe(self):
        value = filedialog.askdirectory(
            initialdir=str(self.project_root() / "probe_runs")
        )
        if value:
            self.recovery_probe_var.set(value)

    def update_recovery_yaml(self):
        if yaml is None:
            return self._error("Recovery YAML", "PyYAML is not importable in this Python environment.")
        project = self.project_root()
        target = self.recovery_yaml()
        template = project / "configs" / "replay_to_pdm.example.yaml"
        if target.is_file():
            with target.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        elif template.is_file():
            with template.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            return self._error("Recovery YAML", "Neither recovery YAML nor example exists:\n%s\n%s" % (target, template))
        rid = self.recovery_route_var.get().strip() or self.route_var.get().strip()
        probe = Path(self.recovery_probe_var.get().strip()).expanduser()
        case = Path(self.recovery_case_var.get().strip()).expanduser()
        if not rid:
            return self._error("Recovery YAML", "Route ID is empty.")
        if not probe.is_dir():
            return self._error("Recovery YAML", "Invalid probe:\n%s" % probe)
        if not case.is_file():
            return self._error("Recovery YAML", "Invalid case:\n%s" % case)
        case_obj = read_json(case, {}) or {}
        mode = nested_get(case_obj, ["replacement", "mode"], "pdm_expert")
        cfg["route_xml"] = str(self.route_xml().resolve())
        cfg["route_id"] = str(rid)
        cfg["output_root"] = str((project / "outputs_recovery_data").resolve())
        replay = cfg.setdefault("replay", {})
        replay["tape_dir"] = str(probe.resolve())
        replay["intervention_spec"] = str(case.resolve())
        replay["shadow_pdm"] = (mode == "pdm_expert")
        replay["sensor_warmup_seconds"] = 1.0
        replay["abort_after_tape"] = True
        replay["require_complete_tape"] = True
        replay["terminate_on_end"] = True
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        self._log(
            "Updated recovery YAML: %s\n  route=%s\n  probe=%s\n  case=%s\n  mode=%s\n"
            % (target, rid, probe, case, mode)
        )

    def run_preflight(self):
        if not self.update_recovery_yaml():
            return

        script = self.project_root() / "tools" / "preflight_recovery.py"
        if not script.is_file():
            return self._error("Preflight", "Missing %s" % script)

        cmd = [sys.executable, str(script), str(self.recovery_yaml())]
        try:
            self.runner.start(
                "PREFLIGHT", cmd,
                cwd=self.project_root(),
                env=self.collector_env(),
            )
        except Exception as exc:
            self._error("Preflight failed", exc)

    def run_recovery(self):
        if not self.update_recovery_yaml():
            return

        rid = self.recovery_route_var.get().strip() or self.route_var.get().strip()
        project = self.project_root()
        script = project / "scripts" / "run_replay_to_pdm.sh"
        if not script.is_file():
            return self._error("Recovery", "Missing %s" % script)

        egg = self.carla_egg()
        if not egg:
            return self._error(
                "Recovery",
                "Cannot find a Python 3.7 CARLA egg/wheel.\n"
                "Set the correct CARLA root in Setup."
            )

        (project / "outputs_recovery").mkdir(parents=True, exist_ok=True)
        (project / "outputs_recovery_data").mkdir(parents=True, exist_ok=True)
        (project / "logs").mkdir(parents=True, exist_ok=True)

        env = self.collector_env()
        env["CARLA_EGG"] = egg
        env["PORT"] = self.settings_vars["carla_port"].get().strip() or "23000"
        env["TM_PORT"] = self.settings_vars["tm_port"].get().strip() or "23050"
        env["TM_SEED"] = "0"
        env["CHECKPOINT"] = str(
            project / "outputs_recovery" / ("recovery_%s.json" % rid)
        )

        cmd = [
            "bash",
            str(script),
            str(self.recovery_yaml()),
            str(self.route_xml()),
            str(rid),
        ]

        try:
            self.runner.start(
                "RECOVERY", cmd,
                cwd=project,
                env=env,
                callback=self._recovery_finished,
            )
        except Exception as exc:
            self._error("Recovery failed", exc)

    def _recovery_finished(self, rc):
        self.refresh_probes()
        self._find_latest_clip()
        if rc == 0:
            self._log(
                "\nRecovery process exited with 0. "
                "Use 'Validate latest dataset' for the final data check.\n"
            )
        else:
            self._log("\nRecovery process exited with %d.\n" % rc)

    # ------------------------------------------------------------------
    # Dataset validation / export
    # ------------------------------------------------------------------

    def _find_latest_clip(self):
        root = self.project_root() / "outputs_recovery_data"
        if not root.is_dir():
            self.latest_clip = None
            return None
        dirs = [p for p in root.iterdir() if p.is_dir()]
        if not dirs:
            self.latest_clip = None
            return None
        self.latest_clip = max(dirs, key=lambda p: p.stat().st_mtime)
        return self.latest_clip

    def validate_latest_dataset(self):
        clip = self._find_latest_clip()
        if clip is None:
            return self._error(
                "Dataset",
                "No dataset directory found under outputs_recovery_data."
            )

        self._update_dataset_summary(clip)

        cmd = [
            sys.executable,
            "-m",
            "b2d_collector.validate_dataset",
            str(clip),
        ]
        try:
            self.runner.start(
                "VALIDATE_DATASET", cmd,
                cwd=self.project_root(),
                env=self.collector_env(),
            )
        except Exception as exc:
            self._error("Dataset validation failed", exc)

    def _update_dataset_summary(self, clip):
        measurements = Path(clip) / "measurements"
        files = sorted(measurements.glob("*.json.gz"))
        rows = []
        for path in files:
            try:
                with gzip.open(str(path), "rt", encoding="utf-8") as f:
                    rows.append(json.load(f))
            except Exception as exc:
                self._log("Could not read %s: %s\n" % (path, exc))

        if not rows:
            self.dataset_summary_var.set(
                "Latest clip: %s\nNo readable measurement frames." % clip
            )
            return

        times = [as_float(r.get("timestamp")) for r in rows]
        sources = [
            nested_get(r, ["collector", "control_source"], "unknown")
            for r in rows
        ]

        transitions = []
        previous = None
        for r in rows:
            source = nested_get(r, ["collector", "control_source"], "unknown")
            if source != previous:
                transitions.append(
                    (
                        int(r.get("frame_index", len(transitions))),
                        as_float(r.get("timestamp")),
                        source,
                    )
                )
                previous = source

        dt_errors = [
            abs((times[i] - times[i - 1]) - 0.1)
            for i in range(1, len(times))
        ]

        lines = [
            "Latest clip: %s" % clip,
            "frames = %d" % len(rows),
            "time = %.3f .. %.3f" % (times[0], times[-1]),
            "source counts = %s" % dict(Counter(sources)),
            "transitions = %s" % transitions,
            "max dt error = %.9f s" % (max(dt_errors) if dt_errors else 0.0),
        ]
        self.dataset_summary_var.set("\n".join(lines))

    def open_latest_dataset(self):
        clip = self._find_latest_clip()
        if clip is None:
            return self._error("Dataset", "No dataset found.")
        try:
            subprocess.Popen(["xdg-open", str(clip)])
        except Exception as exc:
            self._error("Open folder failed", exc)

    def export_latest_dataset(self):
        clip = self._find_latest_clip()
        if clip is None:
            return self._error("Export", "No dataset found.")

        dest = filedialog.askdirectory(title="Choose export destination")
        if not dest:
            return

        dest = Path(dest)
        target = dest / clip.name
        if target.exists():
            return self._error(
                "Export",
                "Destination already exists:\n%s" % target
            )

        cmd = ["cp", "-a", str(clip), str(dest)]
        try:
            self.runner.start(
                "EXPORT", cmd,
                cwd=self.project_root(),
                env=os.environ.copy(),
            )
        except Exception as exc:
            self._error("Export failed", exc)

    # ------------------------------------------------------------------
    # Events / logging
    # ------------------------------------------------------------------

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                if not event:
                    continue
                if event[0] == "log":
                    self._log(event[1])
                elif event[0] == "done":
                    _, name, rc = event
                    self._log("[%s] EXIT=%d\n" % (name, rc))
                    callback = self.runner.callbacks.pop(name, None)
                    if callback is not None:
                        try:
                            callback(rc)
                        except Exception as exc:
                            self._log("[%s] callback error: %s\n" % (name, exc))
                    self.status_var.set("%s finished (exit %d)" % (name, rc))
        except queue.Empty:
            pass

        running = [
            name for name, proc in self.runner.processes.items()
            if proc.poll() is None
        ]
        if running:
            self.status_var.set("Running: " + ", ".join(running))

        self.after(100, self._poll_events)

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", str(text))
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _save_log(self):
        value = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log", "*.log"), ("Text", "*.txt"), ("All", "*.*")],
        )
        if not value:
            return
        text = self.log.get("1.0", "end")
        Path(value).write_text(text, encoding="utf-8")

    def _error(self, title, value):
        message = str(value)
        self._log("ERROR: %s: %s\n" % (title, message))
        messagebox.showerror(title, message)
        return False

    def _on_close(self):
        running = [
            name for name, proc in self.runner.processes.items()
            if proc.poll() is None
        ]
        if running:
            answer = messagebox.askyesno(
                APP_NAME,
                "Processes are still running:\n%s\n\nStop them and exit?"
                % ", ".join(running)
            )
            if not answer:
                return
            self.runner.stop_all()
        self.destroy()


def main():
    app = FailureReplayWorkbench()
    app.mainloop()


if __name__ == "__main__":
    main()
