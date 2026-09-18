#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-safe launcher for Failure Replay Workbench (v2.3.1).

This wrapper deliberately does not replace tools/failure_replay_gui.py.  It
loads the currently installed GUI (including the Manual Planner page from the
previous patch) and adds:

* an always-visible Recovery Live Console window;
* stdout/stderr mirroring back to the terminal that launched the GUI;
* one-click synchronous Recovery preflight;
* a fail-closed GUI -> Agent contract for replacement.mode / manual plan path;
* explicit Manual/PDM runtime status in the GUI.

Target: Python 3.7+.
"""
from __future__ import print_function

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:
    import Tkinter as tk
    import tkMessageBox as messagebox
    import ttk


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SRC_ROOT = PROJECT_ROOT / "src"

# Prefer the files in this collector checkout even when b2d_collector was also
# installed into the conda environment in the past.  The Manual Planner panel
# and Manual PID runtime must resolve to the same patched source tree.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import b2d_collector.failure_replay.manual_plan as _manual_plan_bootcheck
except Exception as exc:
    raise RuntimeError(
        "Manual Planner runtime is incomplete. Expected module under %s: %s"
        % (SRC_ROOT, exc)
    )

_term_boot_module = getattr(_manual_plan_bootcheck, "__file__", "<unknown>")
print("[Workbench] manual_plan module: %s" % _term_boot_module, flush=True)

BASE_GUI = HERE / "failure_replay_gui.py"

if not BASE_GUI.is_file():
    raise RuntimeError("Base Workbench GUI not found: %s" % BASE_GUI)

spec = importlib.util.spec_from_file_location("failure_replay_gui_base", str(BASE_GUI))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def _term_write(text):
    try:
        stream = getattr(sys, "__stdout__", None) or sys.stdout
        stream.write(str(text))
        stream.flush()
    except Exception:
        pass


def _normalize_mode(value):
    text = str(value or "pdm_expert").strip().lower()
    aliases = {
        "pdm": "pdm_expert",
        "expert": "pdm_expert",
        "pdm-expert": "pdm_expert",
        "manual": "manual_pid",
        "manual_path": "manual_pid",
        "manual-path": "manual_pid",
        "manual_path_pid": "manual_pid",
    }
    return aliases.get(text, text)


def _case_runtime_contract(case_path):
    path = Path(case_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError("Case JSON does not exist: %s" % path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    repl = raw.get("replacement") or {}
    mode = _normalize_mode(repl.get("mode", "pdm_expert"))
    if mode not in ("pdm_expert", "manual_pid"):
        raise RuntimeError("Unsupported replacement.mode: %s" % mode)
    plan = None
    if mode == "manual_pid":
        value = str(repl.get("plan") or "").strip()
        if not value:
            raise RuntimeError("manual_pid case is missing replacement.plan")
        plan = Path(value).expanduser()
        if not plan.is_absolute():
            plan = path.parent / plan
        plan = plan.resolve()
        if not plan.is_file():
            raise RuntimeError("Manual plan does not exist: %s" % plan)
    return path, raw, mode, plan


# ---------------------------------------------------------------------------
# Make every child-process line visible in BOTH places:
#   1) GUI event stream / Process log
#   2) terminal that launched the Workbench
# ---------------------------------------------------------------------------
def _reader_with_terminal(self, name, proc):
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            rendered = "[%s] %s" % (name, line)
            _term_write(rendered)
            self.event_queue.put(("log", rendered))
    except Exception as exc:
        rendered = "[%s] reader error: %s\n" % (name, exc)
        _term_write(rendered)
        self.event_queue.put(("log", rendered))
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        rc = proc.wait()
        _term_write("[%s] EXIT=%d\n" % (name, rc))
        self.event_queue.put(("done", name, rc))


base.ProcessRunner._reader = _reader_with_terminal
_orig_runner_start = base.ProcessRunner.start


def _runner_start_with_terminal(self, name, cmd, cwd=None, env=None, callback=None):
    _term_write("\n[%s] START\n$ %s\n" % (name, base.shell_join(cmd)))
    return _orig_runner_start(self, name, cmd, cwd=cwd, env=env, callback=callback)


base.ProcessRunner.start = _runner_start_with_terminal


class FailureReplayWorkbenchV23(base.FailureReplayWorkbench):
    def __init__(self):
        self._runtime_console = None
        self._runtime_console_text = None
        self._runtime_state_var = None
        super(FailureReplayWorkbenchV23, self).__init__()
        self.title("Failure Replay Workbench · Runtime Safe")
        self._install_runtime_panel()

    # ----------------------------- console UI -----------------------------
    def _install_runtime_panel(self):
        if not hasattr(self, "tab_recovery"):
            return
        panel = ttk.Frame(self.tab_recovery)
        children = self.tab_recovery.winfo_children()
        manager = ""
        for child in children:
            if child is panel:
                continue
            try:
                manager = child.winfo_manager()
            except Exception:
                manager = ""
            if manager:
                break

        self._runtime_state_var = tk.StringVar(value="Runtime: not started")
        ttk.Button(
            panel,
            text="Open Recovery Live Console",
            command=self._show_recovery_console,
        ).pack(side="left", padx=(0, 8))
        ttk.Label(panel, textvariable=self._runtime_state_var).pack(side="left")

        try:
            if manager == "grid":
                # A high row number does not create blank rows in Tk; it simply
                # places this below all normal recovery controls.
                panel.grid(row=99, column=0, columnspan=99, sticky="ew", padx=8, pady=8)
            else:
                panel.pack(fill="x", padx=8, pady=8)
        except Exception:
            # Never let a cosmetic panel prevent the Workbench from opening.
            pass

    def _show_recovery_console(self):
        if self._runtime_console is not None:
            try:
                if self._runtime_console.winfo_exists():
                    self._runtime_console.deiconify()
                    self._runtime_console.lift()
                    return
            except Exception:
                pass

        win = tk.Toplevel(self)
        win.title("Recovery Live Console")
        win.geometry("1180x680")
        win.minsize(840, 420)
        self._runtime_console = win

        head = ttk.Frame(win, padding=(10, 8))
        head.pack(fill="x")
        ttk.Label(
            head,
            textvariable=self._runtime_state_var,
            font=(getattr(self, "_ui_font_family", "TkDefaultFont"), 11, "bold"),
        ).pack(side="left")
        ttk.Button(
            head,
            text="Stop Recovery",
            command=lambda: self.runner.stop("RECOVERY"),
        ).pack(side="right")
        ttk.Button(
            head,
            text="Clear",
            command=self._clear_runtime_console,
        ).pack(side="right", padx=6)

        body = ttk.Frame(win, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(body, orient="vertical")
        scroll.pack(side="right", fill="y")
        palette = getattr(self, "_ui_colors", {})
        text = tk.Text(
            body,
            wrap="none",
            yscrollcommand=scroll.set,
            font=(getattr(self, "_ui_fixed_font_family", "TkFixedFont"), 10),
            background=palette.get("canvas", "#191D22"),
            foreground="#D8DCE1",
            insertbackground=palette.get("text", "#E7E9EC"),
            selectbackground=palette.get("selection", "#405A57"),
            selectforeground="#F5F7F6",
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=11,
        )
        text.pack(side="left", fill="both", expand=True)
        scroll.configure(command=text.yview)
        text.tag_configure("manual", foreground="#8CB8A0")
        text.tag_configure("handoff", foreground="#8EAECC")
        text.tag_configure("error", foreground="#D98A8A")
        text.tag_configure("warn", foreground="#D0AD73")
        self._runtime_console_text = text
        win.protocol("WM_DELETE_WINDOW", win.withdraw)

        self._runtime_console_append(
            "Recovery output will appear here immediately after you click Run Recovery.\n"
            "The same child-process output is also mirrored to the terminal.\n\n"
        )

    def _clear_runtime_console(self):
        text = self._runtime_console_text
        if text is not None:
            text.delete("1.0", "end")

    def _runtime_console_append(self, value):
        text = self._runtime_console_text
        if text is None:
            return
        value = str(value)
        upper = value.upper()
        tag = None
        if "ERROR" in upper or "TRACEBACK" in upper or "RESULT             = FAIL" in upper:
            tag = "error"
        elif "WARNING" in upper or "[WARN]" in upper:
            tag = "warn"
        elif "HANDOFF E2E ->" in upper:
            tag = "handoff"
        elif "[MANUAL]" in upper or "MANUAL_PID" in upper:
            tag = "manual"
        text.insert("end", value, tag or ())
        text.see("end")

    def _log(self, text):
        # Preserve whatever Process log/layout the installed Workbench already has.
        try:
            super(FailureReplayWorkbenchV23, self)._log(text)
        except Exception:
            pass
        self._runtime_console_append(text)

        upper = str(text).upper()
        if self._runtime_state_var is not None:
            if "HANDOFF E2E -> MANUAL_PID" in upper:
                self._runtime_state_var.set("Runtime: MANUAL_PID ACTIVE")
            elif "HANDOFF E2E -> PDM_EXPERT" in upper:
                self._runtime_state_var.set("Runtime: PDM_EXPERT ACTIVE")
            elif "RESULT             = FAIL" in upper or "RECOVERY PROCESS EXITED WITH" in upper and " 0" not in upper:
                self._runtime_state_var.set("Runtime: ERROR — inspect console")

    # ----------------------------- E2E / Probe -----------------------------
    def _probe_dirs(self):
        root = self.project_root() / "probe_runs"
        if not root.is_dir():
            return []
        return sorted(
            [item for item in root.glob("probe_*") if item.is_dir()],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )

    def _probe_newest(self):
        items = self._probe_dirs()
        return items[0] if items else None

    def _announce_probe_output(self, attempt=0):
        candidate = self._probe_newest()
        before = getattr(self, "_probe_dirs_before", set())
        if candidate is not None and str(candidate.resolve()) not in before:
            self._active_probe_output = candidate.resolve()
            try:
                self.probe_var.set(str(self._active_probe_output))
            except Exception:
                pass
            self._log("[PROBE] output directory = %s\n" % self._active_probe_output)
            self._log("[PROBE] listener ready; E2E may start now.\n")
            if getattr(self, "_e2e_waiting_for_probe", False):
                self._e2e_waiting_for_probe = False
                self.after(100, self._launch_e2e_now)
            return

        if self.runner.running("PROBE") and attempt < 40:
            self.after(250, lambda: self._announce_probe_output(attempt + 1))
            return

        if getattr(self, "_e2e_waiting_for_probe", False):
            self._e2e_waiting_for_probe = False
            self._log("ERROR: Probe listener did not create an output directory. E2E was not started.\n")
            messagebox.showerror(
                "Probe did not become ready",
                "Probe Listener did not create probe_runs/probe_* within 10 seconds.\n\n"
                "Check the terminal output above for the CARLA connection error.",
            )

    def _summarize_probe_output(self, candidate=None):
        if candidate is None:
            candidate = getattr(self, "_active_probe_output", None) or self._probe_newest()
        if candidate is None:
            self._log("ERROR: Probe process finished but no probe_* directory exists.\n")
            return False
        candidate = Path(candidate).resolve()
        try:
            self.probe_var.set(str(candidate))
        except Exception:
            pass
        manifest_path = candidate / "manifest.json"
        frames_path = candidate / "frames.jsonl"
        complete_path = candidate / "COMPLETE"
        result = {}
        if manifest_path.is_file():
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                result = raw.get("result") or {}
            except Exception as exc:
                self._log("[PROBE][WARN] cannot read manifest: %s\n" % exc)
        samples = int(result.get("samples") or 0)
        reason = str(result.get("reason") or "unknown")
        integrity_ok = bool(result.get("integrity_ok", False))
        frame_bytes = frames_path.stat().st_size if frames_path.is_file() else 0
        self._log("\n========== E2E PROBE OUTPUT ==========\n")
        self._log("[PROBE] directory    = %s\n" % candidate)
        self._log("[PROBE] frames       = %s (%d bytes)\n" % (frames_path, frame_bytes))
        self._log("[PROBE] samples      = %d\n" % samples)
        self._log("[PROBE] finish reason= %s\n" % reason)
        self._log("[PROBE] integrity_ok = %s\n" % integrity_ok)
        self._log("[PROBE] COMPLETE     = %s\n" % complete_path.is_file())
        self._log("======================================\n\n")
        try:
            self.refresh_probes()
            self.probe_var.set(str(candidate))
        except Exception:
            pass
        if samples <= 0 or frame_bytes <= 0:
            messagebox.showwarning(
                "E2E finished but Probe has no frames",
                "A probe directory was created, but it contains no Behavior Tape frames.\n\n"
                "Output:\n%s\n\nReason: %s\n\n"
                "Please copy the [PROBE] lines from the terminal if this happens again."
                % (candidate, reason),
            )
            return False
        return True

    def _probe_finished(self, rc):
        self._log("[PROBE] process exited rc=%d\n" % int(rc))
        self._summarize_probe_output()

    def start_probe(self):
        if self.runner.running("PROBE"):
            self._log("[PROBE] listener is already running.\n")
            return True

        project = self.project_root()
        script = project / "tools" / "e2e_probe_listener.py"
        if not script.is_file():
            self._error("Probe", "Missing %s" % script)
            return False

        root = project / "probe_runs"
        root.mkdir(parents=True, exist_ok=True)
        self._probe_dirs_before = set(str(x.resolve()) for x in self._probe_dirs())
        self._active_probe_output = None

        port = self.settings_vars["carla_port"].get().strip() or "23000"
        cmd = [
            sys.executable,
            "-u",
            str(script),
            "--host", "127.0.0.1",
            "--port", port,
            "--output-root", str(root),
            "--ego-role", "hero",
            "--no-native-recorder",
        ]
        try:
            route_xml = self.route_xml()
            if route_xml.is_file():
                cmd.extend(["--route-xml", str(route_xml)])
        except Exception:
            pass
        rid = ""
        try:
            rid = self.route_var.get().strip()
        except Exception:
            pass
        if rid:
            cmd.extend(["--route-id", rid])

        self._log("\n[PROBE] starting listener before E2E...\n")
        try:
            self.runner.start(
                "PROBE",
                cmd,
                cwd=project,
                env=self.collector_env(),
                callback=self._probe_finished,
            )
        except Exception as exc:
            self._error("Start Probe failed", exc)
            return False
        self.after(200, lambda: self._announce_probe_output(0))
        return True

    def run_e2e(self):
        if self.runner.running("E2E"):
            return self._error("E2E", "E2E is already running.")

        # One click now owns the complete Probe -> E2E transaction.  A Behavior
        # Tape cannot be accidentally omitted by forgetting the separate Probe
        # button or starting it too late.
        if not self.runner.running("PROBE"):
            self._log("\n[E2E] Probe is not running; auto-starting it first.\n")
            self._e2e_waiting_for_probe = True
            if not self.start_probe():
                self._e2e_waiting_for_probe = False
                return False
            return True

        self._log("[E2E] existing Probe listener detected; launching route.\n")
        return self._launch_e2e_now()

    def _launch_e2e_now(self):
        if self.runner.running("E2E"):
            return True
        try:
            base.FailureReplayWorkbench.run_e2e(self)
        except Exception as exc:
            self._log("ERROR: E2E launch failed: %s\n" % exc)
            if self.runner.running("PROBE"):
                self.runner.stop("PROBE")
            return False

        if self.runner.running("E2E"):
            # Replace the base refresh-only callback with a transaction callback
            # that also finalizes the Probe deterministically.
            self.runner.callbacks["E2E"] = self._e2e_finished
            self._log("[E2E] route started with Probe capture active.\n")
            return True

        self._log("ERROR: E2E process did not start; stopping Probe.\n")
        if self.runner.running("PROBE"):
            self.runner.stop("PROBE")
        return False

    def _e2e_finished(self, rc):
        self._log("\n[E2E] evaluator exited rc=%d\n" % int(rc))
        self._log("[E2E] finalizing Behavior Tape...\n")
        # Give CARLA one short moment to deliver the final tick, then request a
        # graceful SIGTERM.  The patched listener catches it, drains its writer
        # queue, writes manifest.result, and creates COMPLETE when integrity is OK.
        self.after(800, self._finalize_probe_after_e2e)

    def _finalize_probe_after_e2e(self):
        if self.runner.running("PROBE"):
            self._log("[PROBE] requesting graceful finalize after E2E exit.\n")
            self.runner.stop("PROBE")
        else:
            self._summarize_probe_output()

    # ------------------------- recovery launch ----------------------------
    def _run_inline_preflight(self):
        script = self.project_root() / "tools" / "preflight_recovery.py"
        if not script.is_file():
            raise RuntimeError("Missing preflight script: %s" % script)
        cmd = [sys.executable, str(script), str(self.recovery_yaml())]
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(self.project_root()),
            env=self.collector_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        out, _ = proc.communicate()
        self._log("\n[RECOVERY PREFLIGHT]\n%s" % out)
        if proc.returncode != 0:
            raise RuntimeError("Recovery preflight failed. See the Live Console above.")

    def _update_recovery_yaml_verified(self):
        """Update recovery YAML and verify the file that was actually written.

        Older/newer Workbench revisions do not agree on the return value of
        update_recovery_yaml(): some return True, while others successfully write
        the YAML and return None.  Treat an explicit False as failure, but for
        None verify the on-disk YAML instead of reporting a false failure.
        """
        rid = self.recovery_route_var.get().strip() or self.route_var.get().strip()
        expected_probe = Path(self.recovery_probe_var.get().strip()).expanduser().resolve()
        expected_case = Path(self.recovery_case_var.get().strip()).expanduser().resolve()
        if not rid:
            raise RuntimeError("Route ID is empty")
        if not expected_probe.is_dir():
            raise RuntimeError("Selected Probe directory does not exist: %s" % expected_probe)
        if not expected_case.is_file():
            raise RuntimeError("Selected Case JSON does not exist: %s" % expected_case)

        result = self.update_recovery_yaml()
        if result is False:
            raise RuntimeError("Recovery YAML update explicitly failed")

        target = Path(self.recovery_yaml()).expanduser().resolve()
        if not target.is_file():
            raise RuntimeError("Recovery YAML was not created: %s" % target)

        yaml_mod = getattr(base, "yaml", None)
        if yaml_mod is None:
            # If PyYAML were unavailable the base updater could not have emitted a
            # valid YAML file.  Still fail loudly rather than trusting a stale file.
            raise RuntimeError("PyYAML is unavailable; cannot verify %s" % target)

        try:
            with target.open("r", encoding="utf-8") as handle:
                cfg = yaml_mod.safe_load(handle) or {}
        except Exception as exc:
            raise RuntimeError("Cannot read updated recovery YAML %s: %s" % (target, exc))

        replay = cfg.get("replay") or {}

        problems = []
        if str(cfg.get("route_id", "")).strip() != str(rid):
            problems.append("route_id=%r expected=%r" % (cfg.get("route_id"), rid))

        tape_value = str(replay.get("tape_dir") or "").strip()
        case_value = str(replay.get("intervention_spec") or "").strip()
        try:
            actual_probe = Path(tape_value).expanduser().resolve() if tape_value else None
        except Exception:
            actual_probe = None
        try:
            actual_case = Path(case_value).expanduser().resolve() if case_value else None
        except Exception:
            actual_case = None

        if actual_probe != expected_probe:
            problems.append("replay.tape_dir=%r expected=%r" % (tape_value, str(expected_probe)))
        if actual_case != expected_case:
            problems.append("replay.intervention_spec=%r expected=%r" % (case_value, str(expected_case)))

        if problems:
            raise RuntimeError(
                "Recovery YAML verification failed after update:\n  " + "\n  ".join(problems)
            )

        self._log("[GUI] Recovery YAML verified on disk: %s\n" % target)
        return True

    def run_preflight(self):
        self._show_recovery_console()
        self._runtime_state_var.set("Runtime: validating recovery YAML...")
        self._log("\n========== RUN PREFLIGHT CLICKED ==========\n")
        try:
            self._update_recovery_yaml_verified()
            self._run_inline_preflight()
            self._runtime_state_var.set("Runtime: preflight PASS")
        except Exception as exc:
            self._runtime_state_var.set("Runtime: PRECHECK ERROR")
            self._log("ERROR: Preflight blocked: %s\n" % exc)

    def run_recovery(self):
        self._show_recovery_console()
        self._runtime_state_var.set("Runtime: validating selected case...")
        self._log("\n========== RUN RECOVERY CLICKED ==========\n")

        try:
            self._update_recovery_yaml_verified()

            case_path, case_raw, mode, plan_path = _case_runtime_contract(
                self.recovery_case_var.get().strip()
            )
            self._log("[GUI] selected case   = %s\n" % case_path)
            self._log("[GUI] replacement     = %s\n" % mode)
            if plan_path is not None:
                self._log("[GUI] manual plan     = %s\n" % plan_path)

            # One click now means: verify first, then run.  This prevents a stale
            # YAML or stale Manual Planner file from silently starting a PDM run.
            self._run_inline_preflight()

            rid = self.recovery_route_var.get().strip() or self.route_var.get().strip()
            project = self.project_root()
            script = project / "scripts" / "run_replay_to_pdm.sh"
            if not script.is_file():
                raise RuntimeError("Missing recovery runner: %s" % script)

            egg = self.carla_egg()
            if not egg:
                raise RuntimeError(
                    "Cannot find a Python 3.7 CARLA egg/wheel. Set CARLA root in Setup."
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
            env["RECOVERY_EXPECT_MODE"] = mode
            if plan_path is not None:
                env["RECOVERY_EXPECT_PLAN"] = str(plan_path)
            else:
                env.pop("RECOVERY_EXPECT_PLAN", None)
            env["PYTHONUNBUFFERED"] = "1"

            cmd = [
                "bash",
                str(script),
                str(self.recovery_yaml()),
                str(self.route_xml()),
                str(rid),
            ]

            self._runtime_state_var.set("Runtime: starting %s" % mode.upper())
            self._log("[GUI] fail-closed expected mode = %s\n" % mode)
            if plan_path is not None:
                self._log("[GUI] fail-closed expected plan = %s\n" % plan_path)
            self._log("[GUI] live file log = %s\n" % (project / "logs" / "recovery_live.log"))

            self.runner.start(
                "RECOVERY",
                cmd,
                cwd=project,
                env=env,
                callback=getattr(self, "_recovery_finished", None),
            )
        except Exception as exc:
            self._runtime_state_var.set("Runtime: launch blocked")
            self._log("ERROR: Recovery launch blocked: %s\n" % exc)
            messagebox.showerror("Recovery blocked", str(exc))


if __name__ == "__main__":
    app = FailureReplayWorkbenchV23()
    app.mainloop()
