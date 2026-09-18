#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -n "${B2D_COLLECTOR_PYTHON:-}" ]]; then
  PY="$B2D_COLLECTOR_PYTHON"
else
  PY="$(command -v python)"
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "[Workbench] project    : $PROJECT_ROOT"
echo "[Workbench] python     : $PY"
echo "[Workbench] PYTHONPATH : $PROJECT_ROOT/src (project source first)"

# Fail before Tk starts if the installed ManualPlanPanel expects an API that the
# current source tree does not export.  This checks the *actual panel file on the
# user's machine*, avoiding one-missing-symbol-at-a-time crashes.
"$PY" -u - "$PROJECT_ROOT" <<'PY'
from __future__ import print_function
import ast
import importlib
import os
import sys

root = os.path.abspath(sys.argv[1])
mod = importlib.import_module("b2d_collector.failure_replay.manual_plan")
print("[Workbench] manual_plan module:", os.path.abspath(mod.__file__))

required = []
panel = os.path.join(root, "tools", "manual_plan_panel.py")
if os.path.isfile(panel):
    try:
        with open(panel, "r") as f:
            tree = ast.parse(f.read(), filename=panel)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "b2d_collector.failure_replay.manual_plan":
                required.extend(alias.name for alias in node.names)
    except Exception as exc:
        print("[Workbench][WARN] could not inspect manual_plan_panel.py:", exc)

required = sorted(set(required))
if required:
    print("[Workbench] ManualPlanPanel API:", ", ".join(required))
missing = [name for name in required if not hasattr(mod, name)]
if missing:
    print("[Workbench][FATAL] manual_plan.py is missing API(s):", ", ".join(missing))
    print("[Workbench][FATAL] panel:", panel)
    raise SystemExit(41)
print("[Workbench] Manual Planner API check: PASS")
PY

exec "$PY" -u "$PROJECT_ROOT/tools/failure_replay_gui_v23.py"
