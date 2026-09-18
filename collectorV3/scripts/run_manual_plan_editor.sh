#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! python - <<'PY' >/dev/null 2>&1
import tkinter
import numpy
import matplotlib
PY
then
  echo "[manual-plan] Missing Python dependencies in the active environment."
  echo "[manual-plan] Required: tkinter, numpy, matplotlib"
  exit 2
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -u "$ROOT/tools/manual_plan_editor.py" --project-root "$ROOT" "$@"
