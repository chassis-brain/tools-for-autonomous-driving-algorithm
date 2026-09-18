#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CARLA_ROOT:-}" || -z "${B2D_ROOT:-}" ]]; then
  echo "Please set CARLA_ROOT and B2D_ROOT first." >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENARIO_RUNNER_PY_ROOT="${SCENARIO_RUNNER_PY_ROOT:-$B2D_ROOT/scenario_runner}"

# Keep one durable file log. stdout/stderr remain attached to the parent pipe;
# the GUI ProcessRunner mirrors every line into the Live Console AND back to the
# launching terminal. This is more reliable than writing directly to /proc/fd.
mkdir -p "$PROJECT_ROOT/logs"
RECOVERY_LIVE_LOG="${RECOVERY_LIVE_LOG:-$PROJECT_ROOT/logs/recovery_live.log}"
: > "$RECOVERY_LIVE_LOG"
exec > >(tee -a "$RECOVERY_LIVE_LOG") 2>&1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "[RecoveryRunner] live log      : $RECOVERY_LIVE_LOG"
echo "[RecoveryRunner] expected mode : ${RECOVERY_EXPECT_MODE:-<not-set>}"
echo "[RecoveryRunner] expected plan : ${RECOVERY_EXPECT_PLAN:-<not-set>}"

CONFIG="${1:-$PROJECT_ROOT/configs/replay_to_pdm.yaml}"
ROUTES="${2:-}"
ROUTES_SUBSET="${3:-}"
CONFIG="$(realpath "$CONFIG")"

if [[ -z "$ROUTES" ]]; then
  ROUTES="$(python - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], 'r') as f:
    print(yaml.safe_load(f)['route_xml'])
PY
)"
fi
ROUTES="$(realpath "$ROUTES")"

PORT="${PORT:-23000}"
TM_PORT="${TM_PORT:-23050}"
GPU_RANK="${GPU_RANK:-0}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/outputs_recovery/evaluation.json}"
SAVE_PATH="${SAVE_PATH:-$PROJECT_ROOT/outputs_recovery/eval_logs}"

CARLA_EGG="${CARLA_EGG:-}"
if [[ -z "$CARLA_EGG" ]]; then
  CARLA_EGG="$(find "$CARLA_ROOT/PythonAPI/carla/dist" -maxdepth 1 -type f \
    \( -name 'carla-0.9.15-*.egg' -o -name 'carla-0.9.15-*.whl' \) \
    | sort | head -n 1 || true)"
fi
if [[ -z "$CARLA_EGG" || ! -e "$CARLA_EGG" ]]; then
  echo "Cannot find CARLA 0.9.15 Python API egg/wheel; set CARLA_EGG explicitly." >&2
  exit 3
fi

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/carla_garage/team_code:$B2D_ROOT:$B2D_ROOT/leaderboard:$SCENARIO_RUNNER_PY_ROOT:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_EGG"
export IS_BENCH2DRIVE=True
export PLANNER_TYPE="${PLANNER_TYPE:-traj}"
export GPU_RANK
export SAVE_PATH
mkdir -p "$(dirname "$CHECKPOINT")" "$SAVE_PATH"

ARGS=(
  --routes="$ROUTES"
  --repetitions=1
  --track="${TRACK:-SENSORS}"
  --checkpoint="$CHECKPOINT"
  --agent="$PROJECT_ROOT/replay_to_pdm_agent.py"
  --agent-config="$CONFIG"
  --debug=0
  --port="$PORT"
  --traffic-manager-port="$TM_PORT"
  --traffic-manager-seed="${TM_SEED:-0}"
  --gpu-rank="$GPU_RANK"
)

if [[ -n "$ROUTES_SUBSET" ]]; then
  ARGS+=(--routes-subset="$ROUTES_SUBSET")
fi

echo "[RecoveryRunner] project        : $PROJECT_ROOT"
echo "[RecoveryRunner] config         : $CONFIG"
echo "[RecoveryRunner] route subset   : ${ROUTES_SUBSET:-<all>}"
echo "[RecoveryRunner] launching evaluator..."

(
  cd "$B2D_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU_RANK" python \
    "$B2D_ROOT/leaderboard/leaderboard/leaderboard_evaluator.py" \
    "${ARGS[@]}"
)
