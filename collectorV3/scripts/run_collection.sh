#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CARLA_ROOT:-}" || -z "${B2D_ROOT:-}" ]]; then
  echo "请先设置 CARLA_ROOT 与 B2D_ROOT" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SCENARIO_RUNNER_PY_ROOT="${SCENARIO_RUNNER_PY_ROOT:-$B2D_ROOT/scenario_runner}"

CONFIG="${1:-$PROJECT_ROOT/configs/manual.yaml}"
ROUTES="${2:-$PROJECT_ROOT/routes/custom_parking_exit.xml}"

DATASET_PROFILE="${B2D_DATASET_PROFILE:-base}"
SENSOR_CONFIG="${B2D_SENSOR_CONFIG:-}"

if [[ $# -ge 2 ]]; then
  shift 2
else
  shift "$#"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-profile)
      [[ $# -ge 2 ]] || { echo "--dataset-profile requires a value" >&2; exit 2; }
      DATASET_PROFILE="$2"
      shift 2
      ;;
    --sensor-config)
      [[ $# -ge 2 ]] || { echo "--sensor-config requires a path" >&2; exit 2; }
      SENSOR_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown collection option: $1" >&2
      exit 2
      ;;
  esac
done

CONFIG="$(realpath "$CONFIG")"
ROUTES="$(realpath "$ROUTES")"

if [[ -n "$SENSOR_CONFIG" ]]; then
  SENSOR_CONFIG="$(realpath "$SENSOR_CONFIG")"
fi

export B2D_DATASET_PROFILE="$DATASET_PROFILE"
export B2D_SENSOR_CONFIG="$SENSOR_CONFIG"

echo "[Collector] dataset profile: $B2D_DATASET_PROFILE"
if [[ -n "$B2D_SENSOR_CONFIG" ]]; then
  echo "[Collector] sensor config: $B2D_SENSOR_CONFIG"
fi
PORT="${PORT:-23000}"
TM_PORT="${TM_PORT:-24000}"
GPU_RANK="${GPU_RANK:-0}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/outputs/evaluation.json}"
SAVE_PATH="${SAVE_PATH:-$PROJECT_ROOT/outputs/eval_logs}"

CARLA_EGG="${CARLA_EGG:-}"
if [[ -z "$CARLA_EGG" ]]; then
  CARLA_EGG="$(find "$CARLA_ROOT/PythonAPI/carla/dist" -maxdepth 1 -type f \
    \( -name 'carla-0.9.15-*.egg' -o -name 'carla-0.9.15-*.whl' \) \
    | sort | head -n 1 || true)"
fi

if [[ -z "$CARLA_EGG" || ! -e "$CARLA_EGG" ]]; then
  echo "找不到 CARLA 0.9.15 Python API egg/wheel；可通过 CARLA_EGG 显式指定。" >&2
  exit 3
fi

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/carla_garage/team_code:$B2D_ROOT:$B2D_ROOT/leaderboard:$SCENARIO_RUNNER_PY_ROOT:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_EGG"
export IS_BENCH2DRIVE=True
export PLANNER_TYPE="${PLANNER_TYPE:-traj}"
export GPU_RANK
export SAVE_PATH
mkdir -p "$(dirname "$CHECKPOINT")" "$SAVE_PATH"

(
  cd "$B2D_ROOT"

  CUDA_VISIBLE_DEVICES="$GPU_RANK" python \
    "$B2D_ROOT/leaderboard/leaderboard/leaderboard_evaluator.py" \
    --routes="$ROUTES" \
    --repetitions=1 \
    --track="${TRACK:-SENSORS}" \
    --checkpoint="$CHECKPOINT" \
    --agent="$PROJECT_ROOT/leaderboard_agent.py" \
    --agent-config="$CONFIG" \
    --debug=0 \
    --port="$PORT" \
    --traffic-manager-port="$TM_PORT" \
    --gpu-rank="$GPU_RANK"
)
