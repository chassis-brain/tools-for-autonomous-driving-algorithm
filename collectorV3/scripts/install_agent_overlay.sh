#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${B2D_ROOT:-}" ]]; then
  echo "请先设置 B2D_ROOT" >&2
  exit 2
fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$B2D_ROOT/leaderboard/team_code/b2d_collector"
mkdir -p "$TARGET"
cp -R "$PROJECT_ROOT/src/b2d_collector/." "$TARGET/"
cp "$PROJECT_ROOT/leaderboard_agent.py" "$B2D_ROOT/leaderboard/team_code/b2d_custom_collector_agent.py"
echo "已安装到 $TARGET；入口为 leaderboard/team_code/b2d_custom_collector_agent.py。"
