#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from b2d_collector.failure_replay.spec import build_intervention_spec
from b2d_collector.failure_replay.tape import BehaviorTape


def main():
    parser = argparse.ArgumentParser(description="Create an intervention spec without the GUI")
    parser.add_argument("--run", required=True)
    parser.add_argument("--record-start", type=float, required=True, help="Relative seconds from first tape sample")
    parser.add_argument("--handoff", type=float, required=True, help="Relative seconds from first tape sample")
    parser.add_argument("--record-end", type=float, required=True, help="Relative seconds from first tape sample")
    parser.add_argument("--case-id", default="case_0001")
    parser.add_argument("--type", default="failure")
    parser.add_argument("--description", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tape = BehaviorTape(args.run)
    start = tape.index_at_relative_time(args.record_start)
    handoff = tape.index_at_relative_time(args.handoff)
    end = tape.index_at_relative_time(args.record_end)
    if not (start <= handoff <= end):
        raise SystemExit("record-start <= handoff <= record-end is required")

    payload = build_intervention_spec(
        tape=tape,
        source_run=str(tape.run_dir),
        case_id=args.case_id,
        intervention_type=args.type,
        description=args.description,
        record_start_index=start,
        handoff_index=handoff,
        record_end_index=end,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(output)


if __name__ == "__main__":
    main()
