#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Validate an E2E Behavior Tape")
    parser.add_argument("run")
    args = parser.parse_args()
    run = Path(args.run).expanduser().resolve()
    frames = run / "frames.jsonl"
    if not frames.is_file():
        raise SystemExit("missing %s" % frames)

    rows = []
    with frames.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("FAIL: no frames")

    world_frames = [int(row["world"]["frame"]) for row in rows]
    times = [float(row["world"]["elapsed_seconds"]) for row in rows]
    gaps = [(a, b) for a, b in zip(world_frames[:-1], world_frames[1:]) if b != a + 1]
    bad_times = [(a, b) for a, b in zip(times[:-1], times[1:]) if b <= a]
    missing_control = [
        row["world"]["frame"]
        for row in rows
        if not (row.get("ego") or {}).get("applied_control")
    ]

    complete = (run / "COMPLETE").is_file()
    manifest_path = run / "manifest.json"
    manifest_result = {}
    manifest_error = None
    if manifest_path.is_file():
        try:
            manifest_result = (json.loads(manifest_path.read_text(encoding="utf-8")).get("result") or {})
        except Exception as exc:
            manifest_error = repr(exc)

    print("run                =", run)
    print("samples            =", len(rows))
    print("world_frame        = %d .. %d" % (world_frames[0], world_frames[-1]))
    print("sim_time           = %.3f .. %.3f" % (times[0], times[-1]))
    print("frame_gaps         =", gaps[:10])
    print("non_monotonic_time =", bad_times[:10])
    print("missing_control    =", missing_control[:10])
    print("COMPLETE           =", complete)
    if manifest_error is not None:
        print("manifest_error      =", manifest_error)
    elif manifest_result:
        print("manifest_integrity  =", manifest_result.get("integrity_ok"))
        print("manifest_samples    =", manifest_result.get("samples"))
        print("manifest_written    =", manifest_result.get("frames_written"))

    manifest_ok = (
        manifest_error is None
        and bool(manifest_result)
        and bool(manifest_result.get("integrity_ok", False))
        and int(manifest_result.get("samples", -1)) == len(rows)
        and int(manifest_result.get("frames_written", -1)) == len(rows)
    )
    ok = not gaps and not bad_times and not missing_control and complete and manifest_ok
    print("RESULT             =", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
