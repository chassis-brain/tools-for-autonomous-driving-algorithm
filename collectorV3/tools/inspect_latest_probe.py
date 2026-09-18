#!/usr/bin/env python3
from __future__ import print_function

import json
import sys
from pathlib import Path


def main():
    project = Path(__file__).resolve().parent.parent
    root = project / "probe_runs"
    items = []
    if root.is_dir():
        items = sorted(
            [p for p in root.glob("probe_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not items:
        print("RESULT = FAIL")
        print("No probe_* directory under: %s" % root)
        return 2

    run = items[0].resolve()
    manifest = run / "manifest.json"
    frames = run / "frames.jsonl"
    complete = run / "COMPLETE"
    raw = {}
    if manifest.is_file():
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception as exc:
            print("manifest read error: %s" % exc)
    result = raw.get("result") or {}
    print("probe_dir      = %s" % run)
    print("manifest       = %s" % manifest.is_file())
    print("frames         = %s" % frames.is_file())
    print("frames_bytes   = %d" % (frames.stat().st_size if frames.is_file() else 0))
    print("samples        = %s" % result.get("samples"))
    print("frames_written = %s" % result.get("frames_written"))
    print("finish_reason  = %s" % result.get("reason"))
    print("integrity_ok   = %s" % result.get("integrity_ok"))
    print("COMPLETE       = %s" % complete.is_file())
    ok = bool(frames.is_file() and frames.stat().st_size > 0 and result.get("samples", 0) > 0)
    print("RESULT         = %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
