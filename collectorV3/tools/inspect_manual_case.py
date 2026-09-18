#!/usr/bin/env python3
from __future__ import print_function

import json
import sys
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None


def resolve(case_path, value):
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = case_path.parent / p
    return p.resolve()


def main():
    if len(sys.argv) != 2:
        print('usage: python tools/inspect_manual_case.py /path/to/case.json')
        return 2
    case_path = Path(sys.argv[1]).expanduser().resolve()
    raw = json.loads(case_path.read_text(encoding='utf-8'))
    replacement = raw.get('replacement') or {}
    print('case              =', case_path)
    print('schema            =', raw.get('schema'))
    print('replacement.mode  =', replacement.get('mode'))
    print('end               =', raw.get('end'))
    value = replacement.get('plan')
    if not value:
        print('ERROR             = replacement.plan is missing')
        return 1
    plan = resolve(case_path, value)
    print('replacement.plan  =', value)
    print('resolved plan     =', plan)
    print('plan exists       =', plan.is_file())
    if not plan.is_file():
        return 1

    doc = json.loads(plan.read_text(encoding='utf-8'))
    print('plan schema       =', doc.get('schema'))
    print('plan status       =', doc.get('status'))
    print('path anchors      =', len(doc.get('path_control_points') or doc.get('anchors') or []))
    print('speed points      =', len(doc.get('speed_control_points') or []))

    generated = doc.get('generated') or {}
    candidates = []
    for v in (generated.get('trajectory_file'), doc.get('trajectory_file'), doc.get('trajectory_npz')):
        if v:
            candidates.append(resolve(plan, v))
    stem = plan.name[:-len('.plan.json')] if plan.name.endswith('.plan.json') else plan.stem
    candidates += [
        (plan.parent / (stem + '.trajectory.npz')).resolve(),
        (plan.parent / 'generated' / (stem + '.trajectory.npz')).resolve(),
    ]
    found = None
    for c in candidates:
        if c.is_file():
            found = c
            break
    print('trajectory        =', found if found else 'NOT FOUND (runtime may regenerate from controls)')
    if found and np is not None:
        with np.load(str(found)) as data:
            keys = list(data.keys())
            print('npz keys          =', keys)
            if 'x' in data:
                print('samples           =', len(data['x']))
            if 'target_speed' in data and len(data['target_speed']):
                print('target speed      = %.3f .. %.3f m/s' % (
                    float(np.min(data['target_speed'])), float(np.max(data['target_speed']))
                ))
    print('RESULT            = PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
