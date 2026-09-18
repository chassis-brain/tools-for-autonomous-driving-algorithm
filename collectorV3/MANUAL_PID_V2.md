# Failure Replay Workbench – Three-point / Manual PID V2

This build keeps the validated deterministic Replay contract unchanged:

- Tape state reference: current CARLA world elapsed time.
- Replay control reference: CARLA world elapsed time + one Tape tick.
- `time_offset_seconds` remains `0.0`.

## Three-point case

1. **Record Start** – selected from the original E2E Behavior Tape. Carries time and world position. Base recording begins here.
2. **Handoff** – selected from the same original E2E Tape. Carries time and world position. Control switches here.
3. **End** – freely selected world XY. It has no E2E timestamp and does not need to lie on the E2E trajectory or route reference. Once the ego is within the configured radius for the configured number of frames, the terminal frame is recorded, the Collector stops and the unused remainder of the route is fast-finished.

## Replacement modes

- **PDM Expert** – existing shadow warm-up and handoff path.
- **Manual Path + PID** – user authors a spatial path from Handoff to End and a speed-vs-position curve.

Manual path format is spatial only:

- sparse draggable anchors (default spacing 6 m), endpoints locked to Handoff and End;
- clamped cubic B-spline;
- dense arc-length path `P(s)` (default 0.2 m spacing);
- draggable target-speed profile `v(s)`, interpolated with PCHIP;
- lateral feedback PID + heading feedback + curvature feed-forward;
- longitudinal speed PID for throttle/brake.

There is intentionally no manual path timestamp schedule. Actual execution time emerges from tracking `P(s)` with `v(s)`.
