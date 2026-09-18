#!/usr/bin/env python3
"""Interactive offline BEV review for choosing record/handoff/end points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from b2d_collector.failure_replay.spec import build_intervention_spec
from b2d_collector.failure_replay.tape import BehaviorTape


TYPES = (
    "failure",
    "slow_decision",
    "poor_decision",
    "unnecessary_stop",
    "inefficient_motion",
    "unsafe_behavior",
    "other",
)


def actor_xy(raw):
    out = []
    for actor in ((raw.get("scene") or {}).get("actors") or []):
        transform = actor.get("transform") or {}
        location = transform.get("location") or {}
        try:
            out.append((float(location["x"]), float(location["y"])))
        except Exception:
            continue
    return out


def main():
    parser = argparse.ArgumentParser(description="Offline BEV review and intervention spec editor")
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--case-id", default="case_0001")
    parser.add_argument("--type", default="failure", choices=TYPES)
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    tape = BehaviorTape(args.run)
    output = Path(args.output).expanduser().resolve() if args.output else (
        PROJECT_ROOT / "cases" / (args.case_id + ".json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    xs = [frame.ego_location[0] for frame in tape.frames]
    ys = [frame.ego_location[1] for frame in tape.frames]
    rel = [frame.sim_time - tape.first_time for frame in tape.frames]

    state = {
        "index": 0,
        "record_start": 0,
        "handoff": min(len(tape.frames) - 1, max(0, len(tape.frames) // 2)),
        "record_end": len(tape.frames) - 1,
        "type": args.type,
        "case_id": args.case_id,
        "description": args.description,
    }

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_axes([0.06, 0.20, 0.62, 0.74])
    ax.plot(xs, ys, linewidth=1.5, label="Ego trajectory")
    current, = ax.plot([xs[0]], [ys[0]], marker="o", markersize=8, linestyle="None", label="Current ego")
    actors = ax.scatter([], [], s=24, marker="s", label="Nearby actors")
    mark_start, = ax.plot([], [], marker="^", markersize=9, linestyle="None", label="Record start")
    mark_handoff, = ax.plot([], [], marker="D", markersize=8, linestyle="None", label="Handoff")
    mark_end, = ax.plot([], [], marker="v", markersize=9, linestyle="None", label="Record end")
    ax.set_title("E2E Behavior Tape - Offline BEV Review")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    slider_ax = fig.add_axes([0.08, 0.10, 0.56, 0.035])
    slider = Slider(slider_ax, "Sample", 0, len(tape.frames) - 1, valinit=0, valstep=1)
    info = fig.text(0.70, 0.75, "", fontsize=10, va="top")

    type_ax = fig.add_axes([0.72, 0.43, 0.22, 0.25])
    radio = RadioButtons(type_ax, TYPES, active=TYPES.index(args.type))
    case_ax = fig.add_axes([0.72, 0.35, 0.22, 0.05])
    case_box = TextBox(case_ax, "Case", initial=args.case_id)
    desc_ax = fig.add_axes([0.72, 0.27, 0.22, 0.05])
    desc_box = TextBox(desc_ax, "Note", initial=args.description)

    start_ax = fig.add_axes([0.72, 0.19, 0.10, 0.05])
    hand_ax = fig.add_axes([0.84, 0.19, 0.10, 0.05])
    end_ax = fig.add_axes([0.72, 0.12, 0.10, 0.05])
    save_ax = fig.add_axes([0.84, 0.12, 0.10, 0.05])
    b_start = Button(start_ax, "Set Start")
    b_hand = Button(hand_ax, "Set Handoff")
    b_end = Button(end_ax, "Set End")
    b_save = Button(save_ax, "Save Spec")

    def marker(line, index):
        line.set_data([xs[index]], [ys[index]])

    def redraw(index):
        index = int(index)
        state["index"] = index
        frame = tape.frames[index]
        current.set_data([xs[index]], [ys[index]])
        points = actor_xy(frame.raw)
        if points:
            actors.set_offsets(points)
        else:
            actors.set_offsets([[float("nan"), float("nan")]])
        marker(mark_start, state["record_start"])
        marker(mark_handoff, state["handoff"])
        marker(mark_end, state["record_end"])
        c = frame.control
        info.set_text(
            "Sample: %d / %d\nRelative time: %.2f s\nSimulation time: %.3f s\n"
            "Speed: %.2f m/s\nThrottle: %.3f\nSteer: %.3f\nBrake: %.3f\n\n"
            "Record start: %.2f s\nHandoff: %.2f s\nRecord end: %.2f s"
            % (
                index,
                len(tape.frames) - 1,
                rel[index],
                frame.sim_time,
                frame.ego_speed,
                float(c.get("throttle", 0.0)),
                float(c.get("steer", 0.0)),
                float(c.get("brake", 0.0)),
                rel[state["record_start"]],
                rel[state["handoff"]],
                rel[state["record_end"]],
            )
        )
        fig.canvas.draw_idle()

    def nearest_index(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        best = min(
            range(len(xs)),
            key=lambda i: (xs[i] - event.xdata) ** 2 + (ys[i] - event.ydata) ** 2,
        )
        slider.set_val(best)

    def set_start(_):
        state["record_start"] = state["index"]
        redraw(state["index"])

    def set_handoff(_):
        state["handoff"] = state["index"]
        redraw(state["index"])

    def set_end(_):
        state["record_end"] = state["index"]
        redraw(state["index"])

    def save(_):
        start = int(state["record_start"])
        handoff = int(state["handoff"])
        end = int(state["record_end"])
        if not (start <= handoff <= end):
            print("[review] invalid order: record_start <= handoff <= record_end is required")
            return
        payload = build_intervention_spec(
            tape=tape,
            source_run=str(tape.run_dir),
            case_id=case_box.text.strip() or args.case_id,
            intervention_type=state["type"],
            description=desc_box.text.strip(),
            record_start_index=start,
            handoff_index=handoff,
            record_end_index=end,
        )
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print("[review] saved", output)

    slider.on_changed(redraw)
    fig.canvas.mpl_connect("button_press_event", nearest_index)
    b_start.on_clicked(set_start)
    b_hand.on_clicked(set_handoff)
    b_end.on_clicked(set_end)
    b_save.on_clicked(save)
    radio.on_clicked(lambda label: state.__setitem__("type", label))
    redraw(0)
    plt.show()


if __name__ == "__main__":
    main()
