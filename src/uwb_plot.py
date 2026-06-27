import argparse
import re
import threading
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import serial

# ── CLI args ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="UWB real-time 2D plot")
parser.add_argument("--port",  default="COM3")
parser.add_argument("--baud",  default=115200, type=int)
parser.add_argument("--a1",    default="0,0")
parser.add_argument("--a2",    default="122,0")
parser.add_argument("--flip",  action="store_true")
parser.add_argument("--trail", default=40, type=int)
args = parser.parse_args()

A1 = np.array([float(v) for v in args.a1.split(",")])
A2 = np.array([float(v) for v in args.a2.split(",")])

# ── Shared state ─────────────────────────────────────────
lock = threading.Lock()
state = {"d1": None, "d2": None, "pos": None}
trail = deque(maxlen=args.trail)

# ── Trilateration ────────────────────────────────────────
def trilaterate(a1, a2, d1, d2):
    dx, dy = a2[0] - a1[0], a2[1] - a1[1]
    baseline = np.hypot(dx, dy)

    if baseline < 1e-6:
        return None

    x = (d1**2 - d2**2 + baseline**2) / (2 * baseline)
    disc = d1**2 - x**2

    if disc < 0:
        disc = 0  # tolerate noise

    y = np.sqrt(disc)
    if args.flip:
        y = -y

    ux = np.array([dx, dy]) / baseline
    uy = np.array([-dy, dx]) / baseline

    return a1 + x * ux + y * uy

# ── Serial reader ────────────────────────────────────────
PAT_AA = re.compile(r"Dist\s+AA\s*:\s*([-\d.]+)")
PAT_BB = re.compile(r"Dist\s+BB\s*:\s*([-\d.]+)")
PAT_ANCHOR = re.compile(r"Ranging\s+to\s+anchor\s+0x([0-9A-Fa-f]+)")
PAT_DIST = re.compile(r"dist\s*=\s*([-\d.]+)", re.IGNORECASE)
M_TO_CM = 100.0


def is_valid_distance(value):
    return 0.1 < value < 5000

def serial_reader():
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
        print(f"[serial] connected to {args.port}")
    except Exception as e:
        print("Serial error:", e)
        return

    last_d1 = None
    last_d2 = None
    current_anchor = None

    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue

        print("[rx]", line)

        m1 = PAT_AA.search(line)
        if m1:
            d = abs(float(m1.group(1))) * M_TO_CM
            if is_valid_distance(d):
                last_d1 = d

        m2 = PAT_BB.search(line)
        if m2:
            d = abs(float(m2.group(1))) * M_TO_CM
            if is_valid_distance(d):
                last_d2 = d

        ma = PAT_ANCHOR.search(line)
        if ma:
            current_anchor = ma.group(1).upper()

        md = PAT_DIST.search(line)
        if md and current_anchor is not None:
            d = abs(float(md.group(1))) * M_TO_CM
            if is_valid_distance(d):
                if current_anchor.endswith("AA"):
                    last_d1 = d
                elif current_anchor.endswith("BB"):
                    last_d2 = d

        if last_d1 is None and last_d2 is None:
            continue

        pos = trilaterate(A1, A2, last_d1, last_d2) if (last_d1 is not None and last_d2 is not None) else None

        with lock:
            state["d1"] = last_d1
            state["d2"] = last_d2
            state["pos"] = pos

            if pos is not None:
                trail.append(pos.copy())

        print(f"[pos] ({pos[0]:.2f}, {pos[1]:.2f})" if pos is not None else "[pos] None")

threading.Thread(target=serial_reader, daemon=True).start()

# ── Plot ────────────────────────────────────────────────
fig, ax = plt.subplots()

trail_line, = ax.plot([], [], "-", alpha=0.3)
tag_dot, = ax.plot([], [], "o")

circle1 = plt.Circle(A1, 0, fill=False, linestyle="--")
circle2 = plt.Circle(A2, 0, fill=False, linestyle="--")
ax.add_patch(circle1)
ax.add_patch(circle2)

ax.plot(*A1, "s", label="Anchor AA")
ax.plot(*A2, "s", label="Anchor BB")

info = ax.text(0.02, 0.95, "Waiting for data...", transform=ax.transAxes)

ax.set_aspect("equal")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.grid()
ax.legend()

def update():
    with lock:
        d1 = state["d1"]
        d2 = state["d2"]
        pos = state["pos"]
        pts = list(trail)

    if pos is not None:
        tag_dot.set_data([pos[0]], [pos[1]])
    else:
        tag_dot.set_data([], [])

    if len(pts) > 1:
        xs, ys = zip(*pts)
        trail_line.set_data(xs, ys)
    else:
        trail_line.set_data([], [])

    if d1 is not None:
        circle1.set_radius(d1)
    if d2 is not None:
        circle2.set_radius(d2)

    if pos is not None and d1 is not None and d2 is not None:
        info.set_text(f"d1={d1:.1f} cm  d2={d2:.1f} cm\n({pos[0]:.1f}, {pos[1]:.1f}) cm")
    elif d1 is not None or d2 is not None:
        d1_txt = f"{d1:.1f} cm" if d1 is not None else "--"
        d2_txt = f"{d2:.1f} cm" if d2 is not None else "--"
        info.set_text(f"d1={d1_txt}  d2={d2_txt}\nWaiting for pair...")
    else:
        info.set_text("Waiting for data...")

    ax.relim()
    ax.autoscale_view()

    fig.canvas.draw_idle()

timer = fig.canvas.new_timer(interval=200)
timer.add_callback(update)
timer.start()

plt.show()