#!/usr/bin/env python3
import time
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ================== CONFIG ==================
# Anchor positions (cm)
anchors = [(0.0, 0.0), (120.0, 0.0), (0.0, 120.0)]
anchor_names = ["A1", "A2", "A3"]

# Simulation parameters
center_x = 60.0
center_y = 60.0
radius = 30.0          # how big the circular path is
speed = 0.3            # rad per second (change for faster/slower movement)

# Plot settings
plt.ion()  # interactive mode for live update
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-20, 160)
ax.set_ylim(-20, 160)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_title('UWB Trilateration Simulation - 2D Live View', fontsize=14)
ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')

# Draw fixed anchors once
anchor_x = [a[0] for a in anchors]
anchor_y = [a[1] for a in anchors]
ax.plot(anchor_x, anchor_y, 'bo', markersize=12, label='Anchors')
for i, name in enumerate(anchor_names):
    ax.text(anchor_x[i] + 5, anchor_y[i] + 5, name, fontsize=12, fontweight='bold', color='blue')

# Create moving objects (will be updated live)
true_point, = ax.plot([], [], 'go', markersize=10, label='True Tag Position')
est_point,  = ax.plot([], [], 'ro', markersize=10, label='Estimated Position (Trilateration)')
true_trail, = ax.plot([], [], 'g-', linewidth=1.5, alpha=0.4)
est_trail,  = ax.plot([], [], 'r-', linewidth=1.5, alpha=0.4)

ax.legend(loc='upper right')
plt.tight_layout()

# Storage for trails
true_trail_x = []
true_trail_y = []
est_trail_x = []
est_trail_y = []

def trilaterate(d1: float, d2: float, d3: float) -> tuple[float, float]:
    """Exact 2D trilateration for our 3 anchors"""
    a = 120.0
    x = (d1**2 - d2**2 + a**2) / (2 * a)
    y = (d1**2 - d3**2 + a**2) / (2 * a)
    return x, y

def update(frame):
    global t
    t += speed

    # Move tag in a circle
    tag_x = center_x + radius * math.sin(t)
    tag_y = center_y + radius * math.cos(t)

    # Compute real distances to each anchor
    distances = [math.hypot(tag_x - ax, tag_y - ay) for ax, ay in anchors]

    # Trilateration (recover position from distances)
    est_x, est_y = trilaterate(distances[0], distances[1], distances[2])

    # Update trails
    true_trail_x.append(tag_x)
    true_trail_y.append(tag_y)
    est_trail_x.append(est_x)
    est_trail_y.append(est_y)
    if len(true_trail_x) > 50:  # keep last 50 points only
        true_trail_x.pop(0)
        true_trail_y.pop(0)
        est_trail_x.pop(0)
        est_trail_y.pop(0)

    # Update plot objects
    true_point.set_data([tag_x], [tag_y])
    est_point.set_data([est_x], [est_y])
    true_trail.set_data(true_trail_x, true_trail_y)
    est_trail.set_data(est_trail_x, est_trail_y)

    # Console output (same as before)
    print(f"\nTime: {t:5.1f}s  |  True: ({tag_x:6.1f}, {tag_y:6.1f}) cm")
    print(f"Distances → A1: {distances[0]:6.2f}  A2: {distances[1]:6.2f}  A3: {distances[2]:6.2f} cm")
    print(f"Estimated: ({est_x:6.1f}, {est_y:6.1f}) cm   ← Trilateration works perfectly!")

    return true_point, est_point, true_trail, est_trail


# =============== START SIMULATION ===============
print("🚀 UWB 2D Live Visualization started!")
print("Green = True tag position")
print("Red   = Estimated position (from distances)")
print("Blue  = Fixed anchors")
print("Press Ctrl+C in the terminal to stop.\n")

t = 0.0
ani = FuncAnimation(fig, update, interval=1000, blit=False, cache_frame_data=False)

try:
    plt.show(block=True)
except KeyboardInterrupt:
    print("\n\nSimulation stopped. Bye! 👋")
