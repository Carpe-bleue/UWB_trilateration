#!/usr/bin/env python3
"""
Fused Odometry Visualization (ROS2 + Matplotlib GUI)

Demo visualizer contrasting two position estimates on one plot:
  - "Naive" (orange): raw per-message trilateration straight off
    /uwb/distances with light EMA smoothing only -- no NLOS awareness,
    trusts every anchor equally. Same math as uwb_position_node.py.
  - "Fused" (green): /odometry/filtered, the NLOS-aware ESKF/EKF/UKF
    output -- covariance-gated against /uwb/nlos and /uwb/anchor_age.

Watch them diverge when someone steps into an anchor's line of sight: the
naive dot jumps/glitches, the fused dot barely reacts, and the anchor link
line to the blocked anchor turns red at the same moment.

The fused trail is also color-coded per point (green/red) by whether any
anchor was NLOS-flagged when that point was recorded, so walking a loop
builds up a visual map of where NLOS tends to happen.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

import matplotlib
matplotlib.use("TkAgg")  # MUST be before pyplot

import matplotlib.pyplot as plt
import numpy as np
import sys
import math


class FusedOdomVisualizer(Node):
    def __init__(self):
        super().__init__('fused_odom_viz')

        # =========================
        # FIXED ANCHOR POSITIONS (cm) — same layout as uwb_position_node.py,
        # shown here only as a spatial reference, converted to meters since
        # /odometry/filtered publishes in meters.
        # =========================
        self.anchors_cm = [
            (926.0, 0.0),   # A1
            (0.0, 0.0),     # A2
            (0.0, 305.0),   # A3
        ]
        self.anchor_names = ["A1", "A2", "A3"]
        self.anchors_m = [(x / 100.0, y / 100.0) for x, y in self.anchors_cm]

        # latest fused state
        self.latest_pose = None       # (x, y) in meters
        self.latest_yaw = 0.0         # radians
        self.latest_pos_cov = None    # (sigma_x, sigma_y) for a simple ellipse hint

        # latest per-anchor NLOS flags (0.0=LOS, 1.0=NLOS), same order as anchors_m
        self.latest_nlos = [0.0] * len(self.anchors_cm)

        self.max_trail = 500

        # fused trail: parallel arrays, trail_colors[i] is "green"/"red" based
        # on whether any anchor was NLOS-flagged when trail_x[i]/trail_y[i]
        # was recorded
        self.trail_x = []
        self.trail_y = []
        self.trail_colors = []

        # naive (NLOS-unaware) estimate: computed here straight from
        # /uwb/distances, same trilateration + EMA as uwb_position_node.py
        self.naive_alpha = 0.5
        self._naive_x = None
        self._naive_y = None
        self.naive_trail_x = []
        self.naive_trail_y = []

        # =========================
        # MATPLOTLIB SETUP
        # =========================
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.set_title("Naive vs NLOS-aware Fused Position")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True)
        self.ax.set_aspect("equal", adjustable="box")

        # autoscale around anchors (converted to meters)
        all_x = [a[0] for a in self.anchors_m]
        all_y = [a[1] for a in self.anchors_m]
        margin = 3.0  # meters

        self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        # draw anchors for reference
        self.anchor_markers = []
        for (x, y), name in zip(self.anchors_m, self.anchor_names):
            marker, = self.ax.plot(x, y, "bo")
            self.anchor_markers.append(marker)
            self.ax.text(x + 0.05, y + 0.05, name, color="blue")

        # tag-to-anchor links: gray/thin when LOS, red/thick when NLOS-flagged
        self.anchor_links = [
            self.ax.plot([], [], "-", color="gray", alpha=0.4, linewidth=1)[0]
            for _ in self.anchors_m
        ]

        # naive estimate point + trail (no NLOS awareness, for comparison)
        self.naive_point, = self.ax.plot([], [], "o", color="orange", markersize=8, label="Naive estimate")
        self.naive_trail_line, = self.ax.plot([], [], "-", color="orange", alpha=0.5, label="Naive trail")

        # fused estimate point, heading arrow, and trail. Trail is a thin
        # connecting line (path continuity) plus a scatter of dots colored
        # per-point by NLOS state at the time (idea #2).
        self.est_point, = self.ax.plot([], [], "go", markersize=8, label="Fused estimate")
        self.est_trail_line, = self.ax.plot([], [], "-", color="dimgray", alpha=0.3, linewidth=1)
        self.trail_scatter = self.ax.scatter([], [], s=14, zorder=5)
        self.heading_arrow = self.ax.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2)
        )

        # legend proxies (real elements above either start empty or don't
        # individually carry the label needed for a clean legend entry)
        self.ax.plot([], [], "-", color="gray", alpha=0.4, linewidth=1, label="LOS link")
        self.ax.plot([], [], "-", color="red", linewidth=2, label="NLOS link")
        self.ax.scatter([], [], color="green", s=14, label="Fused trail (LOS)")
        self.ax.scatter([], [], color="red", s=14, label="Fused trail (NLOS)")

        self.ax.legend(loc="upper right", fontsize=8)

        # =========================
        # ROS SUB
        # =========================
        self.create_subscription(
            Odometry,
            "/odometry/filtered",
            self.cb,
            10
        )
        self.create_subscription(
            Float32MultiArray,
            "/uwb/nlos",
            self.nlos_cb,
            10
        )
        self.create_subscription(
            Float32MultiArray,
            "/uwb/distances",
            self.distances_cb,
            10
        )

        self.get_logger().info("Fused odometry visualizer started, listening on /odometry/filtered")

    def nlos_cb(self, msg: Float32MultiArray):
        self.latest_nlos = list(msg.data)

    def trilaterate(self, d1, d2, d3):
        """Least-squares trilateration in meters. Same math as
        uwb_position_node.py -- no NLOS awareness, trusts every anchor
        equally. Returns (x, y) or (None, None) on degenerate geometry."""
        p1, p2, p3 = (np.array(a) for a in self.anchors_m)
        A = 2.0 * np.array([p2 - p1, p3 - p1])
        b = np.array([
            d1**2 - d2**2 - np.dot(p1, p1) + np.dot(p2, p2),
            d1**2 - d3**2 - np.dot(p1, p1) + np.dot(p3, p3)
        ])
        try:
            x, y = np.linalg.solve(A, b)
            return float(x), float(y)
        except np.linalg.LinAlgError:
            return None, None

    def distances_cb(self, msg: Float32MultiArray):
        if len(msg.data) < 3:
            return
        d1, d2, d3 = (v / 100.0 for v in msg.data[:3])  # cm -> m
        x, y = self.trilaterate(d1, d2, d3)
        if x is None:
            return

        if self._naive_x is None:
            self._naive_x, self._naive_y = x, y
        else:
            self._naive_x = self.naive_alpha * x + (1.0 - self.naive_alpha) * self._naive_x
            self._naive_y = self.naive_alpha * y + (1.0 - self.naive_alpha) * self._naive_y

    def cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # extract yaw from quaternion (2D, only z/w matter for planar yaw)
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)

        # pose covariance is row-major 6x6; x variance is index 0, y variance is index 7
        cov = msg.pose.covariance
        sigma_x = math.sqrt(max(cov[0], 0.0))
        sigma_y = math.sqrt(max(cov[7], 0.0))

        self.latest_pose = (x, y)
        self.latest_yaw = yaw
        self.latest_pos_cov = (sigma_x, sigma_y)

    # =========================
    # PLOT UPDATE
    # =========================
    def update_plot(self):
        if self.latest_pose is None:
            return

        x, y = self.latest_pose

        # fused trail: record this point's color by NLOS state right now
        any_nlos = any(f > 0.5 for f in self.latest_nlos)
        self.trail_x.append(x)
        self.trail_y.append(y)
        self.trail_colors.append("red" if any_nlos else "green")
        if len(self.trail_x) > self.max_trail:
            self.trail_x.pop(0)
            self.trail_y.pop(0)
            self.trail_colors.pop(0)

        self.est_point.set_data([x], [y])
        self.est_trail_line.set_data(self.trail_x, self.trail_y)
        if self.trail_x:
            self.trail_scatter.set_offsets(np.column_stack([self.trail_x, self.trail_y]))
            self.trail_scatter.set_color(self.trail_colors)

        # naive estimate + trail, for direct comparison against the fused one
        if self._naive_x is not None:
            self.naive_trail_x.append(self._naive_x)
            self.naive_trail_y.append(self._naive_y)
            if len(self.naive_trail_x) > self.max_trail:
                self.naive_trail_x.pop(0)
                self.naive_trail_y.pop(0)
            self.naive_point.set_data([self._naive_x], [self._naive_y])
            self.naive_trail_line.set_data(self.naive_trail_x, self.naive_trail_y)

        # tag-to-anchor links, colored red when that anchor is NLOS-flagged
        for i, (ax_pos, link, marker) in enumerate(
            zip(self.anchors_m, self.anchor_links, self.anchor_markers)
        ):
            ax_x, ax_y = ax_pos
            link.set_data([x, ax_x], [y, ax_y])
            is_nlos = i < len(self.latest_nlos) and self.latest_nlos[i] > 0.5
            if is_nlos:
                link.set_color("red")
                link.set_linewidth(2)
                link.set_alpha(0.9)
                marker.set_color("red")
            else:
                link.set_color("gray")
                link.set_linewidth(1)
                link.set_alpha(0.4)
                marker.set_color("blue")

        # heading arrow: short fixed-length line in the direction of yaw
        arrow_len = 0.3  # meters
        dx = arrow_len * math.cos(self.latest_yaw)
        dy = arrow_len * math.sin(self.latest_yaw)
        self.heading_arrow.set_position((x, y))
        self.heading_arrow.xy = (x + dx, y + dy)

        # update title with live uncertainty readout
        if self.latest_pos_cov is not None:
            sx, sy = self.latest_pos_cov
            self.ax.set_title(
                f"Naive vs NLOS-aware Fused Position  |  \u03c3x={sx:.3f}m  \u03c3y={sy:.3f}m"
            )

        self.fig.canvas.draw_idle()


def main():
    rclpy.init(args=sys.argv)
    node = FusedOdomVisualizer()

    try:
        plt.show(block=False)

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.update_plot()
            plt.pause(0.001)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close("all")


if __name__ == "__main__":
    main()
