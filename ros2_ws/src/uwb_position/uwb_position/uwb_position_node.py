#!/usr/bin/env python3
"""
UWB Trilateration Visualization (ROS2 + Matplotlib GUI FIXED)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import matplotlib
matplotlib.use("TkAgg")  # MUST be before pyplot

import matplotlib.pyplot as plt
import numpy as np
import sys


class UWBVisualizer(Node):
    def __init__(self):
        super().__init__('uwb_trilateration_viz')

        # =========================
        # FIXED ANCHOR POSITIONS
        # =========================
        self.anchors = [
            (926.0, 0.0),   # A1
            (0.0, 0.0),    # A2
            (0.0, 305.0)     # A3
        ]
        self.anchor_names = ["A1", "A2", "A3"]

        self.latest_distances = None

        # trajectory
        self.trail_x = []
        self.trail_y = []

        # smoothing (EMA)
        self.alpha = 0.5
        self.fx = None
        self.fy = None

        # =========================
        # MATPLOTLIB SETUP
        # =========================
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.set_title("UWB Trilateration (Anchors centered)")
        self.ax.set_xlabel("X (cm)")
        self.ax.set_ylabel("Y (cm)")
        self.ax.grid(True)
        self.ax.set_aspect("equal", adjustable="box")

        # autoscale around anchors
        all_x = [a[0] for a in self.anchors]
        all_y = [a[1] for a in self.anchors]
        margin = 300

        self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        # draw anchors
        for (x, y), name in zip(self.anchors, self.anchor_names):
            self.ax.plot(x, y, "bo")
            self.ax.text(x + 2, y + 2, name, color="blue")

        # estimated point + trail
        self.est_point, = self.ax.plot([], [], "ro", label="Estimate")
        self.est_trail, = self.ax.plot([], [], "r-", alpha=0.6)

        self.ax.legend()

        # =========================
        # ROS SUB
        # =========================
        self.create_subscription(
            Float32MultiArray,
            "/uwb/distances",
            self.cb,
            10
        )

        self.get_logger().info("UWB visualizer started")

    def cb(self, msg):
        if len(msg.data) >= 3:
            self.latest_distances = list(map(float, msg.data[:3]))

    # =========================
    # TRILATERATION
    # =========================
    def trilaterate(self, d1, d2, d3):
        p1 = np.array(self.anchors[0])
        p2 = np.array(self.anchors[1])
        p3 = np.array(self.anchors[2])

        A = 2 * np.array([p2 - p1, p3 - p1])

        b = np.array([
            d1**2 - d2**2 - np.dot(p1, p1) + np.dot(p2, p2),
            d1**2 - d3**2 - np.dot(p1, p1) + np.dot(p3, p3)
        ])

        try:
            x, y = np.linalg.solve(A, b)
            return float(x), float(y)
        except:
            return None, None

    # =========================
    # PLOT UPDATE
    # =========================
    def update_plot(self):
        if self.latest_distances is None:
            return

        d1, d2, d3 = self.latest_distances
        x, y = self.trilaterate(d1, d2, d3)

        if x is None:
            return

        # EMA filter
        if self.fx is None:
            self.fx, self.fy = x, y
        else:
            self.fx = self.alpha * x + (1 - self.alpha) * self.fx
            self.fy = self.alpha * y + (1 - self.alpha) * self.fy

        self.trail_x.append(self.fx)
        self.trail_y.append(self.fy)

        if len(self.trail_x) > 200:
            self.trail_x.pop(0)
            self.trail_y.pop(0)

        self.est_point.set_data([self.fx], [self.fy])
        self.est_trail.set_data(self.trail_x, self.trail_y)

        self.fig.canvas.draw_idle()


def main():
    rclpy.init(args=sys.argv)
    node = UWBVisualizer()

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
