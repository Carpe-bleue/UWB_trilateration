#!/usr/bin/env python3
"""
Fused Odometry Visualization (ROS2 + Matplotlib GUI)

Listens to /odometry/filtered (output of robot_localization EKF/UKF) and
plots the fused position, heading, and trajectory trail. Also overlays the
raw UWB anchors for visual reference, matching the layout used elsewhere.
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

        # trajectory trail
        self.trail_x = []
        self.trail_y = []
        self.max_trail = 500

        # =========================
        # MATPLOTLIB SETUP
        # =========================
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.set_title("Fused Odometry (/odometry/filtered)")
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

        # fused estimate point, heading arrow, and trail
        self.est_point, = self.ax.plot([], [], "go", markersize=8, label="Fused estimate")
        self.est_trail, = self.ax.plot([], [], "g-", alpha=0.6, label="Trajectory")
        self.heading_arrow = self.ax.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2)
        )

        # legend proxies for the link colors (real link lines start empty)
        self.ax.plot([], [], "-", color="gray", alpha=0.4, linewidth=1, label="LOS link")
        self.ax.plot([], [], "-", color="red", linewidth=2, label="NLOS link")

        self.ax.legend(loc="upper right")

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

        self.get_logger().info("Fused odometry visualizer started, listening on /odometry/filtered")

    def nlos_cb(self, msg: Float32MultiArray):
        self.latest_nlos = list(msg.data)

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

        self.trail_x.append(x)
        self.trail_y.append(y)
        if len(self.trail_x) > self.max_trail:
            self.trail_x.pop(0)
            self.trail_y.pop(0)

        self.est_point.set_data([x], [y])
        self.est_trail.set_data(self.trail_x, self.trail_y)

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
                f"Fused Odometry  |  \u03c3x={sx:.3f}m  \u03c3y={sy:.3f}m"
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
