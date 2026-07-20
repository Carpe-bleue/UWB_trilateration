#!/usr/bin/env python3
"""
UWB Vehicle Pose Estimation & Visualization - Moving Anchors (Vehicle) + Fixed Tag
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import sys


class UWBVehicleVisualizer(Node):
    def __init__(self):
        super().__init__('uwb_vehicle_pose_viz')
      
        # Fixed tag position (world frame) - at (0, 0)
        self.tag_pos = np.array([0.0, 0.0])
      
        # Local anchor positions relative to vehicle center (cm)
        self.local_anchors = [
            np.array([-30.0, 43.0]),  # Board 1 (front left)
            np.array([30.0, 43.0]),   # Board 2 (front right)
            np.array([0.0, -43.0])    # Board 3 (rear)
        ]
        self.anchor_names = ["B1", "B2", "B3"]
      
        self.latest_distances = None
      
        # Pose history for trail (vehicle center)
        self.est_trail_x = []
        self.est_trail_y = []
      
        # Smoothing parameters
        self.alpha = 0.4  # EMA smoothing for position
        self.filtered_tx = None
        self.filtered_ty = None
        self.filtered_theta = None
      
        # === Matplotlib Setup - Robust for ROS 2 ===
        plt.ion()
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Force GUI backend
            self.get_logger().info(f"Matplotlib backend set to: {plt.get_backend()}")
        except Exception as e:
            self.get_logger().warn(f"Could not set TkAgg backend: {e}")

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-700, 700)
        self.ax.set_ylim(-700, 700)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title('UWB Vehicle Pose Estimation (Fixed Tag)', fontsize=14)
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
      
        # Fixed Tag
        self.tag_plot, = self.ax.plot([0], [0], 'g*', markersize=18, label='Fixed Tag')
        self.ax.text(20, 20, 'TAG', fontsize=12, fontweight='bold', color='green')
      
        # Vehicle elements
        self.anchor_plots = []
        for i in range(3):
            plot, = self.ax.plot([], [], 'bo', markersize=12, label=self.anchor_names[i] if i == 0 else "")
            self.anchor_plots.append(plot)
       
        self.vehicle_lines, = self.ax.plot([], [], 'b-', linewidth=3, alpha=0.8, label='Vehicle')
        self.center_point, = self.ax.plot([], [], 'ro', markersize=10, label='Vehicle Center')
        self.est_trail, = self.ax.plot([], [], 'r-', linewidth=2.5, alpha=0.6, label='Center Trail')
       
        # Heading arrow (will be recreated)
        self.heading_arrow = self.ax.arrow(0, 0, 0, 0, head_width=15, head_length=25,
                                         fc='red', ec='red', alpha=0.9)
       
        self.ax.legend(loc='upper right')
        plt.tight_layout()
      
        # Force window to appear
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)  # Critical for opening the window

        # Timer for plot updates
        self.create_timer(0.1, self.update_plot)  # ~10 Hz
      
        # Subscriber
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/uwb/distances',
            self.distances_callback,
            10)
      
        self.get_logger().info('UWB Vehicle Pose Visualizer started (Fixed Tag + Moving Anchors)')
        self.get_logger().info('Local anchors: B1(-30,43), B2(30,43), B3(0,-43) cm')

    def destroy_node(self):
        """Clean up matplotlib when node is destroyed"""
        plt.close('all')
        super().destroy_node()

    def distances_callback(self, msg):
        if len(msg.data) >= 3:
            self.latest_distances = [float(d) for d in msg.data[:3]]

    def estimate_pose(self, distances):
        """Least squares optimization to find (tx, ty, theta)"""
        def residuals(params):
            tx, ty, theta = params
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
            res = []
            for local, d_meas in zip(self.local_anchors, distances):
                world = R @ local + np.array([tx, ty])
                d_pred = np.linalg.norm(world - self.tag_pos)
                res.append(d_pred - d_meas)
            return res

        # Initial guess
        if self.filtered_tx is not None:
            x0 = [self.filtered_tx, self.filtered_ty, self.filtered_theta or 0.0]
        else:
            avg_dist = np.mean(distances)
            x0 = [0.0, avg_dist, 0.0]

        try:
            result = least_squares(residuals, x0, method='lm', ftol=1e-8, xtol=1e-8)
            if result.success:
                tx, ty, theta = result.x
                # Normalize theta
                theta = (theta + np.pi) % (2 * np.pi) - np.pi
                return tx, ty, theta
            else:
                return None, None, None
        except Exception:
            return None, None, None

    def compute_world_anchors(self, tx, ty, theta):
        """Compute world positions of anchors given pose"""
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        world_anchors = []
        for local in self.local_anchors:
            world = R @ local + np.array([tx, ty])
            world_anchors.append(world)
        return world_anchors

    def update_plot(self):
        if self.latest_distances is None:
            # Still update GUI so window stays visible
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            return
          
        distances = self.latest_distances
        tx, ty, theta = self.estimate_pose(distances)
      
        if tx is None or ty is None:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            return

        # === EMA Smoothing ===
        if self.filtered_tx is None:
            self.filtered_tx = tx
            self.filtered_ty = ty
            self.filtered_theta = theta
        else:
            self.filtered_tx = self.alpha * tx + (1 - self.alpha) * self.filtered_tx
            self.filtered_ty = self.alpha * ty + (1 - self.alpha) * self.filtered_ty
            self.filtered_theta = self.alpha * theta + (1 - self.alpha) * self.filtered_theta

        tx_f, ty_f, theta_f = self.filtered_tx, self.filtered_ty, self.filtered_theta

        # Compute current world anchor positions
        world_anchors = self.compute_world_anchors(tx_f, ty_f, theta_f)

        # Update anchor markers
        for i, world in enumerate(world_anchors):
            self.anchor_plots[i].set_data([world[0]], [world[1]])

        # Update vehicle body
        x_coords = [p[0] for p in world_anchors] + [world_anchors[0][0]]
        y_coords = [p[1] for p in world_anchors] + [world_anchors[0][1]]
        self.vehicle_lines.set_data(x_coords, y_coords)

        # Update center
        self.center_point.set_data([tx_f], [ty_f])
       
        # Update trail
        self.est_trail_x.append(tx_f)
        self.est_trail_y.append(ty_f)
        if len(self.est_trail_x) > 200:
            self.est_trail_x.pop(0)
            self.est_trail_y.pop(0)
        self.est_trail.set_data(self.est_trail_x, self.est_trail_y)

        # Update heading arrow
        arrow_length = 60
        dx = arrow_length * np.cos(theta_f + np.pi/2)
        dy = arrow_length * np.sin(theta_f + np.pi/2)
        self.heading_arrow.remove()
        self.heading_arrow = self.ax.arrow(tx_f, ty_f, dx, dy,
                                         head_width=18, head_length=28,
                                         fc='red', ec='darkred', alpha=0.9)

        # Console output
        print(f"Dist B1: {distances[0]:6.1f} | B2: {distances[1]:6.1f} | B3: {distances[2]:6.1f} cm → "
              f"Center: ({tx_f:7.1f}, {ty_f:7.1f}) cm | Θ: {np.degrees(theta_f):6.1f}°", end="\r")

        # Update plot
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

def main():
    rclpy.init(args=sys.argv)
    node = UWBVehicleVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass
        plt.close('all')


if __name__ == '__main__':
    main()
