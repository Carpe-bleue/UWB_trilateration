#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
import numpy as np

class UWBToOdom(Node):
    def __init__(self):
        super().__init__('uwb_to_odom')

        # Anchor positions in cm (match your physical layout)
        self.anchors = [
            np.array([926.0, 0.0]),
            np.array([0.0,   0.0]),
            np.array([0.0,   305.0])
        ]

        # Smoothed state (in meters, consistent with ROS)
        self.last_x = None
        self.last_y = None
        self.last_time = None
        self.alpha = 0.3  # EWM smoothing — lower = smoother but more lag

        # UWB measurement noise std dev (tune to your hardware, in meters)
        self.pos_std = 0.15  # ~15 cm 1-sigma

        # Latest per-anchor NLOS flags (0.0=LOS, 1.0=NLOS), published by the
        # tag alongside /uwb/distances. Used to inflate position covariance
        # below instead of trusting every fix equally.
        self.nlos_flags = [0.0, 0.0, 0.0]
        # Multiply pos_std by this factor based on how many anchors in the
        # current fix are NLOS-flagged and/or stale. 0 bad -> trust as normal.
        self.nlos_std_inflation = {0: 1.0, 1: 3.0, 2: 8.0, 3: 8.0}

        # Milliseconds since each anchor's last successful ranging round,
        # published by the tag. A timed-out anchor keeps republishing its
        # last known distance forever with no other indication it's stale,
        # so anything older than this threshold is treated as untrustworthy.
        self.anchor_age_ms = [0.0, 0.0, 0.0]
        self.stale_threshold_ms = 500.0

        self.odom_pub = self.create_publisher(Odometry, '/uwb/odom', 10)
        self.create_subscription(Float32MultiArray, '/uwb/distances',
                                 self.distances_callback, 10)
        self.create_subscription(Float32MultiArray, '/uwb/nlos',
                                 self.nlos_callback, 10)
        self.create_subscription(Float32MultiArray, '/uwb/anchor_age',
                                 self.anchor_age_callback, 10)
        self.get_logger().info("UWB → Odometry node started")

    def nlos_callback(self, msg):
        self.nlos_flags = list(msg.data)

    def anchor_age_callback(self, msg):
        self.anchor_age_ms = list(msg.data)

    def trilaterate(self, d1, d2, d3):
        """
        Least-squares trilateration. Distances in same units as anchors (cm).
        Returns (x, y) in cm, or (None, None) on failure.
        """
        p1, p2, p3 = self.anchors
        A = 2.0 * np.array([p2 - p1, p3 - p1], dtype=np.float64)
        b = np.array([
            d1**2 - d2**2 - np.dot(p1, p1) + np.dot(p2, p2),
            d1**2 - d3**2 - np.dot(p1, p1) + np.dot(p3, p3)
        ], dtype=np.float64)

        # Degenerate geometry check
        if abs(np.linalg.det(A)) < 1e-6:
            return None, None
        try:
            xy = np.linalg.solve(A, b)
            x, y = float(xy[0]), float(xy[1])
            if not np.isfinite(x) or not np.isfinite(y):
                return None, None
            if abs(x) > 20000 or abs(y) > 20000:   # sanity: 200 m max
                return None, None
            return x, y
        except np.linalg.LinAlgError:
            return None, None

    def distances_callback(self, msg):
        if len(msg.data) < 3:
            return

        d1, d2, d3 = float(msg.data[0]), float(msg.data[1]), float(msg.data[2])
        raw_x_cm, raw_y_cm = self.trilaterate(d1, d2, d3)
        if raw_x_cm is None:
            return

        # Convert to meters immediately — work in SI from here on
        raw_x = raw_x_cm / 100.0
        raw_y = raw_y_cm / 100.0

        now = self.get_clock().now()
        current_time = now.nanoseconds / 1e9

        if self.last_x is None:
            # First valid measurement — initialise filter, publish zero velocity
            self.last_x = raw_x
            self.last_y = raw_y
            self.last_time = current_time
            vx, vy = 0.0, 0.0
        else:
            # Apply EWM smoothing to the raw position
            smooth_x = self.alpha * raw_x + (1.0 - self.alpha) * self.last_x
            smooth_y = self.alpha * raw_y + (1.0 - self.alpha) * self.last_y

            dt = current_time - self.last_time
            if dt > 0.01:   # only estimate velocity if dt is meaningful (>10 ms)
                vx = (smooth_x - self.last_x) / dt
                vy = (smooth_y - self.last_y) / dt
            else:
                vx, vy = 0.0, 0.0

            # Clamp absurd velocities (> 5 m/s) — likely a bad measurement
            max_v = 5.0
            if abs(vx) > max_v or abs(vy) > max_v:
                vx, vy = 0.0, 0.0
                smooth_x = self.last_x
                smooth_y = self.last_y

            self.last_x = smooth_x
            self.last_y = smooth_y
            self.last_time = current_time

        # --- Build Odometry message ---
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = self.last_x
        odom.pose.pose.position.y = self.last_y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.w = 1.0
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = 0.0

        # Pose covariance: 6x6, row-major [x, y, z, roll, pitch, yaw]
        # Only x and y are measured; everything else is unconstrained.
        # Inflate the reported variance when any anchor in this fix was
        # NLOS-flagged or stale (timed out and just repeating an old
        # value), so the ESKF trusts a compromised fix less. Counted per
        # anchor (not two separate totals) so an anchor that's both NLOS
        # and stale isn't double-counted, and two different bad anchors
        # (one NLOS, one stale) aren't undercounted.
        n_bad = sum(
            1 for i in range(len(self.anchors))
            if (i < len(self.nlos_flags) and self.nlos_flags[i] > 0.5)
            or (i < len(self.anchor_age_ms) and self.anchor_age_ms[i] > self.stale_threshold_ms)
        )
        inflation = self.nlos_std_inflation.get(n_bad, 8.0)
        ps = (self.pos_std * inflation) ** 2   # position variance (m^2)
        BIG = 9999.0             # "not measured" variance
        odom.pose.covariance = [
            ps,    0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,   ps,   0.0,  0.0,  0.0,  0.0,
            0.0,   0.0,  BIG,  0.0,  0.0,  0.0,
            0.0,   0.0,  0.0,  BIG,  0.0,  0.0,
            0.0,   0.0,  0.0,  0.0,  BIG,  0.0,
            0.0,   0.0,  0.0,  0.0,  0.0,  BIG,
        ]

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = 0.0

        # Twist covariance: velocity is derived/noisy; tell the EKF not to trust it much
        vv = 0.5   # velocity variance (m/s)^2
        odom.twist.covariance = [
            vv,    0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,   vv,   0.0,  0.0,  0.0,  0.0,
            0.0,   0.0,  BIG,  0.0,  0.0,  0.0,
            0.0,   0.0,  0.0,  BIG,  0.0,  0.0,
            0.0,   0.0,  0.0,  0.0,  BIG,  0.0,
            0.0,   0.0,  0.0,  0.0,  0.0,  BIG,
        ]

        self.odom_pub.publish(odom)
        self.get_logger().info(
            f"UWB: x={self.last_x:.3f}m y={self.last_y:.3f}m "
            f"vx={vx:.3f}m/s vy={vy:.3f}m/s"
        )


def main(args=None):
    rclpy.init(args=args)
    node = UWBToOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
