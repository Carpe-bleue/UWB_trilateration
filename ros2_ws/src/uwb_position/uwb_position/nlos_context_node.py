#!/usr/bin/env python3
"""
Classifies each NLOS-flagged anchor as likely self-shadowing (the wearer's own
body blocking that anchor) vs. likely external (something else in the way —
another person, a moving obstruction).

Method: compare the bearing from the tag's current position to the flagged
anchor against the tag's heading. If the anchor sits roughly behind the
tag (opposite its facing direction), it's probably self-shadowing.

Calibration required: TAG_FORWARD_OFFSET_RAD below assumes the IMU's local
+X axis (which /odometry/filtered's heading is measured relative to, per the
anchor-frame alignment done at tag startup) points in the direction the
wearer faces. If that's not how the tag is mounted, rotate the tag in place,
watch which anchor's self-shadow classification flips as you turn, and adjust
this offset until "behind" lines up with reality.
"""

import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# Anchor positions in meters — keep in sync with uwb_to_odom.py.
ANCHORS = [
    (9.26, 0.0),
    (0.0, 0.0),
    (0.0, 3.05),
]

# Radians to add to the IMU heading to get the direction the wearer faces.
# 0.0 = IMU local +X axis is the wearer's forward direction. Tune empirically.
TAG_FORWARD_OFFSET_RAD = 0.0

# Half-width of the "behind the body" cone used to call something
# self-shadowing. 60 deg is a reasonable start given a human torso's angular
# width at typical anchor ranges.
SELF_SHADOW_HALF_ANGLE_RAD = math.radians(60.0)

# Context codes published per anchor.
LOS = 0.0
SELF_SHADOW = 1.0
EXTERNAL = 2.0


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class NLOSContextNode(Node):
    def __init__(self):
        super().__init__('nlos_context_node')

        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._have_pose = False

        self._pub = self.create_publisher(Float32MultiArray, '/uwb/nlos_context', 10)
        self.create_subscription(Odometry, '/odometry/filtered', self._on_odom, 10)
        self.create_subscription(Float32MultiArray, '/uwb/nlos', self._on_nlos, 10)

        self.get_logger().info('NLOS context node started — awaiting first pose')

    def _on_odom(self, msg: Odometry):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self._heading = 2.0 * math.atan2(qz, qw)
        self._have_pose = True

    def _on_nlos(self, msg: Float32MultiArray):
        if not self._have_pose:
            return

        forward = self._heading + TAG_FORWARD_OFFSET_RAD
        behind = forward + math.pi

        context = []
        for i, flag in enumerate(msg.data):
            if flag <= 0.5 or i >= len(ANCHORS):
                context.append(LOS)
                continue

            ax, ay = ANCHORS[i]
            bearing = math.atan2(ay - self._y, ax - self._x)
            angular_offset = abs(normalize_angle(bearing - behind))

            if angular_offset < SELF_SHADOW_HALF_ANGLE_RAD:
                context.append(SELF_SHADOW)
            else:
                context.append(EXTERNAL)

        out = Float32MultiArray()
        out.data = context
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = NLOSContextNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
