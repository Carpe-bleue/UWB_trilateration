#!/usr/bin/env python3
"""
Error-State Kalman Filter (ESKF) — UWB + IMU fusion for a 2-D ground robot.

Architecture
------------
Nominal state  x̂ = [px, py, vx, vy, θ, bgz, bax, bay]
  Propagated at full IMU rate by exact kinematic mechanization (no noise term).

Error state   δx = [δpx, δpy, δvx, δvy, δθ, δbgz, δbax, δbay]  (N = 8)
  A small perturbation tracked by the KF.  Always near zero → linearization
  stays accurate even for large robot motions.

Correction (UWB /uwb/odom, ~5–10 Hz)
  1. Compute innovation: innov = z_uwb − p̂
  2. Standard KF update on δx.
  3. Inject δx into x̂, then reset δx → 0 (covariance P stays).

Propagation (IMU /imu/data, ~100 Hz)
  1. Remove estimated biases from raw readings.
  2. Integrate x̂ with Euler step.
  3. Propagate P with linearized error dynamics F.

Publishes /odometry/filtered — same topic as robot_localization, drop-in
compatible with fused_odom_node.py visualizer.

Noise tuning knobs (all 1-sigma)
  _SIG_ACC  : accelerometer white noise    (m/s²)
  _SIG_GYRO : gyro white noise             (rad/s)
  _SIG_BA   : accel bias random walk       (m/s²/√s)
  _SIG_BG   : gyro bias random walk        (rad/s/√s)

UWB measurement noise is not a fixed constant here: it's read per-message
from /uwb/odom's pose.covariance, which uwb_to_odom.py inflates when an
anchor in the fix is NLOS-flagged (see /uwb/nlos).
"""

import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu

N = 8  # error-state dimension

_SIG_ACC  = 0.3    # m/s²      accelerometer white noise
_SIG_GYRO = 0.02   # rad/s     gyro white noise
_SIG_BA   = 5e-5   # m/s²/√s  accel bias random walk
_SIG_BG   = 1e-5   # rad/s/√s gyro bias random walk


class ESKFNode(Node):
    def __init__(self):
        super().__init__('eskf_node')

        # ── Nominal state ─────────────────────────────────────────────────────
        self._px = self._py = 0.0      # position (m)
        self._vx = self._vy = 0.0      # velocity (m/s)
        self._th = 0.0                 # heading θ (rad)
        self._bgz = 0.0               # gyro-z bias (rad/s)
        self._bax = self._bay = 0.0   # accel x/y biases (m/s²)

        # ── Error-state covariance P (8×8) ───────────────────────────────────
        # Starts with large uncertainty on velocity and biases (unobservable
        # until the filter has integrated enough IMU + UWB data).
        self._P = np.diag([
            0.10, 0.10,   # δp  (m²)
            0.50, 0.50,   # δv  (m²/s²)
            0.10,         # δθ  (rad²)
            0.05,         # δbgz
            0.10, 0.10,   # δba
        ]).astype(np.float64)

        # ── Measurement matrix H: z = H δx  (observes δpx, δpy) ─────────────
        self._H = np.zeros((2, N))
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0

        self._last_imu_t: float | None = None
        self._ready = False   # wait for first UWB fix before integrating IMU

        self._pub = self.create_publisher(Odometry, '/odometry/filtered', 10)
        self.create_subscription(Imu, '/imu/data', self._on_imu, 100)
        self.create_subscription(Odometry, '/uwb/odom', self._on_uwb, 10)

        self.get_logger().info('ESKF node started — awaiting first UWB fix')

    # ──────────────────────────────────────────────────────────────────────────
    # IMU callback: high-rate nominal propagation + covariance prediction
    # ──────────────────────────────────────────────────────────────────────────
    def _on_imu(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_imu_t is None:
            self._last_imu_t = t
            return
        dt = t - self._last_imu_t
        self._last_imu_t = t
        if dt <= 0.0 or dt > 0.5 or not self._ready:
            return

        # Bias-corrected IMU readings
        ax_b = msg.linear_acceleration.x - self._bax
        ay_b = msg.linear_acceleration.y - self._bay
        wz   = msg.angular_velocity.z    - self._bgz

        c, s = math.cos(self._th), math.sin(self._th)

        # Rotate body-frame acceleration into world frame
        ax_w = c * ax_b - s * ay_b
        ay_w = s * ax_b + c * ay_b

        # ── Nominal state integration (Euler, accurate at ~100 Hz) ───────────
        self._px += self._vx * dt + 0.5 * ax_w * dt * dt
        self._py += self._vy * dt + 0.5 * ay_w * dt * dt
        self._vx += ax_w * dt
        self._vy += ay_w * dt
        self._th += wz * dt
        # biases modelled as constant — only KF correction updates them

        # ── Error-state transition  F = I + Fc·dt ────────────────────────────
        # Fc is the continuous-time Jacobian of the error dynamics.
        # State order: [δpx(0), δpy(1), δvx(2), δvy(3), δθ(4), δbgz(5), δbax(6), δbay(7)]
        #
        # δṗ = δv
        # δv̇ = (∂R·a_b/∂θ)·δθ − R·δba     (linearized)
        # δθ̇ = −δbgz
        # δḃ  = 0  (random walk)
        #
        # ∂(ax_w)/∂θ = ∂(c·ax_b − s·ay_b)/∂θ = −s·ax_b − c·ay_b
        # ∂(ay_w)/∂θ = ∂(s·ax_b + c·ay_b)/∂θ =  c·ax_b − s·ay_b
        #
        # −R (for δba columns):  R = [[c,-s],[s,c]]
        #   δv̇x ∝ δbax: −c     δv̇x ∝ δbay: −(−s) = s
        #   δv̇y ∝ δbax: −s     δv̇y ∝ δbay: −c
        Fc = np.zeros((N, N))
        Fc[0, 2] = 1.0                        # δṗx = δvx
        Fc[1, 3] = 1.0                        # δṗy = δvy
        Fc[2, 4] = -s * ax_b - c * ay_b      # δv̇x / δθ
        Fc[3, 4] =  c * ax_b - s * ay_b      # δv̇y / δθ
        Fc[2, 6] = -c;  Fc[2, 7] =  s        # δv̇x / δba
        Fc[3, 6] = -s;  Fc[3, 7] = -c        # δv̇y / δba
        Fc[4, 5] = -1.0                       # δθ̇ = −δbgz

        F = np.eye(N) + Fc * dt

        # ── Discrete process noise Q ──────────────────────────────────────────
        Q = np.zeros((N, N))
        Q[2, 2] = Q[3, 3] = _SIG_ACC  ** 2 * dt   # velocity from accel noise
        Q[4, 4]            = _SIG_GYRO ** 2 * dt   # heading from gyro noise
        Q[5, 5]            = _SIG_BG   ** 2 * dt   # gyro bias random walk
        Q[6, 6] = Q[7, 7] = _SIG_BA   ** 2 * dt   # accel bias random walk

        self._P = F @ self._P @ F.T + Q

    # ──────────────────────────────────────────────────────────────────────────
    # UWB callback: error-state correction + nominal state injection + reset
    # ──────────────────────────────────────────────────────────────────────────
    def _on_uwb(self, msg: Odometry):
        zx = msg.pose.pose.position.x
        zy = msg.pose.pose.position.y

        if not self._ready:
            # Bootstrap nominal position from the first UWB fix
            self._px, self._py = zx, zy
            self._ready = True
            self.get_logger().info(f'ESKF initialised at ({zx:.3f}, {zy:.3f}) m')
            self._publish(msg.header.stamp)
            return

        # Innovation: measurement minus nominal position prediction
        innov = np.array([zx - self._px, zy - self._py])

        # Per-message measurement noise: uwb_to_odom.py inflates this when an
        # anchor in the fix is NLOS-flagged, so trust it over the fixed
        # _R_uwb default.
        R_uwb = np.diag([msg.pose.covariance[0], msg.pose.covariance[7]])

        # Standard KF gain and update
        S = self._H @ self._P @ self._H.T + R_uwb
        K = self._P @ self._H.T @ np.linalg.inv(S)
        dx = K @ innov   # best estimate of error state

        # ── Inject correction into the nominal state ─────────────────────────
        self._px  += dx[0];  self._py  += dx[1]
        self._vx  += dx[2];  self._vy  += dx[3]
        self._th  += dx[4]
        self._bgz += dx[5]
        self._bax += dx[6];  self._bay += dx[7]

        # ── Covariance update (Joseph form — numerically stable) ─────────────
        IKH = np.eye(N) - K @ self._H
        self._P = IKH @ self._P @ IKH.T + K @ R_uwb @ K.T

        # Error state is now reset to zero implicitly (no separate vector
        # needed — the nominal state absorbed it all above).

        self._publish(msg.header.stamp)

    # ──────────────────────────────────────────────────────────────────────────
    def _publish(self, stamp):
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'

        odom.pose.pose.position.x = self._px
        odom.pose.pose.position.y = self._py
        odom.pose.pose.position.z = 0.0

        # Heading → planar quaternion (only z/w non-zero)
        half = self._th / 2.0
        odom.pose.pose.orientation.z = math.sin(half)
        odom.pose.pose.orientation.w = math.cos(half)

        odom.twist.twist.linear.x  = self._vx
        odom.twist.twist.linear.y  = self._vy
        odom.twist.twist.angular.z = self._bgz   # corrected yaw rate bias

        BIG = 9999.0
        p = self._P
        odom.pose.covariance = [
            p[0, 0], p[0, 1], 0., 0., 0., 0.,
            p[1, 0], p[1, 1], 0., 0., 0., 0.,
            0.,      0.,      BIG, 0., 0., 0.,
            0.,      0.,      0., BIG, 0., 0.,
            0.,      0.,      0., 0., BIG, 0.,
            0.,      0.,      0., 0., 0., p[4, 4],
        ]
        odom.twist.covariance = [
            p[2, 2], p[2, 3], 0., 0., 0., 0.,
            p[3, 2], p[3, 3], 0., 0., 0., 0.,
            0.,      0.,      BIG, 0., 0., 0.,
            0.,      0.,      0., BIG, 0., 0.,
            0.,      0.,      0., 0., BIG, 0.,
            0.,      0.,      0., 0., 0., p[4, 4],
        ]
        self._pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = ESKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
