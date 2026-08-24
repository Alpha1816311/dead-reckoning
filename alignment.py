"""
orientation.py

Phone IMU orientation estimation and phone-frame -> vehicle-frame transform.

Coordinate conventions
----------------------
Phone/body frame:
    Right-handed, arbitrary according to the phone sensor axes.

Navigation frame:
    X = magnetic/east-referenced horizontal direction
    Y = horizontal direction
    Z = up

Vehicle frame:
    X = vehicle forward
    Y = vehicle left
    Z = vehicle up

Inputs:
    accelerometer : m/s^2
    gyroscope     : rad/s
    magnetometer  : arbitrary units (only direction is used)
    dt            : seconds

Quaternion convention:
    q = [w, x, y, z]

q_phone_to_nav rotates vectors from phone frame into navigation frame.
"""

from __future__ import annotations

import numpy as np


_EPS = 1e-12


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b

    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[1:]


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_norm = np.linalg.norm(axis)

    if axis_norm < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = axis / axis_norm
    half = 0.5 * angle

    return np.array([
        np.cos(half),
        *(axis * np.sin(half)),
    ])


def _integrate_gyro(q: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
    """
    Integrate angular velocity using an exponential-map update.
    gyro is rad/s in phone coordinates.
    """
    angle = np.linalg.norm(gyro) * dt

    if angle < 1e-10:
        dq = np.array([
            1.0,
            0.5 * gyro[0] * dt,
            0.5 * gyro[1] * dt,
            0.5 * gyro[2] * dt,
        ])
    else:
        dq = _quat_from_axis_angle(gyro, angle)

    return _quat_normalize(_quat_mul(q, dq))


# ---------------------------------------------------------------------------
# Vector utilities
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray | None:
    n = np.linalg.norm(v)

    if n < _EPS:
        return None

    return v / n


def _initial_orientation(accel: np.ndarray, mag: np.ndarray) -> np.ndarray | None:
    """
    Construct an initial phone->navigation quaternion.

    Navigation:
        Z = up
        X = magnetic north projection
        Y = completes right-handed frame

    Accelerometer is assumed to measure +g in the upward direction while
    stationary. If the particular platform reports -g, negate accel before
    calling this function.
    """
    up_b = _normalize(accel)
    mag_b = _normalize(mag)

    if up_b is None or mag_b is None:
        return None

    # Project magnetic field onto horizontal plane.
    north_b = mag_b - np.dot(mag_b, up_b) * up_b
    north_b = _normalize(north_b)

    if north_b is None:
        return None

    # Right-handed horizontal axis.
    east_b = _normalize(np.cross(up_b, north_b))

    if east_b is None:
        return None

    # Re-orthogonalize north.
    north_b = np.cross(east_b, up_b)

    # Columns describe navigation axes expressed in phone coordinates.
    #
    # We want:
    #   phone vector -> navigation vector
    #
    # The matrix whose rows are nav axes in body coordinates does that.
    R = np.vstack([
        north_b,
        east_b,
        up_b,
    ])

    return _rotation_matrix_to_quaternion(R)


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to [w,x,y,z]."""
    trace = np.trace(R)

    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return _quat_normalize(np.array([w, x, y, z]))


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

class PhoneOrientation:
    """
    Estimate phone orientation and transform IMU data into vehicle coordinates.

    Parameters
    ----------
    gyro_gain:
        Weight given to gyroscope integration. Higher values mean slower
        correction from accelerometer/magnetometer.

    accel_gain:
        Accelerometer attitude correction strength.

    mag_gain:
        Magnetometer heading correction strength.

    Example
    -------
    estimator = PhoneOrientation()

    # Set once after mounting the phone:
    estimator.set_vehicle_alignment(
        forward_phone=[0.0, 1.0, 0.0],
        up_phone=[0.0, 0.0, 1.0],
    )

    result = estimator.update(
        accel=[...],
        gyro=[...],
        mag=[...],
        dt=0.01,
    )

    vehicle_accel = result["accel"]
    vehicle_gyro = result["gyro"]
    """

    def __init__(
        self,
        gyro_gain: float = 0.98,
        accel_gain: float = 0.05,
        mag_gain: float = 0.02,
    ):
        self.gyro_gain = gyro_gain
        self.accel_gain = accel_gain
        self.mag_gain = mag_gain

        # Phone -> navigation.
        self.q_phone_to_nav = np.array([1.0, 0.0, 0.0, 0.0])

        # Phone -> vehicle.
        #
        # This is intentionally independent of magnetic heading. It describes
        # how the phone is physically mounted in the vehicle.
        self.q_phone_to_vehicle = np.array([1.0, 0.0, 0.0, 0.0])

        self.initialized = False
        self.vehicle_calibrated = False

    # ------------------------------------------------------------------
    # Vehicle mounting calibration
    # ------------------------------------------------------------------

    def set_vehicle_alignment(
        self,
        forward_phone,
        up_phone,
    ):
        """
        Configure the physical phone mounting orientation.

        Parameters
        ----------
        forward_phone:
            Vehicle-forward unit vector expressed in phone coordinates.

        up_phone:
            Vehicle-up unit vector expressed in phone coordinates.

        Example:
            If the phone is mounted flat with its +Y pointing forward
            and +Z pointing upward:

                forward_phone = [0, 1, 0]
                up_phone      = [0, 0, 1]

        The two vectors must be approximately perpendicular.
        """
        forward = _normalize(np.asarray(forward_phone, dtype=float))
        up = _normalize(np.asarray(up_phone, dtype=float))

        if forward is None or up is None:
            raise ValueError("Invalid vehicle alignment vectors.")

        # Remove any non-orthogonality.
        forward = forward - np.dot(forward, up) * up
        forward = _normalize(forward)

        if forward is None:
            raise ValueError("forward_phone and up_phone must not be parallel.")

        # Vehicle Y = left.
        left = _normalize(np.cross(up, forward))

        if left is None:
            raise ValueError("Invalid vehicle alignment.")

        # Re-orthogonalize forward.
        forward = np.cross(left, up)

        # Rotation matrix: phone -> vehicle.
        #
        # Each row is a vehicle axis expressed in phone coordinates.
        R = np.vstack([
            forward,
            left,
            up,
        ])

        self.q_phone_to_vehicle = _rotation_matrix_to_quaternion(R)
        self.vehicle_calibrated = True

    # ------------------------------------------------------------------
    # Orientation update
    # ------------------------------------------------------------------

    def update(
        self,
        accel,
        gyro,
        mag,
        dt: float,
    ) -> dict:
        """
        Process one IMU sample.

        Parameters
        ----------
        accel : array-like, shape (3,)
            Accelerometer in m/s^2.

        gyro : array-like, shape (3,)
            Angular velocity in rad/s.

        mag : array-like, shape (3,)
            Magnetometer vector. Magnitude is irrelevant.

        dt : float
            Time since previous sample, seconds.

        Returns
        -------
        dict containing:

            orientation:
                phone -> navigation quaternion [w,x,y,z]

            accel:
                acceleration rotated into vehicle frame

            gyro:
                angular velocity rotated into vehicle frame

            mag:
                magnetic field rotated into vehicle frame

            initialized:
                whether the attitude estimator has initialized

            vehicle_calibrated:
                whether a phone->vehicle mounting transform exists
        """
        accel = np.asarray(accel, dtype=float)
        gyro = np.asarray(gyro, dtype=float)
        mag = np.asarray(mag, dtype=float)

        if dt <= 0:
            raise ValueError("dt must be positive.")

        # --------------------------------------------------------------
        # Initialization from gravity + magnetic field
        # --------------------------------------------------------------

        if not self.initialized:
            q0 = _initial_orientation(accel, mag)

            if q0 is not None:
                self.q_phone_to_nav = q0
                self.initialized = True

        # --------------------------------------------------------------
        # Gyroscope propagation
        # --------------------------------------------------------------

        if self.initialized:
            self.q_phone_to_nav = _integrate_gyro(
                self.q_phone_to_nav,
                gyro,
                dt,
            )

            # ----------------------------------------------------------
            # Accelerometer correction
            # ----------------------------------------------------------

            a = _normalize(accel)

            if a is not None:
                # Expected +Z/up direction expressed in phone coordinates.
                up_est = _quat_rotate(
                    _quat_conj(self.q_phone_to_nav),
                    np.array([0.0, 0.0, 1.0]),
                )

                error_acc = np.cross(a, up_est)

                correction = self.accel_gain * error_acc

                self.q_phone_to_nav = _integrate_gyro(
                    self.q_phone_to_nav,
                    correction,
                    dt,
                )

            # ----------------------------------------------------------
            # Magnetometer correction
            # ----------------------------------------------------------

            m = _normalize(mag)

            if m is not None:
                # Magnetic north estimate in navigation coordinates.
                #
                # Rather than assuming a particular local magnetic
                # inclination, derive the horizontal direction from the
                # current measurement.
                m_nav = _quat_rotate(self.q_phone_to_nav, m)

                horizontal = np.array([
                    m_nav[0],
                    m_nav[1],
                    0.0,
                ])

                north_nav = _normalize(horizontal)

                if north_nav is not None:
                    north_est_phone = _quat_rotate(
                        _quat_conj(self.q_phone_to_nav),
                        north_nav,
                    )

                    error_mag = np.cross(m, north_est_phone)

                    correction = self.mag_gain * error_mag

                    self.q_phone_to_nav = _integrate_gyro(
                        self.q_phone_to_nav,
                        correction,
                        dt,
                    )

        # --------------------------------------------------------------
        # Phone -> vehicle transformation
        # --------------------------------------------------------------

        accel_vehicle = _quat_rotate(
            self.q_phone_to_vehicle,
            accel,
        )

        gyro_vehicle = _quat_rotate(
            self.q_phone_to_vehicle,
            gyro,
        )

        mag_vehicle = _quat_rotate(
            self.q_phone_to_vehicle,
            mag,
        )

        return {
            "orientation": self.q_phone_to_nav.copy(),
            "accel": accel_vehicle,
            "gyro": gyro_vehicle,
            "mag": mag_vehicle,
            "initialized": self.initialized,
            "vehicle_calibrated": self.vehicle_calibrated,
        }


# ---------------------------------------------------------------------------
# Convenience function for stateless frame transformation
# ---------------------------------------------------------------------------

def phone_to_vehicle(vector, forward_phone, up_phone):
    """
    Rotate a single phone-frame vector into vehicle coordinates.

    This is useful if the orientation estimator is not otherwise needed.
    """
    forward = _normalize(np.asarray(forward_phone, dtype=float))
    up = _normalize(np.asarray(up_phone, dtype=float))

    if forward is None or up is None:
        raise ValueError("Invalid alignment vectors.")

    forward = forward - np.dot(forward, up) * up
    forward = _normalize(forward)

    if forward is None:
        raise ValueError("forward_phone and up_phone must not be parallel.")

    left = _normalize(np.cross(up, forward))

    R = np.vstack([
        forward,
        left,
        up,
    ])

    return R @ np.asarray(vector, dtype=float)