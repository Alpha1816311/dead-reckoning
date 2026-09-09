"""Small, platform-independent sensor preprocessing primitives for IDR.

The live engine receives measurements from Android, an external IMU, or a
dataset replay.  This module deliberately has no Android/FastAPI dependency.
All timestamps are expected to be monotonic seconds from the same clock.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


GRAVITY_MPS2 = 9.80665


@dataclass(frozen=True)
class TimestampedSample:
    timestamp: float
    dt: float


class TimestampNormalizer:
    """Validate timestamps and derive a bounded integration interval."""

    def __init__(self, default_dt: float = 0.01, max_dt: float = 1.0):
        self.default_dt = float(default_dt)
        self.max_dt = float(max_dt)
        self.last_timestamp: float | None = None

    @staticmethod
    def _seconds(value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("timestamp must be finite")

        magnitude = abs(value)
        if magnitude > 1e14:
            value /= 1e9
        elif magnitude > 1e11:
            value /= 1e3
        return value

    def accept(self, timestamp: float) -> TimestampedSample:
        timestamp = self._seconds(timestamp)
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            return TimestampedSample(timestamp, self.default_dt)

        raw_dt = timestamp - self.last_timestamp
        if not math.isfinite(raw_dt) or raw_dt <= 0:
            raise ValueError("timestamp must be strictly increasing")

        self.last_timestamp = timestamp
        return TimestampedSample(
            timestamp=timestamp,
            dt=float(np.clip(raw_dt, 1e-4, self.max_dt)),
        )


def _vector(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    return vector


@dataclass(frozen=True)
class ProcessedIMU:
    raw_accel: np.ndarray
    raw_gyro: np.ndarray
    raw_mag: np.ndarray | None
    gravity_phone: np.ndarray
    linear_accel_phone: np.ndarray
    filtered_linear_accel_phone: np.ndarray
    vibration_rms: float
    is_stationary: bool


class RobustIMUPreprocessor:
    """Remove gravity, compensate for sensor bias, and suppress dynamic noise."""

    def __init__(
        self,
        gravity_time_constant: float = 2.0,
        signal_cutoff_hz: float = 8.0,
        max_linear_accel: float = 35.0,
        stationary_accel_threshold: float = 0.15,
        stationary_gyro_threshold: float = 0.05,
        accel_deadband: float = 0.05,
    ):
        self.gravity_time_constant = float(gravity_time_constant)
        self.signal_cutoff_hz = float(signal_cutoff_hz)
        self.max_linear_accel = float(max_linear_accel)
        self.stationary_accel_thresh = float(stationary_accel_threshold)
        self.stationary_gyro_thresh = float(stationary_gyro_threshold)
        self.accel_deadband = float(accel_deadband)

        self.gravity_phone: np.ndarray | None = None
        self.accel_bias: np.ndarray = np.zeros(3, dtype=float)
        self.gyro_bias: np.ndarray = np.zeros(3, dtype=float)
        self.filtered_linear: np.ndarray | None = None
        self.previous_filtered: np.ndarray | None = None

    def update(self, accel, gyro, mag=None, dt: float = 0.01) -> ProcessedIMU:
        accel = _vector(accel, "accelerometer")
        gyro = _vector(gyro, "gyroscope")
        if mag is not None:
            mag = _vector(mag, "magnetometer")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")

        # 1. Stationary Detection (Zero Velocity Update trigger)
        corrected_gyro = gyro - self.gyro_bias
        accel_mag = np.linalg.norm(accel)
        gyro_mag = np.linalg.norm(corrected_gyro)
        is_stationary = (
            abs(accel_mag - GRAVITY_MPS2) < self.stationary_accel_thresh
            and gyro_mag < self.stationary_gyro_thresh
        )

        # 2. Dynamic Bias Auto-Calibration
        if is_stationary:
            # Calibrate zero-rate gyro offset
            gyro_bias_alpha = 1.0 - math.exp(-dt / 3.0)
            self.gyro_bias += gyro_bias_alpha * (gyro - self.gyro_bias)

            # Calibrate accelerometer zero offset using time-scale alpha
            accel_bias_alpha = 1.0 - math.exp(-dt / 2.0)
            raw_linear = accel - (self.gravity_phone if self.gravity_phone is not None else accel)
            self.accel_bias += accel_bias_alpha * (raw_linear - self.accel_bias)

        # 3. Exact Gyro-Gravity Kinematic Propagation (Rodrigues Formula)
        if self.gravity_phone is None:
            self.gravity_phone = (accel / (accel_mag + 1e-8)) * GRAVITY_MPS2
        else:
            # Rodrigues rotation update for angular velocity vector over dt
            omega_norm = np.linalg.norm(corrected_gyro)
            if omega_norm > 1e-6:
                axis = corrected_gyro / omega_norm
                angle = -omega_norm * dt
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                # Rotate gravity vector around body gyro axis
                self.gravity_phone = (
                    self.gravity_phone * cos_a
                    + np.cross(axis, self.gravity_phone) * sin_a
                    + axis * np.dot(axis, self.gravity_phone) * (1.0 - cos_a)
                )

            # Complementary pull back to accelerometer orientation reference
            tau = self.gravity_time_constant * 0.1 if is_stationary else self.gravity_time_constant * 3.0
            gravity_alpha = 1.0 - math.exp(-dt / max(tau, 1e-3))
            self.gravity_phone += gravity_alpha * (accel - self.gravity_phone)

            # Normalization lock to exact gravity constant
            current_g_norm = np.linalg.norm(self.gravity_phone)
            if current_g_norm > 1e-4:
                self.gravity_phone = (self.gravity_phone / current_g_norm) * GRAVITY_MPS2

        # 4. Linear Acceleration Extraction & Adaptive Dead-Banding
        if is_stationary:
            # ZUPT hard-clamp: prevent integration drift during stationary state
            linear = np.zeros(3, dtype=float)
        else:
            linear = accel - self.gravity_phone - self.accel_bias
            # Suppress values below noise floor
            linear[np.abs(linear) < self.accel_deadband] = 0.0

        clipped = np.clip(linear, -self.max_linear_accel, self.max_linear_accel)

        # 5. Low-Pass Filter Linear Acceleration
        signal_alpha = 1.0 - math.exp(-2.0 * math.pi * self.signal_cutoff_hz * dt)
        if self.filtered_linear is None:
            initial_filtered = clipped.copy()
            initial_filtered[np.abs(initial_filtered) < 1e-8] = 0.0
            self.filtered_linear = initial_filtered
        else:
            if is_stationary:
                # Decay lingering filter memory when stopped
                self.filtered_linear *= math.exp(-dt / 0.05)
            else:
                self.filtered_linear += signal_alpha * (clipped - self.filtered_linear)

        self.filtered_linear[np.abs(self.filtered_linear) < 1e-8] = 0.0

        # 6. Vibration Calculation
        if self.previous_filtered is None:
            vibration_rms = 0.0
        else:
            residual = clipped - self.filtered_linear
            vibration_rms = float(np.sqrt(np.mean(residual * residual)))
        self.previous_filtered = clipped.copy()

        return ProcessedIMU(
            raw_accel=accel.copy(),
            raw_gyro=corrected_gyro,
            raw_mag=None if mag is None else mag.copy(),
            gravity_phone=self.gravity_phone.copy(),
            linear_accel_phone=linear,
            filtered_linear_accel_phone=self.filtered_linear.copy(),
            vibration_rms=vibration_rms,
            is_stationary=is_stationary,
        )