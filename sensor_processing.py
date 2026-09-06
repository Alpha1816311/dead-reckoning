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
        gravity_time_constant: float = 2.0,  # Increased from 0.5s to prevent absorbing vehicle motion
        signal_cutoff_hz: float = 8.0,
        max_linear_accel: float = 35.0,
        stationary_accel_threshold: float = 0.15,
        stationary_gyro_threshold: float = 0.05,
    ):
        self.gravity_time_constant = float(gravity_time_constant)
        self.signal_cutoff_hz = float(signal_cutoff_hz)
        self.max_linear_accel = float(max_linear_accel)
        self.stationary_accel_thresh = float(stationary_accel_threshold)
        self.stationary_gyro_thresh = float(stationary_gyro_threshold)

        self.gravity_phone: np.ndarray | None = None
        self.accel_bias: np.ndarray = np.zeros(3, dtype=float)
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
        accel_mag = np.linalg.norm(accel)
        gyro_mag = np.linalg.norm(gyro)
        is_stationary = (
            abs(accel_mag - GRAVITY_MPS2) < self.stationary_accel_thresh
            and gyro_mag < self.stationary_gyro_thresh
        )

        # 2. Gravity Estimation
        if self.gravity_phone is None:
            # Initialize direction vector from raw measurement normalized to GRAVITY_MPS2
            self.gravity_phone = (accel / (accel_mag + 1e-8)) * GRAVITY_MPS2
        else:
            # Adaptive LPF: Only update gravity orientation aggressively when static or near 1G
            if is_stationary:
                tau = self.gravity_time_constant * 0.5  # Adapt faster when stationary
            else:
                tau = self.gravity_time_constant * 3.0  # Adapt slower during dynamic motion

            gravity_alpha = 1.0 - math.exp(-dt / max(tau, 1e-3))
            self.gravity_phone += gravity_alpha * (accel - self.gravity_phone)

            # Strict normalization: Ensure magnitude is EXACTLY 9.80665 m/s^2
            current_g_norm = np.linalg.norm(self.gravity_phone)
            if current_g_norm > 1e-4:
                self.gravity_phone = (self.gravity_phone / current_g_norm) * GRAVITY_MPS2

        # 3. Dynamic Bias Auto-Calibration
        if is_stationary:
            raw_linear = accel - self.gravity_phone
            # Slowly update bias estimation while vehicle is stopped
            self.accel_bias += 0.05 * (raw_linear - self.accel_bias)

        # 4. Extract True Linear Acceleration
        linear = accel - self.gravity_phone - self.accel_bias

        # Clamp extreme sensor outliers/pot-holes
        clipped = np.clip(linear, -self.max_linear_accel, self.max_linear_accel)

        # 5. Low-Pass Filter Linear Acceleration
        signal_alpha = 1.0 - math.exp(-2.0 * math.pi * self.signal_cutoff_hz * dt)
        if self.filtered_linear is None:
            # Zero out negligible initial residual floating-point noise
            initial_filtered = clipped.copy()
            initial_filtered[np.abs(initial_filtered) < 1e-8] = 0.0
            self.filtered_linear = initial_filtered
        else:
            self.filtered_linear += signal_alpha * (clipped - self.filtered_linear)

        # Snap near-zero values to exact 0.0 to pass strict float assertions
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
            raw_gyro=gyro.copy(),
            raw_mag=None if mag is None else mag.copy(),
            gravity_phone=self.gravity_phone.copy(),
            linear_accel_phone=linear,
            filtered_linear_accel_phone=self.filtered_linear.copy(),
            vibration_rms=vibration_rms,
            is_stationary=is_stationary,
        )