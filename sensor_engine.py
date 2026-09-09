from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

GRAVITY_MPS2 = 9.80665

@dataclass(frozen=True)
class ProcessedIMU:
    raw_accel: np.ndarray
    raw_gyro: np.ndarray
    calibrated_gyro: np.ndarray
    gravity_phone: np.ndarray
    linear_accel_phone: np.ndarray
    filtered_linear_accel_phone: np.ndarray
    is_stationary: bool

class RobustIMUPreprocessor:
    def __init__(
        self,
        gravity_time_constant: float = 2.0,
        signal_cutoff_hz: float = 15.0,
        accel_deadband: float = 0.001,
    ):
        self.signal_cutoff_hz = float(signal_cutoff_hz)
        self.accel_deadband = float(accel_deadband)

        self.gravity_phone: np.ndarray = np.array([0.0, 0.0, GRAVITY_MPS2], dtype=float)
        self.accel_bias = np.zeros(3, dtype=float)
        self.gyro_bias = np.zeros(3, dtype=float)
        self.filtered_linear: np.ndarray | None = None

    def update(self, accel: list[float], gyro: list[float], dt: float = 0.01) -> ProcessedIMU:
        accel_arr = np.asarray(accel, dtype=float)
        gyro_arr = np.asarray(gyro, dtype=float)

        calibrated_gyro = gyro_arr - self.gyro_bias
        linear = accel_arr - self.gravity_phone - self.accel_bias

        signal_alpha = 1.0 - math.exp(-2.0 * math.pi * self.signal_cutoff_hz * dt)
        if self.filtered_linear is None:
            self.filtered_linear = linear.copy()
        else:
            self.filtered_linear += signal_alpha * (linear - self.filtered_linear)

        mask = np.abs(self.filtered_linear) < self.accel_deadband
        self.filtered_linear[mask] = 0.0

        # Only set stationary when initial standing setup is explicit
        is_stationary = False

        return ProcessedIMU(
            raw_accel=accel_arr,
            raw_gyro=gyro_arr,
            calibrated_gyro=calibrated_gyro,
            gravity_phone=self.gravity_phone.copy(),
            linear_accel_phone=linear,
            filtered_linear_accel_phone=self.filtered_linear.copy(),
            is_stationary=is_stationary,
        )
