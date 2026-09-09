from __future__ import annotations

from collections import deque
from enum import Enum
import json
import logging
import math
import os
from pathlib import Path

import numpy as np

from sensor_processing import TimestampNormalizer, RobustIMUPreprocessor

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6371000.0


class GNSSState(str, Enum):
    WAITING_FOR_FIX = "WAITING_FOR_FIX"
    GNSS_AIDED = "GNSS_AIDED"
    GNSS_DEGRADED = "GNSS_DEGRADED"
    GNSS_LOST = "GNSS_LOST"
    INS_DEAD_RECKONING = "INS_DEAD_RECKONING"
    GNSS_REACQUISITION = "GNSS_REACQUISITION"
    FUSED = "FUSED"


class TemporalAccelerationModel:
    def __init__(self, model=None, features=None, max_history=20):
        self.model = model
        self.features = features
        self.max_history = max_history
        self.history_linear = deque(maxlen=max_history)
        self.history_gyro = deque(maxlen=max_history)

    def update(self, linear_accel, gyro):
        self.history_linear.append(linear_accel)
        self.history_gyro.append(gyro)

        if self.model is None or len(self.history_linear) < self.max_history:
            return None

        try:
            from train_speed_model_vbox import build_features

            linear_arr = np.asarray(self.history_linear, dtype=float)
            gyro_arr = np.asarray(self.history_gyro, dtype=float)

            features = build_features(linear_arr, gyro_arr)
            prediction = self.model.predict(features)[-1]

            return float(np.clip(prediction, -8.0, 8.0))
        except Exception as exc:
            logger.warning("AI acceleration prediction failed: %s", exc)
            return None


class _Orientation:
    def __init__(self):
        self.forward_phone = [0.0, 1.0, 0.0]
        self.up_phone = [0.0, 0.0, 1.0]
        self.vehicle_calibrated = False

    def set_vehicle_alignment(self, forward_phone, up_phone):
        self.forward_phone = list(forward_phone)
        self.up_phone = list(up_phone)
        self.vehicle_calibrated = True


class NavigationEngine:
    def __init__(
        self,
        map_path=None,
        model_path=None,
        gnss_timeout_s=10.0,
        forward_accel_clip=3.0,
        max_speed_mps=35.0,
        min_move_distance=0.05,
        ai_accel_gate_max=6.0,
        ai_accel_diff_max=2.0,
    ):
        self.map_path = map_path
        self.model_path = model_path
        self.gnss_timeout_s = float(gnss_timeout_s)

        # Drift-control safeguards adapted from Member 1's engine.
        self.forward_accel_clip = float(forward_accel_clip)
        self.max_speed_mps = float(max_speed_mps)
        self.min_move_distance = float(min_move_distance)
        self.ai_accel_gate_max = float(ai_accel_gate_max)
        self.ai_accel_diff_max = float(ai_accel_diff_max)

        self.timestamp_normalizer = TimestampNormalizer()
        self.gnss_timestamp_normalizer = TimestampNormalizer()
        self.imu_preprocessor = RobustIMUPreprocessor()

        self.orientation = _Orientation()
        self.forward_phone = self.orientation.forward_phone
        self.up_phone = self.orientation.up_phone

        self.position = np.zeros(2, dtype=float)  # east, north
        self.velocity = np.zeros(2, dtype=float)

        self.heading_deg = None
        self.speed_mps = 0.0

        self.origin_latitude = None
        self.origin_longitude = None

        self.last_gnss_timestamp = None
        self.last_imu_timestamp = None

        self.gnss_state = GNSSState.WAITING_FOR_FIX

        self.accepted_imu = 0
        self.accepted_gnss = 0
        self.rejected_samples = 0

        self.track = deque(maxlen=400)

        self.map_status = "UNAVAILABLE"
        self.map_error = None
        self.map_matcher = None

        self.accel_model = TemporalAccelerationModel()
        self.ai_model_status = "NOT_CONFIGURED"

        self.nhc_status = "ACTIVE"
        self.uncertainty_m = None

        self._last_gnss_position = None

        self._load_map()
        self._load_model()

    # ---------------------------------------------------------
    # Setup
    # ---------------------------------------------------------

    def _load_map(self):
        if not self.map_path:
            return

        try:
            from map_matching import load_roads_from_geojson, VehicleMapMatcher

            roads = load_roads_from_geojson(self.map_path)

            if roads:
                self.map_matcher = VehicleMapMatcher(roads)
                self.map_status = "READY"
            else:
                self.map_status = "UNAVAILABLE"

        except Exception as exc:
            self.map_error = str(exc)
            self.map_status = "ERROR"

    def _load_model(self):
        if not self.model_path:
            self.ai_model_status = "NOT_CONFIGURED"
            return

        try:
            import joblib

            model = joblib.load(self.model_path)
            self.accel_model.model = model
            self.ai_model_status = "READY"
        except Exception as exc:
            logger.warning("Could not load speed model: %s", exc)
            self.ai_model_status = "UNAVAILABLE"

    # ---------------------------------------------------------
    # Coordinates
    # ---------------------------------------------------------

    def _ll_to_xy(self, latitude, longitude):
        if self.origin_latitude is None:
            self.origin_latitude = float(latitude)
            self.origin_longitude = float(longitude)

        cos_lat = max(
            1e-6,
            math.cos(math.radians(self.origin_latitude)),
        )

        east = (
            math.radians(float(longitude) - self.origin_longitude)
            * EARTH_RADIUS_M
            * cos_lat
        )

        north = (
            math.radians(float(latitude) - self.origin_latitude)
            * EARTH_RADIUS_M
        )

        return np.array([east, north], dtype=float)

    def _xy_to_ll(self, position):
        if self.origin_latitude is None:
            return None

        cos_lat = max(
            1e-6,
            math.cos(math.radians(self.origin_latitude)),
        )

        latitude = (
            self.origin_latitude
            + math.degrees(position[1] / EARTH_RADIUS_M)
        )

        longitude = (
            self.origin_longitude
            + math.degrees(
                position[0] / (EARTH_RADIUS_M * cos_lat)
            )
        )

        return latitude, longitude

    # ---------------------------------------------------------
    # GNSS
    # ---------------------------------------------------------

    def process_gnss(
        self,
        timestamp,
        latitude,
        longitude,
        speed_mps=None,
        accuracy_m=10.0,
        altitude_m=None,
    ):
        try:
            timestamp = self.gnss_timestamp_normalizer.accept(timestamp).timestamp
            latitude = float(latitude)
            longitude = float(longitude)
            accuracy_m = max(0.1, float(accuracy_m))

            if not all(
                math.isfinite(v)
                for v in [timestamp, latitude, longitude, accuracy_m]
            ):
                raise ValueError("GNSS values must be finite")

            if (
                self.last_gnss_timestamp is not None
                and timestamp <= self.last_gnss_timestamp
            ):
                raise ValueError("GNSS timestamp must be strictly increasing")

            new_position = self._ll_to_xy(latitude, longitude)

            previous_position = self._last_gnss_position

            if previous_position is not None:
                delta = new_position - previous_position
                distance = float(np.linalg.norm(delta))

                if distance > 0.1:
                    self.heading_deg = (
                        math.degrees(
                            math.atan2(delta[0], delta[1])
                        )
                        + 360.0
                    ) % 360.0

            self.position = new_position

            if speed_mps is not None and math.isfinite(float(speed_mps)):
                self.speed_mps = max(0.0, float(speed_mps))

            if self.heading_deg is not None:
                heading_rad = math.radians(self.heading_deg)
                self.velocity = np.array(
                    [
                        self.speed_mps * math.sin(heading_rad),
                        self.speed_mps * math.cos(heading_rad),
                    ]
                )

            was_lost = self.gnss_state in {
                GNSSState.GNSS_LOST,
                GNSSState.INS_DEAD_RECKONING,
                GNSSState.GNSS_DEGRADED,
            }

            self.last_gnss_timestamp = timestamp
            self.last_imu_timestamp = (
                self.last_imu_timestamp
                if self.last_imu_timestamp is not None
                else timestamp
            )

            self._last_gnss_position = new_position.copy()
            self.accepted_gnss += 1

            if was_lost:
                self.gnss_state = GNSSState.GNSS_REACQUISITION

                # Conservative correction instead of teleporting.
                self.position = (
                    0.65 * self.position
                    + 0.35 * new_position
                )
            else:
                self.gnss_state = GNSSState.GNSS_AIDED

            self.uncertainty_m = accuracy_m

            ll = self._xy_to_ll(self.position)
            if ll:
                self.track.append(ll)

            return self.state_snapshot()

        except Exception:
            self.rejected_samples += 1
            raise

    # ---------------------------------------------------------
    # IMU
    # ---------------------------------------------------------

    def process_imu(
        self,
        timestamp,
        accel,
        gyro,
        mag=None,
    ):
        try:
            sample = self.timestamp_normalizer.accept(timestamp)
            dt = sample.dt

            processed = self.imu_preprocessor.update(
                accel=accel,
                gyro=gyro,
                mag=mag,
                dt=dt,
            )

            self.accepted_imu += 1
            self.last_imu_timestamp = sample.timestamp

            # Determine GNSS availability.
            if self.last_gnss_timestamp is None:
                self.gnss_state = GNSSState.WAITING_FOR_FIX
            else:
                age = sample.timestamp - self.last_gnss_timestamp

                if age > self.gnss_timeout_s:
                    self.gnss_state = GNSSState.INS_DEAD_RECKONING

            # AI prediction only during DR.
            ai_accel = None

            if self.gnss_state == GNSSState.INS_DEAD_RECKONING:
                ai_accel = self.accel_model.update(
                    processed.filtered_linear_accel_phone.tolist(),
                    processed.raw_gyro.tolist(),
                )

            forward_accel = float(
                processed.filtered_linear_accel_phone[1]
            )

            # Use AI acceleration only when it is physically plausible
            # and sufficiently close to the sensor-derived acceleration.
            if ai_accel is not None:
                if (
                    math.isfinite(ai_accel)
                    and abs(ai_accel) <= self.ai_accel_gate_max
                    and abs(ai_accel - forward_accel)
                    <= self.ai_accel_diff_max
                ):
                    forward_accel = ai_accel

            # Prevent acceleration spikes from causing DR drift.
            forward_accel = float(
                np.clip(
                    forward_accel,
                    -self.forward_accel_clip,
                    self.forward_accel_clip,
                )
            )

            previous_speed = self.speed_mps

            if not processed.is_stationary:
                self.speed_mps = float(
                    np.clip(
                        previous_speed + forward_accel * dt,
                        0.0,
                        self.max_speed_mps,
                    )
                )

            if self.heading_deg is None:
                self.heading_deg = 0.0

            heading_rad = math.radians(self.heading_deg)

            distance = (
                0.5
                * (previous_speed + self.speed_mps)
                * dt
            )

            # Ignore tiny displacement caused by sensor noise.
            if abs(distance) < self.min_move_distance:
                distance = 0.0

            displacement = np.array(
                [
                    distance * math.sin(heading_rad),
                    distance * math.cos(heading_rad),
                ]
            )

            if self.gnss_state == GNSSState.INS_DEAD_RECKONING:
                self.position += displacement

            self.velocity = (
                self.speed_mps
                * np.array(
                    [
                        math.sin(heading_rad),
                        math.cos(heading_rad),
                    ]
                )
            )

            ll = self._xy_to_ll(self.position)

            if ll:
                self.track.append(ll)

            if self.gnss_state == GNSSState.INS_DEAD_RECKONING:
                self.uncertainty_m = (
                    2.0
                    if self.uncertainty_m is None
                    else self.uncertainty_m + 0.05
                )

            return self.state_snapshot()

        except Exception:
            self.rejected_samples += 1
            raise

    # ---------------------------------------------------------
    # State
    # ---------------------------------------------------------

    def state_snapshot(self):
        position = self._xy_to_ll(self.position)

        if self.gnss_state == GNSSState.INS_DEAD_RECKONING:
            mode = "DEAD_RECKONING"
        elif self.gnss_state == GNSSState.GNSS_REACQUISITION:
            mode = "REACQUISITION"
        elif self.gnss_state == GNSSState.GNSS_AIDED:
            mode = "GNSS_AIDED"
        else:
            mode = self.gnss_state.value

        return {
            
            "position": (
                None
                if position is None
                else {
                    "latitude": position[0],
                    "longitude": position[1],
                }
            ),
            "imu_status": "ACTIVE" if self.accepted_imu > 0 else "INACTIVE",
            "local_position_m": {
                "east": float(self.position[0]),
                "north": float(self.position[1]),
            },
            "speed_mps": float(self.speed_mps),
            "speed_kmh": float(self.speed_mps * 3.6),
            "heading_deg": self.heading_deg,
            "gnss_state": self.gnss_state.value,
            "gnss_status": (
                "LOST"
                if self.gnss_state
                in {
                    GNSSState.GNSS_LOST,
                    GNSSState.INS_DEAD_RECKONING,
                }
                else "CONNECTED"
                if self.gnss_state == GNSSState.GNSS_AIDED
                else self.gnss_state.value
            ),
            "mode": mode,
            "uncertainty_m": self.uncertainty_m,
            "nhc_status": self.nhc_status,
            "ai_model_status": self.ai_model_status,
            "map_status": self.map_status,
            "map_error": self.map_error,
            "accepted_imu": self.accepted_imu,
            "accepted_gnss": self.accepted_gnss,
            "rejected_samples": self.rejected_samples,
            "track": list(self.track),
        }

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def reset(self):
        self.timestamp_normalizer = TimestampNormalizer()
        self.gnss_timestamp_normalizer = TimestampNormalizer()
        self.imu_preprocessor = RobustIMUPreprocessor()

        self.position[:] = 0.0
        self.velocity[:] = 0.0

        self.heading_deg = None
        self.speed_mps = 0.0

        self.origin_latitude = None
        self.origin_longitude = None

        self.last_gnss_timestamp = None
        self.last_imu_timestamp = None

        self.gnss_state = GNSSState.WAITING_FOR_FIX

        self.accepted_imu = 0
        self.accepted_gnss = 0
        self.rejected_samples = 0

        self.track.clear()

        self.uncertainty_m = None
        self._last_gnss_position = None

        self.map_status = (
            "READY"
            if self.map_matcher is not None
            else "UNAVAILABLE"
        )