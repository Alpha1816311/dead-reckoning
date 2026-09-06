from collections import deque
import logging
import math

import numpy as np


logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6371000.0


class TemporalAccelerationModel:
    def __init__(self, model, features, max_history=20):
        self.model = model
        self.features = features
        self.max_history = max_history
        self.history_linear = deque(maxlen=max_history)
        self.history_gyro = deque(maxlen=max_history)

    def update(self, linear_accel: list[float], gyro: list[float]):
        self.history_linear.append(linear_accel)
        self.history_gyro.append(gyro)

        if len(self.history_linear) < self.max_history:
            return None

        linear_arr = np.array(self.history_linear, dtype=np.float64)
        gyro_arr = np.array(self.history_gyro, dtype=np.float64)

        try:
            from train_speed_model_vbox import build_features

            feature_vector = build_features(linear_arr, gyro_arr)
            prediction = self.model.predict(feature_vector)[-1]
            return float(np.clip(prediction, -8.0, 8.0))
        except Exception as exc:
            logger.error("Error during AI model prediction: %s", exc)
            return None


class NavigationEngine:
    def __init__(
        self,
        origin_latitude: float,
        origin_longitude: float,
        forward_accel_clip: float = 3.0,
        max_speed_mps: float = 35.0,
        min_move_distance: float = 0.05,
        ai_accel_gate_max: float = 6.0,
        ai_accel_diff_max: float = 2.0,
    ):
        self.origin_latitude = float(origin_latitude)
        self.origin_longitude = float(origin_longitude)

        self.forward_accel_clip = float(forward_accel_clip)
        self.max_speed_mps = float(max_speed_mps)
        self.min_move_distance = float(min_move_distance)
        self.ai_accel_gate_max = float(ai_accel_gate_max)
        self.ai_accel_diff_max = float(ai_accel_diff_max)

        self.position = [0.0, 0.0]  # [east, north], metres
        self.speed_mps = 0.0
        self.bearing_deg = 0.0

        # Important: GNSS and IMU may legally have the same timestamp.
        self.last_gnss_timestamp = None
        self.last_imu_timestamp = None

        self.track = deque(maxlen=400)
        self.map_status = "UNKNOWN"
        self.map_matcher = None

    def _xy_to_ll(self, position: list[float]) -> tuple[float, float]:
        cos_lat = max(1e-6, math.cos(math.radians(self.origin_latitude)))

        lat = self.origin_latitude + math.degrees(
            float(position[1]) / EARTH_RADIUS_M
        )
        lon = self.origin_longitude + math.degrees(
            float(position[0]) / (EARTH_RADIUS_M * cos_lat)
        )
        return lat, lon

    def _bearing_from_delta(
        self,
        dx: float,
        dy: float,
        min_distance: float = 0.1,
    ) -> float:
        if math.hypot(dx, dy) < min_distance:
            return self.bearing_deg

        bearing = math.degrees(math.atan2(dx, dy))
        return (bearing + 360.0) % 360.0

    def process_gnss(
        self,
        timestamp: float,
        lat: float,
        lon: float,
        accuracy_m: float,
    ):
        """Apply a GNSS position update without rejecting same-time IMU data."""
        timestamp = float(timestamp)

        if not math.isfinite(timestamp):
            logger.warning("Invalid GNSS timestamp. Skipping GNSS frame.")
            return

        if (
            self.last_gnss_timestamp is not None
            and timestamp <= self.last_gnss_timestamp
        ):
            logger.warning("Ignoring out-of-order or duplicate GNSS timestamp.")
            return

        lat = float(lat)
        lon = float(lon)
        accuracy_m = float(accuracy_m)

        if not (math.isfinite(lat) and math.isfinite(lon)):
            logger.warning("Invalid GNSS coordinates. Skipping GNSS frame.")
            return

        self.last_gnss_timestamp = timestamp
        accuracy_m = max(0.1, accuracy_m)

        cos_lat = max(1e-6, math.cos(math.radians(self.origin_latitude)))
        north_m = math.radians(lat - self.origin_latitude) * EARTH_RADIUS_M
        east_m = (
            math.radians(lon - self.origin_longitude)
            * EARTH_RADIUS_M
            * cos_lat
        )

        previous_position = self.position.copy()
        self.position = [east_m, north_m]

        dx = east_m - previous_position[0]
        dy = north_m - previous_position[1]
        self.bearing_deg = self._bearing_from_delta(dx, dy)

        self.track.append(self._xy_to_ll(self.position))

    def _select_forward_accel(
        self,
        linear_accel: list[float],
        ai_acceleration: float | None,
        outage_mode: bool,
    ) -> float:
        accel = np.asarray(linear_accel, dtype=float)
        if accel.shape != (3,) or not np.all(np.isfinite(accel)):
            raise ValueError("linear_accel must contain three finite values")

        speed_accel = float(accel[0])

        if outage_mode and ai_acceleration is not None:
            ai_acceleration = float(ai_acceleration)

            if (
                math.isfinite(ai_acceleration)
                and abs(ai_acceleration) <= self.ai_accel_gate_max
                and abs(ai_acceleration - speed_accel)
                <= self.ai_accel_diff_max
            ):
                speed_accel = ai_acceleration

        return float(
            np.clip(
                speed_accel,
                -self.forward_accel_clip,
                self.forward_accel_clip,
            )
        )

    def process_imu(
        self,
        timestamp: float,
        dt: float,
        linear_accel: list[float],
        ai_acceleration: float | None,
        outage_mode: bool,
    ):
        """Integrate IMU-derived forward acceleration for dead reckoning."""
        timestamp = float(timestamp)
        dt = float(dt)

        if not math.isfinite(timestamp):
            logger.warning("Invalid IMU timestamp. Skipping IMU frame.")
            return

        if not math.isfinite(dt) or dt <= 0.0:
            logger.warning("Invalid dt <= 0 in IMU processing. Skipping frame.")
            return

        if (
            self.last_imu_timestamp is not None
            and timestamp <= self.last_imu_timestamp
        ):
            logger.warning("Out-of-order IMU timestamp encountered. Skipping.")
            return

        self.last_imu_timestamp = timestamp

        speed_accel = self._select_forward_accel(
            linear_accel=linear_accel,
            ai_acceleration=ai_acceleration,
            outage_mode=outage_mode,
        )

        previous_speed = self.speed_mps
        self.speed_mps = float(
            np.clip(
                previous_speed + speed_accel * dt,
                0.0,
                self.max_speed_mps,
            )
        )

        distance_m = 0.5 * (previous_speed + self.speed_mps) * dt

        if abs(distance_m) < self.min_move_distance:
            distance_m = 0.0

        bearing_rad = math.radians(self.bearing_deg)
        self.position[0] += distance_m * math.sin(bearing_rad)
        self.position[1] += distance_m * math.cos(bearing_rad)

        ll = self._xy_to_ll(self.position)
        self.track.append(ll)

        if outage_mode and self.map_matcher:
            self._apply_map_matching(ll)

    def _apply_map_matching(self, ll: tuple[float, float]):
        try:
            matched = self.map_matcher.match(
                state={
                    "lat": ll[0],
                    "lon": ll[1],
                    "bearing": self.bearing_deg,
                }
            )
            self.map_status = "MATCHED" if matched else "UNMATCHED"
        except Exception as exc:
            logger.error("Map matching module failure: %s", exc, exc_info=True)
            self.map_status = "ERROR"

    def _build_output(self) -> dict:
        latitude, longitude = self._xy_to_ll(self.position)

        return {
            "latitude": latitude,
            "longitude": longitude,
            "position": {"latitude": latitude, "longitude": longitude},
            "local_position_m": {
                "east": self.position[0],
                "north": self.position[1],
            },
            "speed_mps": self.speed_mps,
            "bearing_deg": self.bearing_deg,
            "map_status": self.map_status,
            "track": list(self.track),
        }

    def state_snapshot(self) -> dict:
        return self._build_output()