from collections import deque
import logging
import math
import numpy as np

# Setup logger for map-matching and pipeline warnings
logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6371000.0


class TemporalAccelerationModel:
    def __init__(self, model, features, max_history=20):
        self.model = model
        self.features = features
        self.max_history = max_history

        # Use deques with maxlen for O(1) circular ring buffer behavior
        self.history_linear = deque(maxlen=max_history)
        self.history_gyro = deque(maxlen=max_history)

    def update(self, linear_accel: list[float], gyro: list[float]):
        """
        Updates sensor history and runs model prediction when enough samples exist.
        Runs efficiently without per-frame imports or DataFrame allocations.
        """
        self.history_linear.append(linear_accel)
        self.history_gyro.append(gyro)

        if len(self.history_linear) < self.max_history:
            return None

        # Convert ring buffers to arrays directly
        linear_arr = np.array(self.history_linear, dtype=np.float64)
        gyro_arr = np.array(self.history_gyro, dtype=np.float64)

        try:
            # Build features directly using NumPy / model interface
            # Note: Ensure build_features returns a numpy array or dict suitable for predict
            from train_speed_model_vbox import build_features  # Explicit dynamic import fallback
            feature_vector = build_features(linear_arr, gyro_arr)
            
            # Predict single sample directly
            prediction = self.model.predict(feature_vector)[-1]
            return float(np.clip(prediction, -8.0, 8.0))
        except Exception as e:
            logger.error(f"Error during AI model prediction: {e}")
            return None


class NavigationEngine:
    def __init__(self, origin_latitude: float, origin_longitude: float):
        self.origin_latitude = origin_latitude
        self.origin_longitude = origin_longitude

        self.position = [0.0, 0.0]  # [x, y] in meters
        self.speed_mps = 0.0
        self.bearing_deg = 0.0
        self.last_processed_timestamp = None

        self.track = deque(maxlen=400)
        self.map_status = "UNKNOWN"
        self.map_matcher = None  # To be set externally if needed

    def _xy_to_ll(self, position: list[float]) -> tuple[float, float]:
        """
        Converts local x, y coordinates (meters) to latitude and longitude (degrees).
        Clamps cosine term to prevent zero-division near the poles.
        """
        # Clamp cosine term to prevent zero division
        cos_lat = max(1e-6, math.cos(math.radians(self.origin_latitude)))

        lat = self.origin_latitude + math.degrees(position[1] / EARTH_RADIUS_M)
        lon = self.origin_longitude + math.degrees(position[0] / (EARTH_RADIUS_M * cos_lat))

        return lat, lon

    def _bearing_from_delta(self, dx: float, dy: float, min_distance: float = 0.1) -> float:
        """
        Calculates heading/bearing from displacement vector.
        Ignores noise when movement is under the min_distance threshold.
        """
        dist = math.hypot(dx, dy)
        if dist < min_distance:
            return self.bearing_deg  # Retain previous bearing if stationary
        
        bearing = math.degrees(math.atan2(dx, dy))
        return (bearing + 360.0) % 360.0

    def process_gnss(self, timestamp: float, lat: float, lon: float, accuracy_m: float):
        """Processes incoming GNSS measurements."""
        if self.last_processed_timestamp is not None and timestamp <= self.last_processed_timestamp:
            logger.warning("Ignoring out-of-order or duplicate GNSS timestamp.")
            return

        # Ensure accuracy values are non-zero
        accuracy_m = max(0.1, accuracy_m)
        self.last_processed_timestamp = timestamp

        # Convert LL to local XY relative to origin
        cos_lat = max(1e-6, math.cos(math.radians(self.origin_latitude)))
        dy = math.radians(lat - self.origin_latitude) * EARTH_RADIUS_M
        dx = math.radians(lon - self.origin_longitude) * EARTH_RADIUS_M * cos_lat

        self.position = [dx, dy]
        self.track.append(self._xy_to_ll(self.position))

    def process_imu(self, timestamp: float, dt: float, linear_accel: list[float], ai_acceleration: float | None, outage_mode: bool):
        """Processes IMU measurements and dead-reckons position during GNSS outages."""
        if dt <= 0.0:
            logger.warning("Invalid dt <= 0 in IMU processing. Skipping frame.")
            return

        if self.last_processed_timestamp is not None and timestamp <= self.last_processed_timestamp:
            logger.warning("Out-of-order IMU timestamp encountered. Skipping.")
            return

        self.last_processed_timestamp = timestamp

        # Determine effective forward acceleration
        speed_accel = linear_accel[0]  # Default to forward body accelerometer
        if outage_mode and ai_acceleration is not None:
            speed_accel = ai_acceleration

        prev_speed = self.speed_mps
        # Integrate forward velocity
        self.speed_mps = max(0.0, prev_speed + speed_accel * dt)

        # Trapezoidal distance integration
        avg_speed = 0.5 * (prev_speed + self.speed_mps)
        distance = avg_speed * dt

        # Update 2D position based on current bearing
        bearing_rad = math.radians(self.bearing_deg)
        dx = distance * math.sin(bearing_rad)
        dy = distance * math.cos(bearing_rad)

        self.position[0] += dx
        self.position[1] += dy

        ll = self._xy_to_ll(self.position)
        self.track.append(ll)

        if outage_mode and self.map_matcher:
            self._apply_map_matching(ll)

    def _apply_map_matching(self, ll: tuple[float, float]):
        """Executes map matching with proper error handling."""
        try:
            matched = self.map_matcher.match(state={"lat": ll[0], "lon": ll[1], "bearing": self.bearing_deg})
            if matched:
                self.map_status = "MATCHED"
            else:
                self.map_status = "UNMATCHED"
        except Exception as e:
            logger.error(f"Map matching module failure: {e}", exc_info=True)
            self.map_status = "ERROR"

    def _build_output(self) -> dict:
        """Constructs output dictionary for navigation state."""
        current_ll = self._xy_to_ll(self.position)
        return {
            "latitude": current_ll[0],
            "longitude": current_ll[1],
            "speed_mps": self.speed_mps,
            "bearing_deg": self.bearing_deg,
            "map_status": self.map_status,
            "track": list(self.track),
        }