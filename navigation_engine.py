from __future__ import annotations

from collections import deque
from enum import Enum
import logging
import math
import os
import time

import numpy as np

from alignment import AlignmentState, PhoneOrientation, VehicleAlignmentCalibrator
from sensor_processing import MotionState, TimestampNormalizer, RobustIMUPreprocessor
from speed_model import SpeedModel

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6371000.0
SPEED_FEATURES = [
    "linear_accel_x",
    "linear_accel_y",
    "linear_accel_z",
    "accel_magnitude",
    "accel_magnitude_smooth",
    "gyro_magnitude",
]


class GNSSState(str, Enum):
    WAITING_FOR_FIX = "WAITING_FOR_FIX"
    GNSS_AIDED = "GNSS_AIDED"
    GNSS_DEGRADED = "GNSS_DEGRADED"
    GNSS_LOST = "GNSS_LOST"
    INS_DEAD_RECKONING = "INS_DEAD_RECKONING"
    GNSS_REACQUISITION = "GNSS_REACQUISITION"
    FUSED = "FUSED"


def _wrap_heading(degrees: float) -> float:
    return float(degrees) % 360.0


def _heading_blend(current: float | None, measured: float, weight: float) -> float:
    if current is None:
        return _wrap_heading(measured)
    delta = ((measured - current + 180.0) % 360.0) - 180.0
    return _wrap_heading(current + weight * delta)


class TemporalAccelerationModel:
    def __init__(self, model=None, max_history=20):
        self.model = model
        self.max_history = max_history
        self.history_linear = deque(maxlen=max_history)
        self.history_gyro = deque(maxlen=max_history)

    def reset_history(self):
        self.history_linear.clear()
        self.history_gyro.clear()

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
            n_features = getattr(self.model, "n_features_in_", None)
            if n_features is not None and features.shape[1] != n_features:
                return None
            prediction = self.model.predict(features)[-1]
            return float(np.clip(prediction, -8.0, 8.0))
        except Exception as exc:
            logger.warning("AI acceleration prediction failed: %s", exc)
            return None


class NavigationEngine:
    def __init__(
        self,
        map_path=None,
        model_path=None,
        gnss_timeout_s=3.0,
        forward_accel_clip=3.0,
        max_speed_mps=35.0,
        min_move_distance=0.05,
        ai_accel_gate_max=6.0,
        ai_accel_diff_max=2.0,
        _clock=None,
    ):
        self.map_path = map_path
        self.model_path = model_path
        self.gnss_timeout_s = float(gnss_timeout_s)
        # Monotonic clock function.  Override in tests to use simulated time.
        self._clock = _clock if _clock is not None else time.monotonic

        self.forward_accel_clip = float(forward_accel_clip)
        self.max_speed_mps = float(max_speed_mps)
        self.min_move_distance = float(min_move_distance)
        self.ai_accel_gate_max = float(ai_accel_gate_max)
        self.ai_accel_diff_max = float(ai_accel_diff_max)

        self.timestamp_normalizer = TimestampNormalizer()
        self.imu_preprocessor = RobustIMUPreprocessor()
        self.filter_mode = "balanced"
        self.runtime_profile = "phone"

        self.orientation = PhoneOrientation()
        self.orientation.set_vehicle_alignment([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        self.orientation.vehicle_calibrated = False
        self.alignment_calibrator = VehicleAlignmentCalibrator()
        self.alignment_state = AlignmentState.UNALIGNED
        self.alignment_confidence = 0.0
        self.alignment_source = "NONE"
        self.alignment_locked = False
        self.mounting_status = "DEFAULT"
        self.manual_alignment = False
        self.forward_phone = [0.0, 1.0, 0.0]
        self.up_phone = [0.0, 0.0, 1.0]

        self.position = np.zeros(2, dtype=float)
        self.velocity = np.zeros(2, dtype=float)

        self.heading_deg = None
        self.speed_mps = 0.0
        self.speed_source = "NONE"
        self.speed_confidence = 0.0
        self.motion_state = MotionState.STATIONARY
        self.motion_quality_score = 0.0
        self.mount_disturbance_status = "NORMAL"
        self._mount_disturbance_samples = 0

        self.origin_latitude = None
        self.origin_longitude = None

        self.last_gnss_timestamp = None
        # Backend monotonic receive time — used exclusively for freshness/timeout
        # checks so that Android clock-domain differences can never corrupt gnss_age_s.
        self.last_gnss_receive_time: float | None = None
        self.last_imu_timestamp = None
        self.gnss_accuracy_m = None
        self._alignment_course_heading_deg = None
        self._alignment_course_timestamp = None

        self.gnss_state = GNSSState.WAITING_FOR_FIX

        self.accepted_imu = 0
        self.accepted_gnss = 0
        self.rejected_samples = 0

        self.track = deque(maxlen=400)

        self.map_status = "UNAVAILABLE"
        self.map_error = None
        self.map_matcher = None
        self.map_matching_enabled = True
        self.nhc_enabled = True
        self.last_map_confidence = None
        self.last_map_nearest_road_m: float | None = None  # lateral distance to nearest road (m)
        self.last_map_matched_position: tuple[float, float] | None = None  # (lat, lon) after snapping
        self.last_raw_gnss_position: tuple[float, float] | None = None  # (lat, lon) last GNSS fix
        # Geographic bounds of the loaded road network [min_lat, max_lat, min_lon, max_lon]
        # Used to report honestly whether the current position falls inside the map area.
        self.map_bounds: tuple[float, float, float, float] | None = None
        self.map_out_of_bounds: bool = False

        self.speed_estimator = SpeedModel()
        self.accel_model = TemporalAccelerationModel()
        self.ai_model_status = "NOT_CONFIGURED"
        self.ai_model_kind = None
        self.ai_model_error = None
        self.accel_mag_history = deque(maxlen=10)

        self.nhc_status = "KINEMATIC"
        self.uncertainty_m = None
        self.vibration_rms = 0.0
        self.pitch_deg = None
        self.roll_deg = None
        self.update_rate_hz = None
        self._reacq_remaining = 0
        self._last_gnss_position = None
        self.fusion_method = "complementary_gnss_ins"

        # Yaw axis auto-detection: track which phone gyro axis (0,1,2) best
        # predicts GNSS-derived heading changes. Updated during GNSS-aided flight.
        # Default to axis 2 (Z) for a phone lying flat.
        self._gyro_yaw_axis: int = 2
        self._gyro_yaw_sign: float = -1.0   # sign: negative = CW heading when axis positive
        self._gyro_yaw_buffer_axes: list = [deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)]
        self._gyro_yaw_buffer_dt: deque = deque(maxlen=100)
        self._prev_gnss_heading: float | None = None
        self._last_mag_yaw: float | None = None

        self._load_map()
        self._load_model()
        self._refresh_constraint_status()

    def _load_map(self):
        if not self.map_path:
            self.map_status = "UNAVAILABLE"
            self.map_error = "No GeoJSON road file configured"
            return

        try:
            from map_matching import load_roads_from_geojson, VehicleMapMatcher

            roads = load_roads_from_geojson(self.map_path)
            if roads:
                self.map_matcher = VehicleMapMatcher(
                    roads,
                    use_nonholonomic=self.nhc_enabled,
                )
                self.map_status = "READY"
                self.map_error = None
                # Compute the bounding box of all road coordinates so we can
                # report honestly whether the current position falls inside the
                # map area (and thus whether NO_MATCH is expected or surprising).
                try:
                    all_lats = []
                    all_lons = []
                    for road in roads:
                        for lon, lat in road.geometry.coords:
                            all_lats.append(lat)
                            all_lons.append(lon)
                    if all_lats:
                        self.map_bounds = (
                            min(all_lats), max(all_lats),
                            min(all_lons), max(all_lons),
                        )
                except Exception:
                    self.map_bounds = None
            else:
                self.map_status = "UNAVAILABLE"
                self.map_error = "GeoJSON contained no LineString roads"
        except Exception as exc:
            self.map_error = str(exc)
            self.map_status = "ERROR"
            self.map_matcher = None

    def _load_model(self):
        self.speed_estimator = SpeedModel()
        self.accel_model.model = None
        self.ai_model_kind = None
        self.ai_model_error = None

        if not self.model_path:
            self.ai_model_status = "NOT_CONFIGURED"
            return
        if not os.path.exists(self.model_path):
            self.ai_model_status = "UNAVAILABLE"
            self.ai_model_error = f"model file not found: {self.model_path}"
            return

        try:
            candidate = SpeedModel(self.model_path)
            if candidate.available:
                self.speed_estimator = candidate
                self.ai_model_status = "READY"
                self.ai_model_kind = "speed"
                return

            import joblib

            artifact = joblib.load(self.model_path)
            model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
            features = []
            if isinstance(artifact, dict):
                features = list(artifact.get("features") or [])

            n_features = getattr(model, "n_features_in_", None)
            if features == SPEED_FEATURES or n_features == len(SPEED_FEATURES):
                self.speed_estimator.model = model
                self.speed_estimator.features = features or list(SPEED_FEATURES)
                self.speed_estimator.model_path = self.model_path
                if self.speed_estimator.available:
                    self.ai_model_status = "READY"
                    self.ai_model_kind = "speed"
                    return

            if n_features is not None and n_features >= 20 and hasattr(model, "predict"):
                self.accel_model.model = model
                self.ai_model_status = "READY"
                self.ai_model_kind = "acceleration"
                return

            self.ai_model_status = "UNAVAILABLE"
            self.ai_model_error = candidate.error or "model feature schema is not compatible"
        except Exception as exc:
            logger.warning("Could not load speed model: %s", exc)
            self.ai_model_status = "UNAVAILABLE"
            self.ai_model_error = str(exc)

    def _refresh_constraint_status(self):
        if not self.nhc_enabled:
            self.nhc_status = "DISABLED"
        elif self.map_matcher is not None and self.map_matching_enabled:
            self.nhc_status = "ROAD_CONSTRAINED"
        else:
            self.nhc_status = "KINEMATIC"

        if self.map_matcher is None:
            self.map_status = "ERROR" if self.map_error and self.map_path else "UNAVAILABLE"
        elif not self.map_matching_enabled:
            self.map_status = "DISABLED"
        elif self.map_status not in {"MATCHED", "NO_MATCH", "READY"}:
            self.map_status = "READY"

    def set_runtime_options(
        self,
        filter_mode: str | None = None,
        nhc_enabled: bool | None = None,
        map_matching_enabled: bool | None = None,
        profile: str | None = None,
        gnss_timeout_s: float | None = None,
    ) -> dict:
        if filter_mode is not None:
            normalized_filter = str(filter_mode).strip().lower()
            if normalized_filter not in {"raw", "balanced", "strict"}:
                raise ValueError("filter_mode must be raw, balanced, or strict")
            self.filter_mode = self.imu_preprocessor.set_filter_mode(normalized_filter)
        if nhc_enabled is not None:
            self.nhc_enabled = bool(nhc_enabled)
            if self.map_matcher is not None:
                self.map_matcher.use_nonholonomic = self.nhc_enabled
        if map_matching_enabled is not None:
            self.map_matching_enabled = bool(map_matching_enabled)
        if profile is not None:
            normalized = str(profile).strip().lower()
            if normalized not in {"phone", "edge"}:
                raise ValueError("profile must be phone or edge")
            self.runtime_profile = normalized
            if normalized == "edge":
                self.gnss_timeout_s = 0.25
            else:
                self.gnss_timeout_s = 1.0
        if gnss_timeout_s is not None:
            timeout = float(gnss_timeout_s)
            if not math.isfinite(timeout) or not 0.1 <= timeout <= 10.0:
                raise ValueError("gnss_timeout_s must be finite and between 0.1 and 10.0 seconds")
            self.gnss_timeout_s = timeout
            self.runtime_profile = "custom"
        self._refresh_constraint_status()
        return self.runtime_snapshot()

    def apply_manual_alignment(self, forward_phone, up_phone) -> None:
        self.orientation.set_vehicle_alignment(forward_phone, up_phone)
        self.alignment_calibrator.set_manual()
        self.alignment_state = self.alignment_calibrator.state
        self.alignment_confidence = self.alignment_calibrator.confidence
        self.alignment_source = self.alignment_calibrator.source
        self.alignment_locked = self.alignment_calibrator.locked
        self.forward_phone = list(forward_phone)
        self.up_phone = list(up_phone)
        self.manual_alignment = True
        self.mounting_status = "MANUAL"

    def runtime_snapshot(self) -> dict:
        return {
            "filter_mode": self.filter_mode,
            "nhc_enabled": self.nhc_enabled,
            "map_matching_enabled": self.map_matching_enabled,
            "map_status": self.map_status,
            "nhc_status": self.nhc_status,
            "gnss_timeout_s": self.gnss_timeout_s,
            "runtime_profile": self.runtime_profile,
            "ai_model_status": self.ai_model_status,
            "ai_model_kind": self.ai_model_kind,
            "fusion_method": self.fusion_method,
        }

    def _ll_to_xy(self, latitude, longitude):
        if self.origin_latitude is None:
            self.origin_latitude = float(latitude)
            self.origin_longitude = float(longitude)

        cos_lat = max(1e-6, math.cos(math.radians(self.origin_latitude)))
        east = math.radians(float(longitude) - self.origin_longitude) * EARTH_RADIUS_M * cos_lat
        north = math.radians(float(latitude) - self.origin_latitude) * EARTH_RADIUS_M
        return np.array([east, north], dtype=float)

    def _xy_to_ll(self, position):
        if self.origin_latitude is None:
            return None
        cos_lat = max(1e-6, math.cos(math.radians(self.origin_latitude)))
        latitude = self.origin_latitude + math.degrees(position[1] / EARTH_RADIUS_M)
        longitude = self.origin_longitude + math.degrees(position[0] / (EARTH_RADIUS_M * cos_lat))
        return latitude, longitude

    def _gnss_age(self, timestamp: float | None = None) -> float | None:  # noqa: ARG002
        """Return GNSS freshness in seconds using backend monotonic clock only.

        The ``timestamp`` argument is kept for API compatibility but is no
        longer used — comparing Android device timestamps with Python wall or
        monotonic clocks produces wildly wrong ages (the observed 12 884 954 s
        bug).  Using ``time.monotonic()`` on both sides of the subtraction
        guarantees a correct, clock-domain-consistent measurement.
        """
        if self.last_gnss_receive_time is None:
            return None
        return max(0.0, self._clock() - self.last_gnss_receive_time)

    def _update_mode_from_age(self, timestamp: float) -> None:  # noqa: ARG001
        """Transition to dead-reckoning when GNSS has been absent too long.

        Uses the same backend monotonic clock as ``_gnss_age`` so that the
        timeout decision is made from the same reference as the reported age.
        """
        if self.last_gnss_receive_time is None:
            self.gnss_state = GNSSState.WAITING_FOR_FIX
            return
        age = self._clock() - self.last_gnss_receive_time
        if age > self.gnss_timeout_s:
            self.gnss_state = GNSSState.INS_DEAD_RECKONING

    def _maybe_auto_align(self, sample, processed, gyro_phone) -> None:
        if self.manual_alignment or self.alignment_locked:
            return
        forward_phone = None
        course_age_s = None
        if (
            self.orientation.initialized
            and self._alignment_course_heading_deg is not None
            and self._alignment_course_timestamp is not None
        ):
            course_age_s = sample.timestamp - self._alignment_course_timestamp
            course_rad = math.radians(self._alignment_course_heading_deg)
            # Navigation uses [east, north, up], while course is clockwise
            # from north.  This fixes yaw from measured driving direction.
            forward_nav = np.array([math.sin(course_rad), math.cos(course_rad), 0.0])
            forward_phone = self.orientation.nav_to_phone(forward_nav)

        alignment = self.alignment_calibrator.observe(
            timestamp=sample.timestamp,
            gravity_phone=processed.gravity_phone,
            gyro_phone=gyro_phone,
            vibration_rms=processed.vibration_rms,
            linear_accel_phone=processed.filtered_linear_accel_phone,
            forward_phone=forward_phone,
            course_age_s=course_age_s,
        )
        self.alignment_state = self.alignment_calibrator.state
        self.alignment_confidence = self.alignment_calibrator.confidence
        self.alignment_source = self.alignment_calibrator.source
        self.alignment_locked = self.alignment_calibrator.locked
        if alignment is None:
            return
        forward, up = alignment
        try:
            self.orientation.set_vehicle_alignment(forward, up)
            self.forward_phone = forward.tolist()
            self.up_phone = up.tolist()
            self.mounting_status = "AUTO_LOCKED" if self.alignment_locked else "AUTO"
        except ValueError:
            return

    def _predict_ai_speed(self, linear_vehicle, gyro_vehicle) -> float | None:
        if self.ai_model_kind != "speed" or not self.speed_estimator.available:
            return None
        magnitude = float(np.linalg.norm(linear_vehicle))
        self.accel_mag_history.append(magnitude)
        smooth = float(np.mean(self.accel_mag_history))
        values = {
            "linear_accel_x": float(linear_vehicle[0]),
            "linear_accel_y": float(linear_vehicle[1]),
            "linear_accel_z": float(linear_vehicle[2]),
            "accel_magnitude": magnitude,
            "accel_magnitude_smooth": smooth,
            "gyro_magnitude": float(np.linalg.norm(gyro_vehicle)),
        }
        if not all(math.isfinite(value) for value in values.values()):
            return None
        prediction = self.speed_estimator.predict(
            values
        )
        if prediction is None or not math.isfinite(prediction):
            return None
        return float(prediction)

    def _apply_map_match(self) -> None:
        if (
            self.map_matcher is None
            or not self.map_matching_enabled
            or self.origin_latitude is None
        ):
            return

        ll = self._xy_to_ll(self.position)
        if ll is None:
            return

        # Check whether current position falls within the road network's bounding box.
        # If it doesn't, NO_MATCH is expected and we log it once rather than
        # silently reporting a misleading NO_MATCH status.
        if self.map_bounds is not None:
            min_lat, max_lat, min_lon, max_lon = self.map_bounds
            outside = not (min_lat <= ll[0] <= max_lat and min_lon <= ll[1] <= max_lon)
            if outside and not self.map_out_of_bounds:
                logger.warning(
                    "Map matching: current position (%.5f, %.5f) is outside the road "
                    "network bounding box (lat %.4f–%.4f, lon %.4f–%.4f). "
                    "NO_MATCH is expected — load a map that covers this area for real matching.",
                    ll[0], ll[1], min_lat, max_lat, min_lon, max_lon,
                )
            self.map_out_of_bounds = outside

        try:
            from map_matching import VehicleState

            matched = self.map_matcher.match(
                VehicleState(
                    latitude=ll[0],
                    longitude=ll[1],
                    heading=self.heading_deg,
                    speed=self.speed_mps,
                )
            )
        except Exception as exc:
            logger.warning("Map matching failed: %s", exc)
            self.map_status = "ERROR"
            self.map_error = str(exc)
            return

        self.last_map_confidence = float(matched.confidence)
        # Always store the lateral distance so diagnostics are useful even on NO_MATCH
        if math.isfinite(matched.lateral_error):
            self.last_map_nearest_road_m = float(matched.lateral_error)

        if matched.road_id is None or matched.confidence < 0.25:
            self.map_status = "NO_MATCH"
            self.last_map_matched_position = None
            return

        self.last_map_matched_position = (float(matched.latitude), float(matched.longitude))
        snapped = self._ll_to_xy(matched.latitude, matched.longitude)
        self.position = 0.65 * snapped + 0.35 * self.position
        self.map_status = "MATCHED"

    def _append_track(self) -> None:
        ll = self._xy_to_ll(self.position)
        if not ll:
            return
        self.track.append(
            {
                "latitude": ll[0],
                "longitude": ll[1],
                "east": float(self.position[0]),
                "north": float(self.position[1]),
            }
        )

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
            timestamp = float(timestamp)
            latitude = float(latitude)
            longitude = float(longitude)
            accuracy_m = max(0.1, float(accuracy_m))

            if not all(math.isfinite(v) for v in [timestamp, latitude, longitude, accuracy_m]):
                raise ValueError("GNSS values must be finite")

            if self.last_gnss_timestamp is not None and timestamp <= self.last_gnss_timestamp:
                raise ValueError("GNSS timestamp must be strictly increasing")

            ins_position = self.position.copy()
            new_position = self._ll_to_xy(latitude, longitude)

            previous_position = self._last_gnss_position
            gnss_heading = None
            if previous_position is not None:
                delta = new_position - previous_position
                distance = float(np.linalg.norm(delta))
                if distance > 0.5:
                    gnss_heading = _wrap_heading(
                        math.degrees(math.atan2(delta[0], delta[1]))
                    )

            was_lost = self.gnss_state in {
                GNSSState.GNSS_LOST,
                GNSSState.INS_DEAD_RECKONING,
            }
            # GNSS_DEGRADED means poor-quality but still present GNSS — not a
            # full outage. We treat it as a normal fused update so a sudden
            # accuracy improvement does not trigger an unnecessary reacquisition
            # ramp.  The existing else branch already re-evaluates accuracy on
            # every fix and will assign FUSED when the quality recovers.

            if speed_mps is not None and math.isfinite(float(speed_mps)):
                gnss_speed = max(0.0, float(speed_mps))
            else:
                gnss_speed = None

            if gnss_heading is not None and (gnss_speed is None or gnss_speed > 1.0):
                self.heading_deg = _heading_blend(self.heading_deg, gnss_heading, 0.65)
                if gnss_speed is not None and gnss_speed >= 3.0 and accuracy_m <= 15.0:
                    self._alignment_course_heading_deg = gnss_heading
                    self._alignment_course_timestamp = timestamp

            if gnss_speed is not None:
                if was_lost:
                    self.speed_mps = 0.7 * self.speed_mps + 0.3 * gnss_speed
                else:
                    self.speed_mps = gnss_speed
                self.speed_source = "GNSS"
                self.speed_confidence = 0.98 if accuracy_m <= 12.0 else 0.70

            gnss_weight = 0.85 if accuracy_m <= 12.0 else 0.45
            if self.origin_latitude is None or self.accepted_gnss == 0:
                self.position = new_position
                state = GNSSState.GNSS_AIDED
            elif was_lost:
                self.position = 0.70 * ins_position + 0.30 * new_position
                state = GNSSState.GNSS_REACQUISITION
                self._reacq_remaining = 4
            elif self._reacq_remaining > 0:
                pull = 0.40 + 0.10 * (4 - self._reacq_remaining)
                self.position = (1.0 - pull) * ins_position + pull * new_position
                self._reacq_remaining -= 1
                state = (
                    GNSSState.FUSED
                    if self._reacq_remaining == 0
                    else GNSSState.GNSS_REACQUISITION
                )
            else:
                self.position = gnss_weight * new_position + (1.0 - gnss_weight) * ins_position
                state = GNSSState.GNSS_DEGRADED if accuracy_m > 25.0 else GNSSState.FUSED

            if self.heading_deg is not None:
                heading_rad = math.radians(self.heading_deg)
                self.velocity = np.array(
                    [
                        self.speed_mps * math.sin(heading_rad),
                        self.speed_mps * math.cos(heading_rad),
                    ]
                )

            self.gnss_state = state
            self.last_gnss_timestamp = timestamp
            # Record backend receive time for freshness calculations.
            # This is the ONLY value used by _gnss_age / _update_mode_from_age.
            self.last_gnss_receive_time = self._clock()
            if self.last_imu_timestamp is None:
                self.last_imu_timestamp = timestamp
            self._last_gnss_position = new_position.copy()
            # Store the raw GNSS lat/lon before any map snapping for diagnostics.
            self.last_raw_gnss_position = (float(latitude), float(longitude))
            self.accepted_gnss += 1
            self.gnss_accuracy_m = accuracy_m
            self.uncertainty_m = accuracy_m

            # --- Yaw axis calibration: record GNSS heading change vs buffered gyro ---
            if (
                gnss_heading is not None
                and self._prev_gnss_heading is not None
                and len(self._gyro_yaw_buffer_dt) >= 5
                and gnss_speed is not None and gnss_speed > 3.0
            ):
                gnss_hdg_change = ((gnss_heading - self._prev_gnss_heading + 180) % 360) - 180
                if abs(gnss_hdg_change) > 2.0:  # only calibrate during real turns
                    # Integrate each axis over the buffered IMU interval
                    dt_arr = np.asarray(list(self._gyro_yaw_buffer_dt), dtype=float)
                    for axis in range(3):
                        ax_arr = np.asarray(list(self._gyro_yaw_buffer_axes[axis]), dtype=float)
                        integrated = float(np.sum(ax_arr * dt_arr))
                        # Find sign+scale: gnss_hdg_change = sign * integrated
                        if abs(integrated) > 0.05:  # at least 3 deg equivalent
                            sign = 1.0 if (integrated * gnss_hdg_change > 0) else -1.0
                            # Score = |correlation|
                            score = abs(gnss_hdg_change) / (abs(integrated) * 180 / math.pi + 0.1)
                            # Update running best axis
                            if not hasattr(self, '_gyro_axis_scores'):
                                self._gyro_axis_scores = [0.0, 0.0, 0.0]
                                self._gyro_axis_sign_votes = [0.0, 0.0, 0.0]
                            # Exponential moving average of |residual|
                            predicted = sign * integrated * 180 / math.pi
                            residual = abs(gnss_hdg_change - predicted)
                            self._gyro_axis_scores[axis] = 0.9 * self._gyro_axis_scores[axis] + 0.1 * residual
                            self._gyro_axis_sign_votes[axis] += sign
                    # Select axis with lowest residual (best prediction)
                    if hasattr(self, '_gyro_axis_scores'):
                        best_axis = int(np.argmin(self._gyro_axis_scores))
                        best_sign = 1.0 if self._gyro_axis_sign_votes[best_axis] >= 0 else -1.0
                        if self._gyro_yaw_axis != best_axis:
                            logger.info(
                                "Yaw axis updated: %d -> %d (sign %+.0f)",
                                self._gyro_yaw_axis, best_axis, best_sign,
                            )
                        self._gyro_yaw_axis = best_axis
                        self._gyro_yaw_sign = best_sign

            self._prev_gnss_heading = gnss_heading if gnss_heading is not None else self._prev_gnss_heading
            # Clear gyro buffers after using them for calibration
            for buf in self._gyro_yaw_buffer_axes:
                buf.clear()
            self._gyro_yaw_buffer_dt.clear()

            self._apply_map_match()
            self._append_track()
            return self.state_snapshot()

        except Exception:
            self.rejected_samples += 1
            raise

    def process_imu(self, timestamp, accel, gyro, mag=None):
        try:
            sample = self.timestamp_normalizer.accept(timestamp)
            dt = sample.dt

            processed = self.imu_preprocessor.update(
                accel=accel,
                gyro=gyro,
                mag=mag,
                dt=dt,
            )
            self.vibration_rms = float(processed.vibration_rms)
            self.motion_state = processed.motion_state
            self.motion_quality_score = float(processed.quality_score)
            if processed.shock_detected and float(np.linalg.norm(processed.raw_gyro)) > 1.5:
                self._mount_disturbance_samples = 20
                self.mount_disturbance_status = "SUSPECTED"
            elif self._mount_disturbance_samples > 0:
                self._mount_disturbance_samples -= 1
                if self._mount_disturbance_samples == 0:
                    self.mount_disturbance_status = "NORMAL"
            self.accepted_imu += 1
            self.last_imu_timestamp = sample.timestamp
            self.update_rate_hz = 1.0 / max(dt, 1e-3)

            mag_in = processed.raw_mag if processed.raw_mag is not None else np.zeros(3)
            oriented = self.orientation.update(
                accel=processed.raw_accel,
                gyro=processed.raw_gyro,
                mag=mag_in,
                dt=dt,
            )
            linear_vehicle = self.orientation.q_phone_to_vehicle
            from alignment import _quat_rotate

            linear_vehicle = _quat_rotate(
                self.orientation.q_phone_to_vehicle,
                processed.filtered_linear_accel_phone,
            )
            gyro_vehicle = oriented["gyro"]

            if self.orientation.initialized:
                self.pitch_deg, self.roll_deg, _yaw = self.orientation.euler_pitch_roll_yaw_deg()

            self._update_mode_from_age(sample.timestamp)
            self._maybe_auto_align(sample, processed, processed.raw_gyro)

            if self.heading_deg is None:
                self.heading_deg = 0.0

            # --- Heading update ---
            # When alignment is locked we use the calibrated vehicle-frame gyro Z.
            # When alignment is still uncertain we use ALL three raw gyro axes by
            # picking the one whose current reading has the largest |rate| (proxy
            # for the actual vertical rotation) — this avoids the common failure
            # where the vehicle yaw maps onto gyro_y (pitch) rather than gyro_z.
            # Additionally, blend in the magnetometer-aided yaw from the orientation
            # quaternion so that slow heading drift is suppressed.
            if self.alignment_locked:
                # Calibrated path: vehicle-frame gyro Z drives heading
                yaw_rate_deg = -math.degrees(float(gyro_vehicle[2]))
                self.heading_deg = _wrap_heading(self.heading_deg + yaw_rate_deg * dt)
            else:
                # Uncalibrated path: use the auto-detected yaw axis with LPF gyro.
                # Buffer all three axes so GNSS can calibrate which axis is yaw.
                gyro_filt = np.asarray(processed.filtered_gyro_phone, dtype=float)
                for axis in range(3):
                    self._gyro_yaw_buffer_axes[axis].append(float(gyro_filt[axis]))
                self._gyro_yaw_buffer_dt.append(dt)

                # Integrate heading using the calibrated yaw axis
                yaw_rate = self._gyro_yaw_sign * float(gyro_filt[self._gyro_yaw_axis])
                yaw_rate_deg = math.degrees(yaw_rate)
                self.heading_deg = _wrap_heading(self.heading_deg + yaw_rate_deg * dt)

                # During dead reckoning, blend magnetometer-derived heading CHANGE
                # (relative, not absolute) to correct gyro drift.
                if (
                    self.gnss_state == GNSSState.INS_DEAD_RECKONING
                    and self.orientation.initialized
                    and processed.raw_mag is not None
                    and np.linalg.norm(processed.raw_mag) > 1.0
                ):
                    _, _, mag_yaw = self.orientation.euler_pitch_roll_yaw_deg()
                    if self._last_mag_yaw is None:
                        self._last_mag_yaw = mag_yaw
                    delta_mag_yaw = ((mag_yaw - self._last_mag_yaw + 180) % 360) - 180
                    self._last_mag_yaw = mag_yaw
                    if abs(delta_mag_yaw) < 20.0:  # reject large jumps
                        self.heading_deg = _wrap_heading(self.heading_deg + 0.3 * delta_mag_yaw)

            quality_weight = {
                MotionState.SHOCK: 0.0,
                MotionState.UNRELIABLE: 0.15,
                MotionState.VIBRATION: 0.35,
            }.get(processed.motion_state, 1.0)
            forward_accel = float(linear_vehicle[0]) * quality_weight
            ai_accel = None
            if (
                self.gnss_state == GNSSState.INS_DEAD_RECKONING
                and self.ai_model_kind == "acceleration"
            ):
                ai_accel = self.accel_model.update(
                    linear_vehicle.tolist(),
                    np.asarray(gyro_vehicle, dtype=float).tolist(),
                )
            else:
                self.accel_model.history_linear.append(linear_vehicle.tolist())
                self.accel_model.history_gyro.append(np.asarray(gyro_vehicle, dtype=float).tolist())

            if ai_accel is not None and math.isfinite(ai_accel):
                if abs(ai_accel) <= self.ai_accel_gate_max and abs(ai_accel - forward_accel) <= self.ai_accel_diff_max:
                    forward_accel = ai_accel

            forward_accel = float(
                np.clip(forward_accel, -self.forward_accel_clip, self.forward_accel_clip)
            )

            previous_speed = self.speed_mps
            truly_stopped = (
                processed.motion_state in {MotionState.STATIONARY, MotionState.IDLING}
                and self.speed_mps < 1.0
            )
            if truly_stopped:
                self.speed_mps = 0.0
                self.speed_source = "STATIONARY"
                self.speed_confidence = 0.98 if processed.motion_state == MotionState.STATIONARY else 0.85
            else:
                integrated = float(
                    np.clip(previous_speed + forward_accel * dt, 0.0, self.max_speed_mps)
                )
                ai_speed = None
                if (
                    self.gnss_state == GNSSState.INS_DEAD_RECKONING
                    and processed.motion_state == MotionState.MOVING
                    and processed.quality_score >= 0.65
                    and self.mount_disturbance_status == "NORMAL"
                ):
                    ai_speed = self._predict_ai_speed(linear_vehicle, gyro_vehicle)
                ai_is_plausible = (
                    ai_speed is not None
                    and 0.0 <= ai_speed <= self.max_speed_mps
                    and abs(ai_speed - integrated) <= max(4.0, 0.75 * max(integrated, 1.0))
                )
                if ai_is_plausible:
                    self.speed_mps = float(np.clip(0.65 * integrated + 0.35 * ai_speed, 0.0, self.max_speed_mps))
                    self.speed_source = "ML"
                    self.speed_confidence = min(0.75, 0.45 + 0.35 * processed.quality_score)
                elif self.gnss_state in {GNSSState.FUSED, GNSSState.GNSS_AIDED, GNSSState.GNSS_DEGRADED, GNSSState.GNSS_REACQUISITION}:
                    self.speed_mps = integrated
                    if self.speed_source != "GNSS":
                        self.speed_source = "INERTIAL"
                        self.speed_confidence = min(0.70, 0.35 + 0.40 * processed.quality_score)
                else:
                    self.speed_mps = integrated
                    self.speed_source = "INERTIAL"
                    self.speed_confidence = min(0.55, 0.20 + 0.40 * processed.quality_score)

            heading_rad = math.radians(self.heading_deg)
            distance = 0.5 * (previous_speed + self.speed_mps) * dt
            if abs(distance) < self.min_move_distance or truly_stopped:
                distance = 0.0

            if self.nhc_enabled:
                displacement = np.array(
                    [
                        distance * math.sin(heading_rad),
                        distance * math.cos(heading_rad),
                    ]
                )
            else:
                displacement = np.array(
                    [
                        distance * math.sin(heading_rad) + float(linear_vehicle[1]) * dt * dt,
                        distance * math.cos(heading_rad),
                    ]
                )

            if self.origin_latitude is not None:
                self.position += displacement

            self.velocity = self.speed_mps * np.array(
                [math.sin(heading_rad), math.cos(heading_rad)]
            )

            if self.gnss_state == GNSSState.INS_DEAD_RECKONING:
                self.uncertainty_m = 2.0 if self.uncertainty_m is None else self.uncertainty_m + 0.05
                self._apply_map_match()

            self._append_track()
            return self.state_snapshot()

        except Exception:
            self.rejected_samples += 1
            raise

    def state_snapshot(self):
        position = self._xy_to_ll(self.position)
        gnss_age = self._gnss_age()

        if self.gnss_state == GNSSState.INS_DEAD_RECKONING:
            mode = "DEAD_RECKONING"
        elif self.gnss_state == GNSSState.GNSS_REACQUISITION:
            mode = "REACQUISITION"
        elif self.gnss_state == GNSSState.FUSED:
            mode = "GNSS_INS_FUSED"
        elif self.gnss_state == GNSSState.GNSS_AIDED:
            mode = "GNSS_AIDED"
        elif self.gnss_state == GNSSState.GNSS_DEGRADED:
            mode = "GNSS_DEGRADED"
        else:
            mode = self.gnss_state.value

        if self.gnss_state in {GNSSState.GNSS_LOST, GNSSState.INS_DEAD_RECKONING}:
            gnss_status = "LOST"
        elif self.gnss_state == GNSSState.GNSS_DEGRADED:
            gnss_status = "DEGRADED"
        elif self.gnss_state in {GNSSState.GNSS_AIDED, GNSSState.FUSED, GNSSState.GNSS_REACQUISITION}:
            gnss_status = "CONNECTED"
        else:
            gnss_status = self.gnss_state.value

        yaw_deg = self.heading_deg
        return {
            "position": None if position is None else {"latitude": position[0], "longitude": position[1]},
            "imu_status": "ACTIVE" if self.accepted_imu > 0 else "INACTIVE",
            "local_position_m": {
                "east": float(self.position[0]),
                "north": float(self.position[1]),
            },
            "speed_mps": float(self.speed_mps),
            "speed_kmh": float(self.speed_mps * 3.6),
            "speed_source": self.speed_source,
            "speed_confidence": float(self.speed_confidence),
            "motion_state": self.motion_state.value,
            "motion_quality_score": float(self.motion_quality_score),
            "mount_disturbance_status": self.mount_disturbance_status,
            "heading_deg": self.heading_deg,
            "yaw_deg": yaw_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "gnss_state": self.gnss_state.value,
            "gnss_status": gnss_status,
            "gnss_age_s": gnss_age,
            "last_gnss_backend_receive_time": self.last_gnss_receive_time,
            "mode": mode,
            "uncertainty_m": self.uncertainty_m,
            "nhc_status": self.nhc_status,
            "nhc_enabled": self.nhc_enabled,
            "ai_model_status": self.ai_model_status,
            "ai_model_kind": self.ai_model_kind,
            "ai_model_error": self.ai_model_error,
            "fusion_method": self.fusion_method,
            "fusion_is_ai": False,
            "map_status": self.map_status,
            "map_error": self.map_error,
            "map_matching_enabled": self.map_matching_enabled,
            "map_confidence": self.last_map_confidence,
            "map_out_of_bounds": self.map_out_of_bounds,
            "nearest_road_distance_m": self.last_map_nearest_road_m,
            "raw_gnss_position": (
                {"latitude": self.last_raw_gnss_position[0], "longitude": self.last_raw_gnss_position[1]}
                if self.last_raw_gnss_position is not None else None
            ),
            "map_matched_position": (
                {"latitude": self.last_map_matched_position[0], "longitude": self.last_map_matched_position[1]}
                if self.last_map_matched_position is not None else None
            ),
            "map_bounds": (
                {
                    "min_lat": self.map_bounds[0],
                    "max_lat": self.map_bounds[1],
                    "min_lon": self.map_bounds[2],
                    "max_lon": self.map_bounds[3],
                }
                if self.map_bounds is not None else None
            ),
            "mounting_calibrated": bool(self.orientation.vehicle_calibrated),
            "mounting_status": self.mounting_status,
            "alignment_state": self.alignment_state.value,
            "alignment_confidence": float(self.alignment_confidence),
            "alignment_source": self.alignment_source,
            "alignment_locked": self.alignment_locked,
            "orientation_initialized": bool(self.orientation.initialized),
            "vibration_rms": float(self.vibration_rms),
            "filter_mode": self.filter_mode,
            "filter_status": self.filter_mode.upper(),
            "gnss_timeout_s": self.gnss_timeout_s,
            "runtime_profile": self.runtime_profile,
            "update_rate_hz": self.update_rate_hz,
            "accepted_imu": self.accepted_imu,
            "accepted_gnss": self.accepted_gnss,
            "rejected_samples": self.rejected_samples,
            "track": list(self.track),
            "timestamp": self.last_imu_timestamp if self.last_imu_timestamp is not None else self.last_gnss_timestamp,
        }

    def reset(self):
        saved_filter = self.filter_mode
        self.timestamp_normalizer = TimestampNormalizer()
        self.imu_preprocessor = RobustIMUPreprocessor()
        self.imu_preprocessor.set_filter_mode(saved_filter)
        self.orientation.reset_attitude()
        self.accel_model.reset_history()
        self.accel_mag_history.clear()

        self.position[:] = 0.0
        self.velocity[:] = 0.0
        self.heading_deg = None
        self.speed_mps = 0.0
        self.speed_source = "NONE"
        self.speed_confidence = 0.0
        self.motion_state = MotionState.STATIONARY
        self.motion_quality_score = 0.0
        self.mount_disturbance_status = "NORMAL"
        self._mount_disturbance_samples = 0
        self.origin_latitude = None
        self.origin_longitude = None
        self.last_gnss_timestamp = None
        self.last_gnss_receive_time = None
        self.last_imu_timestamp = None
        self.gnss_accuracy_m = None
        self.gnss_state = GNSSState.WAITING_FOR_FIX
        self.accepted_imu = 0
        self.accepted_gnss = 0
        self.rejected_samples = 0
        self.track.clear()
        self.uncertainty_m = None
        self.vibration_rms = 0.0
        self.pitch_deg = None
        self.roll_deg = None
        self.update_rate_hz = None
        self._reacq_remaining = 0
        self._last_gnss_position = None
        self.last_map_confidence = None
        self.last_map_nearest_road_m = None
        self.last_map_matched_position = None
        self.last_raw_gnss_position = None
        self.map_out_of_bounds = False
        self._alignment_course_heading_deg = None
        self._alignment_course_timestamp = None
        if self.manual_alignment:
            self.alignment_calibrator.set_manual()
        self.alignment_state = self.alignment_calibrator.state
        self.alignment_confidence = self.alignment_calibrator.confidence
        self.alignment_source = self.alignment_calibrator.source
        self.alignment_locked = self.alignment_calibrator.locked
        self._refresh_constraint_status()
