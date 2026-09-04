"""Live, sensor-source-independent Intelligent Dead Reckoning engine.

The legacy CSV workflow remains in ``main.py``.  This engine is the runtime
path used by FastAPI, Android, external IMUs, and replay tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import os
from typing import Any
from collections import deque

import numpy as np
from joblib import load

from alignment import PhoneOrientation
from map_matching import VehicleMapMatcher, load_roads_from_geojson
from sensor_processing import (
    RobustIMUPreprocessor,
    TimestampNormalizer,
)


EARTH_RADIUS_M = 6_378_137.0


# The trained VBOX model is an acceleration model.  Its feature
# generator is kept identical to train_speed_model_vbox.py so training and
# runtime inference use the same 10 Hz causal feature definition.
class TemporalAccelerationModel:
    def __init__(self, model_path: str | None):
        self.model = None
        self.features: list[str] = []
        self.error: str | None = None
        self.history_linear = deque(maxlen=20)
        self.history_gyro = deque(maxlen=20)

        if not model_path:
            return

        if not os.path.exists(model_path):
            self.error = f"model file not found: {model_path}"
            return

        try:
            bundle = load(model_path)
            self.model = bundle["model"]
            self.features = list(bundle["features"])
        except Exception as exc:
            self.error = f"acceleration model could not be loaded: {exc}"

    def reset(self) -> None:
        self.history_linear.clear()
        self.history_gyro.clear()

    def update(
        self,
        linear_phone,
        gyro_phone,
    ) -> float | None:
        if self.model is None:
            return None

        self.history_linear.append(
            np.asarray(linear_phone, dtype=float).copy()
        )
        self.history_gyro.append(
            np.asarray(gyro_phone, dtype=float).copy()
        )

        try:
            # Import the exact feature builder used during training.
            from train_speed_model_vbox import build_features

            linear = np.asarray(
                list(self.history_linear),
                dtype=float,
            )
            gyro = np.asarray(
                list(self.history_gyro),
                dtype=float,
            )

            feature_df = build_features(
                linear,
                gyro,
            )

            row = feature_df.iloc[-1:]

            # Use the exact feature names saved with the model.
            row = row.reindex(
                columns=self.features,
                fill_value=0.0,
            )

            prediction = float(
                self.model.predict(row)[0]
            )

            return float(
                np.clip(prediction, -8.0, 8.0)
            )

        except Exception as exc:
            self.error = f"acceleration inference failed: {exc}"
            return None


class GNSSState(str, Enum):
    WAITING_FOR_FIX = "WAITING_FOR_FIX"
    GNSS_AIDED = "GNSS_AIDED"
    GNSS_DEGRADED = "GNSS_DEGRADED"
    GNSS_LOST = "GNSS_LOST"
    INS_DEAD_RECKONING = "INS_DEAD_RECKONING"
    GNSS_REACQUISITION = "GNSS_REACQUISITION"
    FUSED = "FUSED"


@dataclass
class GNSSFix:
    timestamp: float
    latitude: float
    longitude: float
    speed_mps: float | None
    accuracy_m: float
    altitude_m: float | None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _wrap_heading(value: float) -> float:
    return float(value % 360.0)


def _bearing_from_delta(east: float, north: float) -> float | None:
    if math.hypot(east, north) < 1e-6:
        return None
    return _wrap_heading(math.degrees(math.atan2(east, north)))


class NavigationEngine:
    """Incremental GNSS/IMU fusion with outage handling and optional maps."""

    def __init__(
        self,
        *,
        forward_phone=(0.0, 1.0, 0.0),
        up_phone=(0.0, 0.0, 1.0),
        gnss_timeout_s: float = 1.5,
        gnss_degraded_accuracy_m: float = 35.0,
        map_path: str | None = None,
        model_path: str | None = None,
    ):
        self.forward_phone = np.asarray(forward_phone, dtype=float)
        self.up_phone = np.asarray(up_phone, dtype=float)
        self.gnss_timeout_s = float(gnss_timeout_s)
        self.gnss_degraded_accuracy_m = float(gnss_degraded_accuracy_m)
        self.map_path = map_path
        self.accel_model = TemporalAccelerationModel(model_path)
        self.map_matcher: VehicleMapMatcher | None = None
        self.map_error: str | None = None
        self._load_map(map_path)
        self.reset()

    def _load_map(self, map_path: str | None) -> None:
        if not map_path:
            return
        if not os.path.exists(map_path):
            self.map_error = f"map file not found: {map_path}"
            return
        try:
            roads = load_roads_from_geojson(map_path)
            if roads:
                self.map_matcher = VehicleMapMatcher(roads)
            else:
                self.map_error = "map contains no LineString roads"
        except Exception as exc:  # map data must not take down live navigation
            self.map_error = f"map could not be loaded: {exc}"

    def reset(self) -> None:
        self.imu_clock = TimestampNormalizer(default_dt=0.01, max_dt=1.0)
        self.gnss_clock = TimestampNormalizer(default_dt=0.1, max_dt=10.0)
        self.preprocessor = RobustIMUPreprocessor()
        self.accel_model.reset()
        self.orientation = PhoneOrientation()
        self.orientation.set_vehicle_alignment(
            self.forward_phone,
            self.up_phone,
        )

        self.state = GNSSState.WAITING_FOR_FIX
        self.origin_latitude: float | None = None
        self.origin_longitude: float | None = None
        self.position_xy = np.zeros(2, dtype=float)
        self.speed_mps = 0.0
        self.heading_deg: float | None = None
        self.uncertainty_m = float("inf")
        self.last_imu_timestamp: float | None = None
        self.last_gnss_timestamp: float | None = None
        self.last_gnss_fix: GNSSFix | None = None
        self.previous_gnss_xy: np.ndarray | None = None
        self.previous_gnss_timestamp: float | None = None
        self.reacquisition_count = 0
        self.sequence = 0
        self.accepted_imu = 0
        self.accepted_gnss = 0
        self.rejected_samples = 0
        self.nhc_active = True
        self.last_processed_timestamp: float | None = None
        self.last_vibration_rms = 0.0
        self.accel_magnitude_smooth = 0.0
        self.speed_source = "inertial"
        self.map_status = "READY" if self.map_matcher is not None else (
            "ERROR" if self.map_error is not None else "UNAVAILABLE"
        )
        self.map_confidence = 0.0
        self.map_road_id: str | None = None
        self.track: list[dict[str, float]] = []

    # --------------------------- coordinate helpers ---------------------
    def _set_origin_if_needed(self, latitude: float, longitude: float) -> None:
        if self.origin_latitude is None:
            self.origin_latitude = latitude
            self.origin_longitude = longitude

    def _ll_to_xy(self, latitude: float, longitude: float) -> np.ndarray:
        self._set_origin_if_needed(latitude, longitude)
        lat0 = math.radians(float(self.origin_latitude))
        return np.array(
            [
                math.radians(longitude - self.origin_longitude)
                * EARTH_RADIUS_M
                * math.cos(lat0),
                math.radians(latitude - self.origin_latitude)
                * EARTH_RADIUS_M,
            ],
            dtype=float,
        )

    def _xy_to_ll(self, position: np.ndarray) -> tuple[float, float] | None:
        if self.origin_latitude is None or self.origin_longitude is None:
            return None
        lat = self.origin_latitude + math.degrees(position[1] / EARTH_RADIUS_M)
        lon = self.origin_longitude + math.degrees(
            position[0]
            / (EARTH_RADIUS_M * math.cos(math.radians(self.origin_latitude)))
        )
        return float(lat), float(lon)

    # --------------------------- health/state ----------------------------
    def _refresh_state(self, timestamp: float) -> None:
        if self.last_gnss_timestamp is None:
            if self.origin_latitude is None:
                self.state = GNSSState.WAITING_FOR_FIX
            return

        age = max(0.0, timestamp - self.last_gnss_timestamp)
        if age > self.gnss_timeout_s:
            if self.state not in (
                GNSSState.GNSS_LOST,
                GNSSState.INS_DEAD_RECKONING,
            ):
                self.state = GNSSState.GNSS_LOST
            else:
                self.state = GNSSState.INS_DEAD_RECKONING

    def _mode(self) -> str:
        return {
            GNSSState.WAITING_FOR_FIX: "WAITING FOR GNSS",
            GNSSState.GNSS_AIDED: "GNSS + INS",
            GNSSState.GNSS_DEGRADED: "GNSS DEGRADED",
            GNSSState.GNSS_LOST: "GNSS LOST",
            GNSSState.INS_DEAD_RECKONING: "DEAD RECKONING",
            GNSSState.GNSS_REACQUISITION: "REACQUISITION",
            GNSSState.FUSED: "FUSED",
        }[self.state]

    def _apply_map_matching(self) -> None:
        if self.map_matcher is None:
            self.map_status = (
                "UNAVAILABLE" if self.map_error is None else "ERROR"
            )
            return
        ll = self._xy_to_ll(self.position_xy)
        if ll is None:
            self.map_status = "WAITING_FOR_POSITION"
            return
        try:
            matched = self.map_matcher.match(
                state=self._vehicle_state(ll),
            )
        except Exception:
            self.map_status = "ERROR"
            return

        self.map_confidence = float(matched.confidence)
        self.map_road_id = matched.road_id
        if matched.road_id is not None and matched.confidence >= 0.55:
            matched_xy = self._ll_to_xy(matched.latitude, matched.longitude)
            # Conservative correction. It uses the road only when both
            # proximity and heading checks in VehicleMapMatcher passed.
            self.position_xy += 0.35 * (matched_xy - self.position_xy)
            self.map_status = "MATCHED"
        else:
            self.map_status = "NO_CONFIDENT_MATCH"

    def _vehicle_state(self, ll):
        # Imported lazily to keep the engine importable if a deployment does
        # not enable map matching.
        from map_matching import VehicleState

        return VehicleState(
            latitude=ll[0],
            longitude=ll[1],
            heading=self.heading_deg,
            speed=self.speed_mps,
        )

    # --------------------------- sensor inputs ---------------------------
    def process_gnss(
        self,
        *,
        timestamp: float,
        latitude: float,
        longitude: float,
        speed_mps: float | None = None,
        accuracy_m: float = 10.0,
        altitude_m: float | None = None,
    ) -> dict[str, Any]:
        try:
            sample = self.gnss_clock.accept(timestamp)
            latitude = float(latitude)
            longitude = float(longitude)
            accuracy_m = float(accuracy_m)
            if not (
                _finite(latitude)
                and -90.0 <= latitude <= 90.0
                and _finite(longitude)
                and -180.0 <= longitude <= 180.0
            ):
                raise ValueError("latitude/longitude are invalid")
            if not _finite(accuracy_m) or accuracy_m <= 0:
                raise ValueError("accuracy_m must be positive")
            if speed_mps is not None:
                speed_mps = float(speed_mps)
                if not _finite(speed_mps) or speed_mps < 0:
                    raise ValueError("speed_mps must be non-negative")
            if altitude_m is not None:
                altitude_m = float(altitude_m)
                if not _finite(altitude_m):
                    altitude_m = None
        except (TypeError, ValueError) as exc:
            self.rejected_samples += 1
            raise ValueError(str(exc)) from exc

        had_origin = self.origin_latitude is not None
        xy = self._ll_to_xy(latitude, longitude)
        previous_gnss_xy = (
            None if self.previous_gnss_xy is None else self.previous_gnss_xy.copy()
        )
        previous_gnss_timestamp = self.previous_gnss_timestamp
        previous_state = self.state
        if previous_gnss_xy is not None and previous_gnss_timestamp is not None:
            gnss_dt = sample.timestamp - previous_gnss_timestamp
            distance = float(np.linalg.norm(xy - previous_gnss_xy))
            maximum = max(100.0, ((speed_mps or self.speed_mps) + 25.0) * max(gnss_dt, 0.1))
            if distance > maximum:
                self.rejected_samples += 1
                self.state = GNSSState.GNSS_DEGRADED
                return self._build_output(sample.timestamp, source="gnss_jump_rejected")

        fix = GNSSFix(
            timestamp=sample.timestamp,
            latitude=latitude,
            longitude=longitude,
            speed_mps=speed_mps,
            accuracy_m=accuracy_m,
            altitude_m=altitude_m,
        )
        self.last_gnss_fix = fix
        self.last_gnss_timestamp = sample.timestamp
        self.previous_gnss_xy = xy.copy()
        self.previous_gnss_timestamp = sample.timestamp
        self.accepted_gnss += 1

        if had_origin and previous_state in (
            GNSSState.GNSS_LOST,
            GNSSState.INS_DEAD_RECKONING,
        ):
            self.state = GNSSState.GNSS_REACQUISITION
            self.reacquisition_count = 1
        elif not had_origin:
            self.state = GNSSState.GNSS_AIDED
            self.position_xy = xy
        else:
            if previous_state == GNSSState.GNSS_REACQUISITION:
                self.reacquisition_count += 1
            if self.state == GNSSState.GNSS_DEGRADED:
                self.state = GNSSState.GNSS_AIDED

            correction_gain = 0.35
            self.position_xy += correction_gain * (xy - self.position_xy)

        if speed_mps is not None:
            if self.speed_mps == 0.0 or previous_state in (
                GNSSState.WAITING_FOR_FIX,
                GNSSState.GNSS_LOST,
                GNSSState.INS_DEAD_RECKONING,
            ):
                self.speed_mps = speed_mps
            else:
                self.speed_mps = 0.5 * self.speed_mps + 0.5 * speed_mps

        if previous_gnss_xy is not None and self.heading_deg is None:
            self.heading_deg = _bearing_from_delta(
                xy[0] - previous_gnss_xy[0],
                xy[1] - previous_gnss_xy[1],
            )

        if self.state == GNSSState.GNSS_REACQUISITION and self.reacquisition_count >= 3:
            self.state = GNSSState.FUSED
        elif accuracy_m > self.gnss_degraded_accuracy_m:
            self.state = GNSSState.GNSS_DEGRADED

        self.uncertainty_m = max(float(accuracy_m), self.uncertainty_m * 0.75 if math.isfinite(self.uncertainty_m) else float(accuracy_m))
        self._apply_map_matching()
        return self._build_output(sample.timestamp, source="gnss")

    def process_imu(
        self,
        *,
        timestamp: float,
        accel,
        gyro,
        mag=None,
    ) -> dict[str, Any]:
        try:
            sample = self.imu_clock.accept(timestamp)
            processed = self.preprocessor.update(
                accel=accel,
                gyro=gyro,
                mag=mag,
                dt=sample.dt,
            )
        except (TypeError, ValueError) as exc:
            self.rejected_samples += 1
            raise ValueError(str(exc)) from exc

        self._refresh_state(sample.timestamp)
        try:
            orientation_result = self.orientation.update(
                accel=processed.raw_accel,
                gyro=processed.raw_gyro,
                mag=processed.raw_mag if processed.raw_mag is not None else np.zeros(3),
                dt=sample.dt,
            )
            vehicle_linear = self.orientation.q_phone_to_vehicle
            # Reuse the established alignment transform; gravity was removed
            # before this step in phone coordinates.
            from alignment import _quat_rotate

            linear_vehicle = _quat_rotate(
                vehicle_linear,
                processed.filtered_linear_accel_phone,
            )
            gyro_vehicle = _quat_rotate(
                vehicle_linear,
                processed.raw_gyro,
            )
        except Exception as exc:
            self.rejected_samples += 1
            raise ValueError(f"orientation processing failed: {exc}") from exc

        forward_accel = float(linear_vehicle[0])
        if self.heading_deg is not None:
            self.heading_deg = _wrap_heading(
                self.heading_deg
                + math.degrees(float(gyro_vehicle[2]) * sample.dt)
            )
        elif abs(forward_accel) > 0.2:
            # Heading remains unknown until GNSS/course or a future alignment
            # calibration supplies it; do not invent north/east direction.
            self.heading_deg = None

        self.accel_magnitude_smooth = (
            0.8 * self.accel_magnitude_smooth
            + 0.2 * float(np.linalg.norm(linear_vehicle))
        )
        # Keep the exact phone-frame IMU history used by the trained
        # acceleration model.  During GNSS loss, the model estimates dv/dt;
        # speed is then propagated from the last GNSS-aided speed.
        ai_acceleration = self.accel_model.update(
            processed.filtered_linear_accel_phone,
            processed.raw_gyro,
        )

        outage_mode = self.state in (
            GNSSState.GNSS_LOST,
            GNSSState.INS_DEAD_RECKONING,
        )

        if outage_mode and ai_acceleration is not None:
            speed_accel = ai_acceleration
            self.speed_source = "ai-acceleration+inertial"
        else:
            speed_accel = forward_accel
            self.speed_source = "inertial"

        integrated_speed = max(
            0.0,
            self.speed_mps + speed_accel * sample.dt,
        )
        self.speed_mps = integrated_speed
        if self.origin_latitude is not None and self.heading_deg is not None:
            previous_speed = self.speed_mps - speed_accel * sample.dt
            distance = 0.5 * (max(0.0, previous_speed) + self.speed_mps) * sample.dt
            heading_rad = math.radians(self.heading_deg)
            self.position_xy += np.array(
                [math.sin(heading_rad) * distance, math.cos(heading_rad) * distance]
            )

        self.accepted_imu += 1
        self.last_imu_timestamp = sample.timestamp
        self.last_processed_timestamp = sample.timestamp
        self.last_vibration_rms = processed.vibration_rms
        if self.origin_latitude is not None:
            if self.state in (GNSSState.GNSS_LOST, GNSSState.INS_DEAD_RECKONING):
                self.state = GNSSState.INS_DEAD_RECKONING
            outage_age = (
                sample.timestamp - self.last_gnss_timestamp
                if self.last_gnss_timestamp is not None
                else 0.0
            )
            if outage_age > 0:
                self.uncertainty_m = max(
                    self.uncertainty_m if math.isfinite(self.uncertainty_m) else 2.0,
                    2.0,
                ) + 0.15 * self.speed_mps * sample.dt + 0.03 * sample.dt
        self._apply_map_matching()
        return self._build_output(sample.timestamp, source="imu")

    # --------------------------- output ----------------------------------
    def _build_output(self, timestamp: float, source: str) -> dict[str, Any]:
        ll = self._xy_to_ll(self.position_xy)
        self.sequence += 1
        if source != "state" and ll is not None:
            self.track.append(
                {
                    "east": float(self.position_xy[0]),
                    "north": float(self.position_xy[1]),
                }
            )
            # Keep the dashboard lightweight during a long phone session.
            if len(self.track) > 400:
                del self.track[:-400]
        return {
            "ok": True,
            "sequence": self.sequence,
            "timestamp": float(timestamp),
            "source": source,
            "position": (
                None
                if ll is None
                else {"latitude": ll[0], "longitude": ll[1]}
            ),
            "local_position_m": {
                "east": float(self.position_xy[0]),
                "north": float(self.position_xy[1]),
            },
            "speed_mps": float(self.speed_mps),
            "speed_kmh": float(self.speed_mps * 3.6),
            "speed_source": self.speed_source,
            "ai_model_status": "READY" if self.accel_model.model is not None else "UNAVAILABLE",
            "ai_model_error": self.accel_model.error,
            "heading_deg": self.heading_deg,
            "uncertainty_m": (
                None
                if not math.isfinite(self.uncertainty_m)
                else float(self.uncertainty_m)
            ),
            "gnss_state": self.state.value,
            "mode": self._mode(),
            "imu_status": "ACTIVE" if self.last_imu_timestamp is not None else "WAITING",
            "gnss_status": (
                "CONNECTED"
                if self.last_gnss_timestamp is not None
                and timestamp - self.last_gnss_timestamp <= self.gnss_timeout_s
                else "LOST"
            ),
            "nhc_status": "ACTIVE" if self.nhc_active else "DISABLED",
            "map_status": self.map_status,
            "map_road_id": self.map_road_id,
            "map_confidence": float(self.map_confidence),
            "orientation_initialized": bool(self.orientation.initialized),
            "mounting_calibrated": bool(self.orientation.vehicle_calibrated),
            "vibration_rms": float(self.last_vibration_rms),
            "accepted_imu": self.accepted_imu,
            "accepted_gnss": self.accepted_gnss,
            "rejected_samples": self.rejected_samples,
            "track": list(self.track),
        }

    def state_snapshot(self) -> dict[str, Any]:
        timestamp = self.last_processed_timestamp or self.last_gnss_timestamp or 0.0
        self._refresh_state(timestamp)
        return self._build_output(timestamp, source="state")