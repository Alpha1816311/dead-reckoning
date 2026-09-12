"""
IDR Sensor Interface — canonical event schema and adapters.

Defines the standard sensor event format used across:
  - Replay ingestion
  - Live WebSocket streaming
  - Android sensor transport
  - Future external IMU sources

Canonical IMU event::

    {
        "timestamp": 123456.789,   # seconds (monotonic or Unix)
        "type": "imu",
        "accelerometer": [ax, ay, az],   # m/s²
        "gyroscope": [gx, gy, gz],       # rad/s
        "magnetometer": [mx, my, mz]     # µT  (optional, None if absent)
    }

Canonical GNSS event::

    {
        "timestamp": 123456.789,
        "type": "gnss",
        "latitude": float,       # degrees
        "longitude": float,      # degrees
        "speed": float | None,   # m/s
        "accuracy": float,       # meters (default 10.0)
        "altitude": float | None # meters (optional)
    }

Navigation state output::

    {
        "timestamp": ...,
        "mode": "GNSS_AIDED" | "DEAD_RECKONING" | "GNSS_RECOVERY" | ...,
        "latitude": ...,
        "longitude": ...,
        "x": ...,         # East metres (local frame)
        "y": ...,         # North metres (local frame)
        "speed_mps": ...,
        "heading_deg": ...,
        "yaw_axis": ...,
        "alignment_confidence": ...,
        "gnss_available": bool,
        "fusion_state": ...
    }
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# EVENT SCHEMA CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

EVENT_TYPE_IMU = "imu"
EVENT_TYPE_GNSS = "gnss"
EVENT_TYPE_STATE = "state"


# ──────────────────────────────────────────────────────────────────────────────
# CANONICAL EVENT DATACLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IMUEvent:
    """Normalised IMU sensor event."""
    timestamp: float              # seconds
    accelerometer: list           # [ax, ay, az] m/s²
    gyroscope: list               # [gx, gy, gz] rad/s
    magnetometer: list | None = None  # [mx, my, mz] µT

    def to_dict(self) -> dict:
        d = {
            "type": EVENT_TYPE_IMU,
            "timestamp": self.timestamp,
            "accelerometer": list(self.accelerometer),
            "gyroscope": list(self.gyroscope),
        }
        if self.magnetometer is not None:
            d["magnetometer"] = list(self.magnetometer)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "IMUEvent":
        return cls(
            timestamp=float(d["timestamp"]),
            accelerometer=list(d["accelerometer"]),
            gyroscope=list(d["gyroscope"]),
            magnetometer=list(d["magnetometer"]) if d.get("magnetometer") else None,
        )


@dataclass(frozen=True)
class GNSSEvent:
    """Normalised GNSS sensor event."""
    timestamp: float
    latitude: float
    longitude: float
    speed: float | None = None    # m/s
    accuracy: float = 10.0        # metres
    altitude: float | None = None

    def to_dict(self) -> dict:
        return {
            "type": EVENT_TYPE_GNSS,
            "timestamp": self.timestamp,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "speed": self.speed,
            "accuracy": self.accuracy,
            "altitude": self.altitude,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GNSSEvent":
        return cls(
            timestamp=float(d["timestamp"]),
            latitude=float(d["latitude"]),
            longitude=float(d["longitude"]),
            speed=float(d["speed"]) if d.get("speed") is not None else None,
            accuracy=float(d.get("accuracy", 10.0)),
            altitude=float(d["altitude"]) if d.get("altitude") is not None else None,
        )


@dataclass(frozen=True)
class NavigationState:
    """Canonical navigation state output from the engine."""
    timestamp: float
    mode: str                    # GNSS_AIDED | DEAD_RECKONING | GNSS_RECOVERY | ...
    latitude: float | None
    longitude: float | None
    x: float | None              # East metres
    y: float | None              # North metres
    speed_mps: float
    heading_deg: float
    yaw_axis: int | None
    alignment_confidence: float
    gnss_available: bool
    fusion_state: str | None = None
    uncertainty_m: float | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "x": self.x,
            "y": self.y,
            "speed_mps": self.speed_mps,
            "heading_deg": self.heading_deg,
            "yaw_axis": self.yaw_axis,
            "alignment_confidence": self.alignment_confidence,
            "gnss_available": self.gnss_available,
            "fusion_state": self.fusion_state,
            "uncertainty_m": self.uncertainty_m,
        }

    @classmethod
    def from_engine_snapshot(cls, snap: dict, gnss_available: bool = True) -> "NavigationState":
        """Build a NavigationState from a NavigationEngine.state_snapshot() dict."""
        pos = snap.get("position") or {}
        local = snap.get("local_position_m") or {}
        mode = snap.get("mode") or snap.get("gnss_state") or "UNKNOWN"

        # Map gnss_state to readable mode name
        mode_map = {
            "GNSS_AIDED": "GNSS_AIDED",
            "GNSS_INS_FUSED": "GNSS_FUSED",
            "DEAD_RECKONING": "DEAD_RECKONING",
            "REACQUISITION": "GNSS_RECOVERY",
            "GNSS_DEGRADED": "GNSS_DEGRADED",
            "WAITING_FOR_FIX": "WAITING_FOR_FIX",
        }
        mode_display = mode_map.get(mode, mode)

        # gnss_state drives gnss_available override
        gnss_state = snap.get("gnss_state", "")
        if gnss_state in ("INS_DEAD_RECKONING", "GNSS_LOST", "DEAD_RECKONING"):
            gnss_available = False

        heading_raw = snap.get("heading_deg")
        heading_deg = float(heading_raw) if heading_raw is not None else 0.0

        align_raw = snap.get("alignment_confidence")
        align_conf = float(align_raw) if align_raw is not None else 0.0

        return cls(
            timestamp=float(snap.get("timestamp", 0.0)),
            mode=mode_display,
            latitude=float(pos["latitude"]) if pos.get("latitude") is not None else None,
            longitude=float(pos["longitude"]) if pos.get("longitude") is not None else None,
            x=float(local["east"]) if local.get("east") is not None else None,
            y=float(local["north"]) if local.get("north") is not None else None,
            speed_mps=float(snap.get("speed_mps") or 0.0),
            heading_deg=heading_deg,
            yaw_axis=snap.get("yaw_axis"),
            alignment_confidence=align_conf,
            gnss_available=gnss_available,
            fusion_state=snap.get("gnss_state"),
            uncertainty_m=snap.get("uncertainty_m"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# SENSOR DISPATCHER
# ──────────────────────────────────────────────────────────────────────────────

class SensorDispatcher:
    """
    Routes canonical sensor events to a NavigationEngine instance.

    Supports both dict payloads (from WebSocket/HTTP) and typed event objects.
    The dispatcher normalises the canonical schema to the engine's method signatures.
    """

    def __init__(self, engine):
        self._engine = engine

    def dispatch(self, event: dict | IMUEvent | GNSSEvent) -> NavigationState:
        """Dispatch one event and return a NavigationState."""
        if isinstance(event, IMUEvent):
            return self._process_imu_event(event)
        elif isinstance(event, GNSSEvent):
            return self._process_gnss_event(event)
        elif isinstance(event, dict):
            et = event.get("type")
            if et == EVENT_TYPE_IMU:
                return self._process_imu_event(IMUEvent.from_dict(event))
            elif et == EVENT_TYPE_GNSS:
                return self._process_gnss_event(GNSSEvent.from_dict(event))
            else:
                raise ValueError(f"Unknown event type: {et!r}")
        else:
            raise TypeError(f"Unsupported event type: {type(event)}")

    def _process_imu_event(self, event: IMUEvent) -> NavigationState:
        snap = self._engine.process_imu(
            timestamp=event.timestamp,
            accel=event.accelerometer,
            gyro=event.gyroscope,
            mag=event.magnetometer,
        )
        return NavigationState.from_engine_snapshot(snap)

    def _process_gnss_event(self, event: GNSSEvent) -> NavigationState:
        snap = self._engine.process_gnss(
            timestamp=event.timestamp,
            latitude=event.latitude,
            longitude=event.longitude,
            speed_mps=event.speed,
            accuracy_m=event.accuracy,
            altitude_m=event.altitude,
        )
        return NavigationState.from_engine_snapshot(snap, gnss_available=True)

    @property
    def engine(self):
        return self._engine


# ──────────────────────────────────────────────────────────────────────────────
# ANDROID ADAPTER
# ──────────────────────────────────────────────────────────────────────────────

def android_imu_to_event(payload: dict) -> IMUEvent:
    """
    Convert Android SensorManager format to canonical IMUEvent.

    Android format (from IDR Android app):
        {
            "timestamp": <nanoseconds or seconds>,
            "type": "imu",
            "ax": ..., "ay": ..., "az": ...,
            "gx": ..., "gy": ..., "gz": ...,
            "mx": ..., "my": ..., "mz": ...  # optional
        }
    """
    ts = float(payload["timestamp"])
    # Detect nanoseconds (Android typical): > 1e12
    if ts > 1e12:
        ts = ts / 1e9

    mag = None
    if payload.get("mx") is not None:
        mag = [float(payload["mx"]), float(payload["my"]), float(payload["mz"])]

    return IMUEvent(
        timestamp=ts,
        accelerometer=[float(payload["ax"]), float(payload["ay"]), float(payload["az"])],
        gyroscope=[float(payload["gx"]), float(payload["gy"]), float(payload["gz"])],
        magnetometer=mag,
    )


def android_gnss_to_event(payload: dict) -> GNSSEvent:
    """
    Convert Android Location format to canonical GNSSEvent.

    Android format (from IDR Android app):
        {
            "timestamp": <seconds>,
            "type": "gnss",
            "latitude": ...,
            "longitude": ...,
            "speed": ...,       # m/s
            "accuracy": ...,    # metres
            "altitude": ...
        }
    """
    return GNSSEvent.from_dict(payload)


def normalize_sensor_event(payload: dict) -> IMUEvent | GNSSEvent:
    """
    Auto-detect and normalize any supported sensor event format.

    Supports:
    - Canonical schema (type = 'imu' or 'gnss', accelerometer/gyroscope keys)
    - Android app schema (type = 'imu', ax/ay/az/gx/gy/gz keys)
    """
    et = payload.get("type", "")
    if et == EVENT_TYPE_IMU:
        if "accelerometer" in payload:
            return IMUEvent.from_dict(payload)
        else:
            # Android style
            return android_imu_to_event(payload)
    elif et == EVENT_TYPE_GNSS:
        return GNSSEvent.from_dict(payload)
    else:
        raise ValueError(f"Cannot normalize event with type={et!r}")
