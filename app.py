from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import math
import os
import logging
import sys
import time
from collections import deque
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from navigation_engine import NavigationEngine


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("IDR_SERVER")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

WEB_DIR = BASE_DIR / "web"
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "models"

DEFAULT_MAP = DATA_DIR / "roads.geojson"

DEFAULT_MODEL_JOBLIB = MODELS_DIR / "speed_model.joblib"
DEFAULT_MODEL_PKL = MODELS_DIR / "speed_model.pkl"


# ============================================================
# MAP PATH
# ============================================================

environment_map = os.getenv("IDR_ROADS_GEOJSON")

if environment_map:
    MAP_PATH = environment_map

elif DEFAULT_MAP.exists():
    MAP_PATH = str(DEFAULT_MAP)

else:
    MAP_PATH = None


# ============================================================
# MODEL PATH
# ============================================================

environment_model = os.getenv("IDR_SPEED_MODEL")

if environment_model:
    MODEL_PATH = environment_model

elif DEFAULT_MODEL_JOBLIB.exists():
    MODEL_PATH = str(DEFAULT_MODEL_JOBLIB)

elif DEFAULT_MODEL_PKL.exists():
    MODEL_PATH = str(DEFAULT_MODEL_PKL)

else:
    MODEL_PATH = None


# ============================================================
# NAVIGATION ENGINE
# ============================================================

engine = NavigationEngine(
    map_path=MAP_PATH,
    model_path=MODEL_PATH,
)

engine_lock = RLock()


# ============================================================
# LIVE TELEMETRY — connection tracking, event rates, blackout gate
# ============================================================

class _LiveTelemetry:
    """Lightweight telemetry for the live phone pipeline."""

    MAX_RATE_WINDOW = 30  # samples to keep for rolling rate calculation

    def __init__(self):
        self._lock = RLock()
        # Connection tracking
        self.ws_clients: int = 0
        self.last_ws_connect_time: float | None = None
        self.last_ws_disconnect_time: float | None = None
        # Event tracking
        self._imu_times: deque = deque(maxlen=self.MAX_RATE_WINDOW)
        self._gnss_times: deque = deque(maxlen=self.MAX_RATE_WINDOW)
        self.total_imu: int = 0
        self.total_gnss: int = 0
        self.total_gnss_suppressed: int = 0
        self.total_errors: int = 0
        self.last_imu_ts: float | None = None
        self.last_gnss_ts: float | None = None
        # GNSS blackout gate
        self.gnss_blackout: bool = False
        self.blackout_start_wall: float | None = None
        # Session logging
        self._session_file: Any | None = None
        self._session_path: Path | None = None

    def client_connect(self):
        with self._lock:
            self.ws_clients += 1
            self.last_ws_connect_time = time.time()

    def client_disconnect(self):
        with self._lock:
            self.ws_clients = max(0, self.ws_clients - 1)
            self.last_ws_disconnect_time = time.time()

    def record_imu(self, ts: float | None = None):
        with self._lock:
            now = time.monotonic()
            self._imu_times.append(now)
            self.total_imu += 1
            if ts is not None:
                self.last_imu_ts = ts

    def record_gnss(self, ts: float | None = None):
        with self._lock:
            now = time.monotonic()
            self._gnss_times.append(now)
            self.total_gnss += 1
            if ts is not None:
                self.last_gnss_ts = ts

    def record_gnss_suppressed(self):
        with self._lock:
            self.total_gnss_suppressed += 1

    def record_error(self):
        with self._lock:
            self.total_errors += 1

    def _rolling_rate(self, times: deque) -> float:
        """Events per second over the window."""
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

    def start_blackout(self):
        with self._lock:
            self.gnss_blackout = True
            self.blackout_start_wall = time.time()

    def stop_blackout(self):
        with self._lock:
            self.gnss_blackout = False

    def is_gnss_blocked(self) -> bool:
        with self._lock:
            return self.gnss_blackout

    def start_session_log(self, path: str = "Data/live_phone_session.jsonl"):
        with self._lock:
            if self._session_file is not None:
                try:
                    self._session_file.close()
                except Exception:
                    pass
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._session_path = p
            self._session_file = p.open("a", encoding="utf-8", buffering=1)

    def log_event(self, record: dict):
        with self._lock:
            if self._session_file is not None:
                try:
                    self._session_file.write(json.dumps(record, default=str) + "\n")
                except Exception:
                    pass

    def stop_session_log(self):
        with self._lock:
            if self._session_file is not None:
                try:
                    self._session_file.close()
                except Exception:
                    pass
                self._session_file = None

    def snapshot(self) -> dict:
        with self._lock:
            imu_rate = self._rolling_rate(self._imu_times)
            gnss_rate = self._rolling_rate(self._gnss_times)
            blackout_duration = None
            if self.gnss_blackout and self.blackout_start_wall is not None:
                blackout_duration = round(time.time() - self.blackout_start_wall, 1)
            return {
                "ws_clients_connected": self.ws_clients,
                "last_ws_connect_time": self.last_ws_connect_time,
                "last_ws_disconnect_time": self.last_ws_disconnect_time,
                "imu_rate_hz": round(imu_rate, 1),
                "gnss_rate_hz": round(gnss_rate, 2),
                "total_imu_events": self.total_imu,
                "total_gnss_events": self.total_gnss,
                "total_gnss_suppressed": self.total_gnss_suppressed,
                "total_errors": self.total_errors,
                "last_imu_timestamp": self.last_imu_ts,
                "last_gnss_timestamp": self.last_gnss_ts,
                "gnss_blackout_active": self.gnss_blackout,
                "gnss_blackout_duration_s": blackout_duration,
                "session_log_active": self._session_file is not None,
                "session_log_path": str(self._session_path) if self._session_path else None,
            }


live_telemetry = _LiveTelemetry()
# Auto-start session logging on server boot
live_telemetry.start_session_log()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

def _display_configured_path(path: str | None) -> str:
    """Return a useful startup path without exposing external host paths."""
    if not path:
        return "not configured"

    try:
        return str(Path(path).resolve().relative_to(BASE_DIR))
    except ValueError:
        return "external path configured"


@asynccontextmanager
async def lifespan(_application: FastAPI):
    """Emit a compact, truthful readiness report once Uvicorn starts."""
    runtime = engine.runtime_snapshot()

    logger.info("IDR backend startup: status=READY version=%s", _application.version)
    logger.info(
        "Navigation engine initialized: fusion=%s gnss_timeout_s=%.2f filter=%s",
        runtime["fusion_method"],
        runtime["gnss_timeout_s"],
        runtime["filter_mode"],
    )
    logger.info(
        "AI model: status=%s kind=%s path=%s",
        runtime["ai_model_status"],
        runtime["ai_model_kind"] or "none",
        _display_configured_path(MODEL_PATH),
    )
    logger.info(
        "Offline map: status=%s path=%s road_constraints=%s",
        runtime["map_status"],
        _display_configured_path(MAP_PATH),
        runtime["nhc_status"],
    )
    logger.info(
        "Runtime configuration: nhc=%s map_matching=%s web_ui=%s python=%s",
        runtime["nhc_enabled"],
        runtime["map_matching_enabled"],
        "available" if WEB_DIR.exists() else "missing",
        sys.version.split()[0],
    )

    if engine.ai_model_error:
        logger.warning("AI model is unavailable: %s", engine.ai_model_error)
    if engine.map_error:
        logger.warning("Offline map is unavailable: %s", engine.map_error)

    yield
    logger.info("IDR backend shutdown complete")

app = FastAPI(
    title="Intelligent Dead Reckoning API",
    version="2.0.0-live",
    description=(
        "Live Android GNSS + IMU Intelligent Dead Reckoning "
        "Navigation System"
    ),
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATA MODELS
# ============================================================

class IMUPayload(BaseModel):

    # Timestamp in seconds
    timestamp: float

    # Accelerometer
    ax: float
    ay: float
    az: float

    # Gyroscope
    gx: float
    gy: float
    gz: float

    # Magnetometer
    mx: float | None = None
    my: float | None = None
    mz: float | None = None

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("timestamp must be finite")
        return value


class GNSSPayload(BaseModel):

    # Timestamp in seconds
    timestamp: float

    latitude: float
    longitude: float

    # Android Location.speed
    # Unit = meters per second
    speed: float | None = None

    # Android Location.accuracy
    # Unit = meters
    accuracy: float = 10.0

    altitude: float | None = None

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("timestamp must be finite")
        return value


class AlignmentPayload(BaseModel):

    forward_phone: list[float] = Field(
        min_length=3,
        max_length=3,
    )

    up_phone: list[float] = Field(
        min_length=3,
        max_length=3,
    )


class RuntimeConfigPayload(BaseModel):
    """The deliberately small set of controls safe to change during a run."""

    model_config = ConfigDict(extra="forbid", strict=True)

    filter_mode: str | None = None
    nhc_enabled: bool | None = None
    map_matching_enabled: bool | None = None
    gnss_timeout_s: float | None = Field(default=None, ge=0.1, le=10.0)
    profile: str | None = None

    @field_validator("filter_mode")
    @classmethod
    def filter_mode_must_be_supported(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip().lower()
        if normalized not in {"raw", "balanced", "strict"}:
            raise ValueError("filter_mode must be one of: raw, balanced, strict")
        return normalized

    @field_validator("profile")
    @classmethod
    def profile_must_be_supported(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip().lower()
        if normalized not in {"phone", "edge"}:
            raise ValueError("profile must be one of: phone, edge")
        return normalized

    @model_validator(mode="after")
    def require_a_setting(self) -> "RuntimeConfigPayload":
        if not self.model_fields_set or not any(
            getattr(self, field) is not None for field in self.model_fields_set
        ):
            raise ValueError("provide at least one runtime configuration setting")
        return self


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def api_error(
    status_code: int,
    message: str,
) -> HTTPException:

    return HTTPException(
        status_code=status_code,
        detail=message,
    )


def get_page(filename: str):

    file_path = WEB_DIR / filename

    if not file_path.exists():

        raise HTTPException(
            status_code=404,
            detail=f"Web page not found: {filename}",
        )

    return FileResponse(file_path)


# ============================================================
# ROOT / FRONTEND
# ============================================================

@app.get("/", include_in_schema=False)
def home():

    return get_page("livenavigation.html")


@app.get("/navigate", include_in_schema=False)
def navigate():

    return get_page("livenavigation.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard():

    return get_page("index.html")


@app.get("/outage", include_in_schema=False)
def outage():

    return get_page("gnssoutage.html")


@app.get("/calibration", include_in_schema=False)
@app.get("/calibrate", include_in_schema=False)
def calibration():

    return get_page("sensorcaliberation.html")


@app.get("/pipeline", include_in_schema=False)
def pipeline():

    return get_page("fusionpipeline.html")


@app.get("/settings", include_in_schema=False)
def settings():

    return get_page("setting.html")


@app.get("/mvp", include_in_schema=False)
def mvp_demo():
    return get_page("mvp_demo.html")


# ============================================================
# API INFORMATION
# ============================================================

@app.get("/api")
def api_info() -> dict[str, Any]:

    return {

        "service": "IDR Live Navigation Gateway",

        "version": app.version,

        "server_status": "RUNNING",

        "pages": {

            "home": "/",

            "navigate": "/navigate",

            "dashboard": "/dashboard",

            "outage": "/outage",

            "calibration": "/calibration",

            "pipeline": "/pipeline",

            "settings": "/settings",
        },

        "endpoints": {

            "health": "/health",

            "imu": "/sensor/imu",

            "gnss": "/sensor/gnss",

            "state": "/navigation/state",

            "position": "/api/position",

            "telemetry": "/api/telemetry",

            "reset": "/api/reset",

            "alignment": "/config/alignment",

            "config": "/config",

            "websocket": "/ws/sensor",
            "websocket_replay": "/ws/replay",
            "live_telemetry": "/api/live_telemetry",
            "blackout_start": "/api/blackout/start",
            "blackout_stop": "/api/blackout/stop",
            "blackout_status": "/api/blackout/status",
            "session_start": "/api/session/start",
            "session_stop": "/api/session/stop",
            "mvp_dashboard": "/mvp",
        },

        "configuration": engine.runtime_snapshot(),
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    with engine_lock:

        return {

            "ok": True,

            "service": "Intelligent Dead Reckoning",

            "version": app.version,

            "engine_status": "READY",

            "map_status": engine.map_status,

            "map_error": engine.map_error,

            "ai_model_status": engine.ai_model_status,

            "accepted_imu": engine.accepted_imu,

            "accepted_gnss": engine.accepted_gnss,

            "rejected_samples": engine.rejected_samples,

            "web_directory_exists": WEB_DIR.exists(),

            "map_configured": MAP_PATH is not None,

            "model_configured": MODEL_PATH is not None,

            "runtime_configuration": engine.runtime_snapshot(),
        }


# ============================================================
# LIVE NAVIGATION STATE
# ============================================================

@app.get("/navigation/state")
def navigation_state():

    with engine_lock:

        return engine.state_snapshot()


# ============================================================
# RUNTIME CONFIGURATION
# ============================================================

@app.get("/config")
def get_runtime_configuration():
    """Return only safe, effective runtime controls and their live status."""
    with engine_lock:
        return {
            "ok": True,
            "configuration": engine.runtime_snapshot(),
            "status": {
                "mode": engine.state_snapshot()["mode"],
                "gnss_status": engine.state_snapshot()["gnss_status"],
                "map_status": engine.map_status,
                "nhc_status": engine.nhc_status,
            },
        }


@app.patch("/config")
def update_runtime_configuration(payload: RuntimeConfigPayload):
    """Apply validated settings directly to the running navigation engine."""
    try:
        with engine_lock:
            configuration = engine.set_runtime_options(
                filter_mode=payload.filter_mode,
                nhc_enabled=payload.nhc_enabled,
                map_matching_enabled=payload.map_matching_enabled,
                gnss_timeout_s=payload.gnss_timeout_s,
                profile=payload.profile,
            )
            state = engine.state_snapshot()
        return {
            "ok": True,
            "configuration": configuration,
            "status": {
                "mode": state["mode"],
                "gnss_status": state["gnss_status"],
                "map_status": state["map_status"],
                "nhc_status": state["nhc_status"],
            },
        }
    except ValueError as exc:
        raise api_error(422, str(exc)) from exc


# ============================================================
# COMPATIBILITY ENDPOINTS
# ============================================================

@app.get("/api/position")
def api_position():

    with engine_lock:

        return engine.state_snapshot()


@app.get("/api/telemetry")
def api_telemetry():

    with engine_lock:

        return engine.state_snapshot()


# ============================================================
# REAL IMU DATA
# ============================================================

@app.post("/sensor/imu")
def sensor_imu(payload: IMUPayload):

    try:

        mag = None

        if (
            payload.mx is not None
            and payload.my is not None
            and payload.mz is not None
        ):

            mag = [

                payload.mx,
                payload.my,
                payload.mz,
            ]

        with engine_lock:

            result = engine.process_imu(

                timestamp=float(payload.timestamp),

                accel=[

                    float(payload.ax),

                    float(payload.ay),

                    float(payload.az),
                ],

                gyro=[

                    float(payload.gx),

                    float(payload.gy),

                    float(payload.gz),
                ],

                mag=mag,
            )

        live_telemetry.record_imu(float(payload.timestamp))
        live_telemetry.log_event({
            "event": "imu",
            "wall_time": time.time(),
            "timestamp": float(payload.timestamp),
            "accel": [float(payload.ax), float(payload.ay), float(payload.az)],
            "gyro": [float(payload.gx), float(payload.gy), float(payload.gz)],
            "nav_mode": result.get("mode"),
            "gnss_state": result.get("gnss_state"),
            "position": result.get("position"),
            "speed_mps": result.get("speed_mps"),
            "heading_deg": result.get("heading_deg"),
        })
        return result

    except ValueError as exc:

        logger.warning(
            "IMU rejected: %s",
            exc,
        )
        live_telemetry.record_error()

        raise api_error(
            422,
            str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "IMU processing failed"
        )
        live_telemetry.record_error()

        raise api_error(
            500,
            f"IMU processing failed: {exc}",
        ) from exc


# ============================================================
# REAL GNSS DATA
# ============================================================

@app.post("/sensor/gnss")
def sensor_gnss(payload: GNSSPayload):

    try:
        # ── RUNTIME GNSS BLACKOUT GATE ─────────────────────────────────────
        # When gnss_blackout is active, GNSS measurements are suppressed from
        # the navigation engine. IMU continues, engine transitions to DR.
        if live_telemetry.is_gnss_blocked():
            live_telemetry.record_gnss_suppressed()
            live_telemetry.log_event({
                "event": "gnss_suppressed",
                "wall_time": time.time(),
                "timestamp": float(payload.timestamp),
                "latitude": float(payload.latitude),
                "longitude": float(payload.longitude),
            })
            # Return current engine state so caller can see DR mode
            with engine_lock:
                return engine.state_snapshot()

        with engine_lock:

            result = engine.process_gnss(

                timestamp=float(payload.timestamp),

                latitude=float(payload.latitude),

                longitude=float(payload.longitude),

                speed_mps=(
                    None
                    if payload.speed is None
                    else float(payload.speed)
                ),

                accuracy_m=float(payload.accuracy),

                altitude_m=(
                    None
                    if payload.altitude is None
                    else float(payload.altitude)
                ),
            )

        live_telemetry.record_gnss(float(payload.timestamp))
        live_telemetry.log_event({
            "event": "gnss",
            "wall_time": time.time(),
            "timestamp": float(payload.timestamp),
            "latitude": float(payload.latitude),
            "longitude": float(payload.longitude),
            "speed_mps": float(payload.speed) if payload.speed is not None else None,
            "accuracy_m": float(payload.accuracy),
            "nav_mode": result.get("mode"),
            "gnss_state": result.get("gnss_state"),
        })
        return result

    except ValueError as exc:

        logger.warning(
            "GNSS rejected: %s",
            exc,
        )
        live_telemetry.record_error()

        raise api_error(
            422,
            str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "GNSS processing failed"
        )
        live_telemetry.record_error()

        raise api_error(
            500,
            f"GNSS processing failed: {exc}",
        ) from exc


# ============================================================
# GENERIC SENSOR ENDPOINT
# ============================================================

@app.post("/sensor")
def sensor_event(payload: dict[str, Any]):

    try:

        event_type = payload.get("type")

        if event_type == "imu":

            imu_payload = (
                IMUPayload.model_validate(payload)
            )

            return sensor_imu(imu_payload)

        if event_type == "gnss":

            gnss_payload = (
                GNSSPayload.model_validate(payload)
            )

            return sensor_gnss(gnss_payload)

        raise api_error(
            422,
            "type must be 'imu' or 'gnss'",
        )

    except ValidationError as exc:

        raise HTTPException(
            status_code=422,
            detail=exc.errors(),
        ) from exc


# ============================================================
# PHONE / VEHICLE ALIGNMENT
# ============================================================

@app.post("/config/alignment")
def configure_alignment(
    payload: AlignmentPayload,
):

    try:

        with engine_lock:

            engine.apply_manual_alignment(payload.forward_phone, payload.up_phone)

            return {

                "ok": True,

                "mounting_calibrated":
                    engine.orientation.vehicle_calibrated,

                "forward_phone":
                    payload.forward_phone,

                "up_phone":
                    payload.up_phone,

                "mounting_status": engine.mounting_status,
            }

    except Exception as exc:

        logger.exception(
            "Alignment configuration failed"
        )

        raise api_error(
            422,
            str(exc),
        ) from exc


# ============================================================
# RESET
# ============================================================

@app.post("/demo/reset")
def demo_reset():

    with engine_lock:

        engine.reset()

        logger.info(
            "Navigation engine reset"
        )

        return {

            "ok": True,

            "message":
                "Navigation engine reset successfully",

            "state":
                engine.state_snapshot(),
        }


@app.post("/api/reset")
def api_reset():

    return demo_reset()


# ============================================================
# LIVE TELEMETRY & GNSS BLACKOUT CONTROL ENDPOINTS
# ============================================================

@app.get("/api/live_telemetry")
def api_live_telemetry():
    """Return live connection stats, IMU/GNSS rates, and blackout state."""
    return {
        "ok": True,
        "telemetry": live_telemetry.snapshot(),
        "navigation": engine.state_snapshot(),
    }


class BlackoutPayload(BaseModel):
    duration_s: float | None = Field(default=None, ge=0.1, le=600.0,
                                     description="Auto-stop after this many seconds (optional)")


@app.post("/api/blackout/start")
def api_blackout_start(payload: BlackoutPayload = BlackoutPayload()):
    """
    Start a runtime GNSS blackout.

    GNSS measurements are suppressed from the navigation engine.
    The engine will transition to DEAD_RECKONING after gnss_timeout_s.
    IMU continues unaffected.
    """
    live_telemetry.start_blackout()
    logger.info("GNSS blackout started via REST API")
    snap = engine.state_snapshot()
    return {
        "ok": True,
        "blackout_active": True,
        "message": "GNSS blackout started — engine will transition to DEAD_RECKONING",
        "current_mode": snap.get("mode"),
        "gnss_timeout_s": snap.get("gnss_timeout_s"),
    }


@app.post("/api/blackout/stop")
def api_blackout_stop():
    """
    Stop the runtime GNSS blackout.

    GNSS measurements resume reaching the navigation engine.
    The engine will transition REACQUISITION → GNSS_INS_FUSED.
    """
    live_telemetry.stop_blackout()
    logger.info("GNSS blackout stopped via REST API")
    snap = engine.state_snapshot()
    return {
        "ok": True,
        "blackout_active": False,
        "message": "GNSS blackout stopped — engine will enter REACQUISITION",
        "current_mode": snap.get("mode"),
    }


@app.get("/api/blackout/status")
def api_blackout_status():
    """Return current blackout state and suppression counters."""
    tel = live_telemetry.snapshot()
    return {
        "ok": True,
        "blackout_active": tel["gnss_blackout_active"],
        "blackout_duration_s": tel["gnss_blackout_duration_s"],
        "total_gnss_suppressed": tel["total_gnss_suppressed"],
    }


@app.post("/api/session/start")
def api_session_start(path: str = "Data/live_phone_session.jsonl"):
    """Start or restart session logging to the given path."""
    live_telemetry.start_session_log(path)
    return {"ok": True, "session_log_path": path}


@app.post("/api/session/stop")
def api_session_stop():
    """Stop session logging and flush the file."""
    live_telemetry.stop_session_log()
    return {"ok": True, "message": "Session log stopped"}


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws/sensor")
async def sensor_websocket(
    websocket: WebSocket,
):

    await websocket.accept()
    live_telemetry.client_connect()

    logger.info(
        "WebSocket client connected (total=%d)",
        live_telemetry.ws_clients,
    )

    try:

        while True:

            payload = (
                await websocket.receive_json()
            )

            try:

                event_type = payload.get("type")

                if event_type == "imu":

                    imu_payload = (
                        IMUPayload.model_validate(payload)
                    )

                    result = sensor_imu(
                        imu_payload
                    )

                elif event_type == "gnss":

                    gnss_payload = (
                        GNSSPayload.model_validate(payload)
                    )

                    result = sensor_gnss(
                        gnss_payload
                    )

                elif event_type == "state":

                    with engine_lock:

                        result = (
                            engine.state_snapshot()
                        )

                elif event_type == "blackout_start":
                    live_telemetry.start_blackout()
                    logger.info("GNSS blackout started via WebSocket")
                    result = {"blackout": True, "message": "GNSS blackout activated"}

                elif event_type == "blackout_stop":
                    live_telemetry.stop_blackout()
                    logger.info("GNSS blackout stopped via WebSocket")
                    result = {"blackout": False, "message": "GNSS blackout deactivated"}

                elif event_type == "telemetry":
                    result = live_telemetry.snapshot()

                else:

                    raise ValueError(
                        "type must be 'imu', 'gnss', 'state', "
                        "'blackout_start', 'blackout_stop', or 'telemetry'"
                    )

                await websocket.send_json({

                    "ok": True,

                    "data": result,
                })

            except Exception as exc:

                logger.warning(
                    "WebSocket processing error: %s",
                    exc,
                )
                live_telemetry.record_error()

                await websocket.send_json({

                    "ok": False,

                    "error": str(exc),
                })

    except WebSocketDisconnect:

        live_telemetry.client_disconnect()
        logger.info(
            "WebSocket client disconnected (remaining=%d)",
            live_telemetry.ws_clients,
        )


# ============================================================
# MVP REPLAY STREAMING WEBSOCKET
# ============================================================

class ReplayConfig(BaseModel):
    dataset: str = "Data/S-S1.csv"
    outage_start: float = 120.0
    outage_duration: float = 30.0
    speedup: float = 10.0
    max_rows: int | None = None
    gnss_speed_unit: str = "mps"


@app.post("/demo/replay/start")
def demo_replay_start(config: ReplayConfig):
    """Start a background replay and return a summary. Non-streaming."""
    from mvp_realtime_replay import run_realtime

    if not Path(config.dataset).exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {config.dataset}")

    summary = run_realtime(
        input_path=config.dataset,
        outage_start=config.outage_start,
        outage_duration=config.outage_duration,
        output_path="Data/mvp_demo_trajectory.jsonl",
        max_rows=config.max_rows,
        gnss_speed_unit=config.gnss_speed_unit,
        speedup=config.speedup,
    )
    return {"ok": True, "summary": summary}


@app.websocket("/ws/replay")
async def replay_websocket(websocket: WebSocket):
    """
    Stream replay navigation states in real time over WebSocket.

    Client sends a JSON config on connect:
        {"dataset": "Data/S-S1.csv", "outage_start": 120, "outage_duration": 30,
         "speedup": 5, "max_rows": null, "gnss_speed_unit": "mps"}

    Server streams JSON navigation state frames at replay speed.
    Final frame includes {"event": "done", "summary": {...}}.
    """
    import pandas as pd
    import numpy as np
    from navigation_engine import NavigationEngine as _NE

    await websocket.accept()
    logger.info("Replay WebSocket connected")

    try:
        raw = await websocket.receive_text()
        cfg = json.loads(raw)
        dataset = cfg.get("dataset", "Data/S-S1.csv")
        outage_start = float(cfg.get("outage_start", 120.0))
        outage_duration = float(cfg.get("outage_duration", 30.0))
        speedup = float(cfg.get("speedup", 10.0))
        max_rows = cfg.get("max_rows")
        gnss_speed_unit = cfg.get("gnss_speed_unit", "mps")

        if not Path(dataset).exists():
            await websocket.send_json({"event": "error", "message": f"Dataset not found: {dataset}"})
            return

        df = pd.read_csv(dataset, encoding="cp1252", engine="python")

        def fc(cols, *words):
            lc = [c.lower() for c in cols]
            for col, l in zip(cols, lc):
                if all(w.lower() in l for w in words):
                    return col
            return None

        time_col = fc(df.columns, "time")
        acc_cols = [fc(df.columns, "accelerometer", a) for a in "xyz"]
        gyro_cols = [fc(df.columns, "gyroscope", a) for a in ("yaw", "pitch", "roll")]
        if any(c is None for c in gyro_cols):
            gyro_cols = [fc(df.columns, "gyroscope", a) for a in "xyz"]
        mag_cols  = [fc(df.columns, "magnetic", a) for a in "xyz"]
        lat_col   = fc(df.columns, "gps", "latitude")
        lon_col   = fc(df.columns, "gps", "longitude")
        speed_col = fc(df.columns, "gps", "speed")
        acc_col   = fc(df.columns, "gps", "accuracy")

        raw_times = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
        timestamps = (raw_times - raw_times[0]) / 1000.0

        limit = len(df) if max_rows is None else min(len(df), int(max_rows))

        replay_clock: list[float] = [0.0]
        eng = _NE(
            map_path=os.getenv("IDR_ROADS_GEOJSON"),
            model_path=os.getenv("IDR_SPEED_MODEL"),
            _clock=lambda: replay_clock[0],
        )

        sim_start: float | None = None
        wall_ref: float = asyncio.get_event_loop().time()
        errors: list[float] = []
        outage_dist_m: float = 0.0
        prev_gt = None
        truth_origin = None
        R = 6_378_137.0

        for index in range(limit):
            row = df.iloc[index]
            timestamp = float(timestamps[index])
            replay_clock[0] = timestamp

            # Real-time pacing
            if speedup > 0 and index > 0:
                if sim_start is None:
                    sim_start = timestamp
                    wall_ref = asyncio.get_event_loop().time()
                else:
                    sim_elapsed = timestamp - sim_start
                    wall_elapsed = asyncio.get_event_loop().time() - wall_ref
                    sleep_s = sim_elapsed / speedup - wall_elapsed
                    if sleep_s > 0.001:
                        await asyncio.sleep(sleep_s)

            lat = float(pd.to_numeric(row[lat_col], errors="coerce"))
            lon = float(pd.to_numeric(row[lon_col], errors="coerce"))

            if truth_origin is None:
                truth_origin = (lat, lon)

            in_outage = outage_start <= timestamp < outage_start + outage_duration

            if not in_outage:
                spd = None
                if speed_col:
                    spd = float(pd.to_numeric(row[speed_col], errors="coerce"))
                    if gnss_speed_unit == "kmh":
                        spd /= 3.6
                acc_v = 10.0
                if acc_col:
                    cand = float(pd.to_numeric(row[acc_col], errors="coerce"))
                    if math.isfinite(cand) and cand > 0:
                        acc_v = cand
                eng.process_gnss(timestamp=timestamp, latitude=lat, longitude=lon,
                                  speed_mps=spd, accuracy_m=acc_v)

            accel = [float(pd.to_numeric(row[c], errors="coerce")) for c in acc_cols]
            gyro  = [float(pd.to_numeric(row[c], errors="coerce")) for c in gyro_cols]
            mag   = None
            if all(c is not None for c in mag_cols):
                mag = [float(pd.to_numeric(row[c], errors="coerce")) for c in mag_cols]

            state = eng.process_imu(timestamp=timestamp, accel=accel, gyro=gyro, mag=mag)
            state["simulated_gnss_available"] = not in_outage

            # Reference truth
            truth_xy = [
                math.radians(lon - truth_origin[1]) * R * math.cos(math.radians(truth_origin[0])),
                math.radians(lat - truth_origin[0]) * R,
            ]
            state["reference_east_m"] = truth_xy[0]
            state["reference_north_m"] = truth_xy[1]

            if state.get("position") is not None:
                local = state["local_position_m"]
                err = math.hypot(local["east"] - truth_xy[0], local["north"] - truth_xy[1])
                state["position_error_m"] = err
                if in_outage:
                    errors.append(err)

            # Outage ground truth distance
            if in_outage:
                if prev_gt is not None:
                    dlat = math.radians(lat - prev_gt[0])
                    dlon = math.radians(lon - prev_gt[1])
                    a = math.sin(dlat/2)**2 + math.cos(math.radians(prev_gt[0]))*math.cos(math.radians(lat))*math.sin(dlon/2)**2
                    outage_dist_m += R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                prev_gt = (lat, lon)

            # Stream to client
            frame = {
                "event": "nav",
                "timestamp": timestamp,
                "mode": state.get("mode") or state.get("gnss_state"),
                "gnss_available": not in_outage,
                "position": state.get("position"),
                "local_position_m": state.get("local_position_m"),
                "reference_east_m": state.get("reference_east_m"),
                "reference_north_m": state.get("reference_north_m"),
                "position_error_m": state.get("position_error_m"),
                "speed_mps": state.get("speed_mps"),
                "heading_deg": state.get("heading_deg"),
                "gnss_state": state.get("gnss_state"),
                "uncertainty_m": state.get("uncertainty_m"),
                "speed_source": state.get("speed_source"),
                "yaw_axis": state.get("yaw_axis"),
                "alignment_confidence": state.get("alignment_confidence"),
            }
            await websocket.send_json(frame)

        # Final summary
        final_err = float(errors[-1]) if errors else None
        drift = (final_err / max(outage_dist_m, 1e-6) * 100.0) if final_err else None
        await websocket.send_json({
            "event": "done",
            "summary": {
                "samples_replayed": limit,
                "outage_drift_pct": drift,
                "outage_final_error_m": final_err,
                "outage_mae_m": float(sum(errors)/len(errors)) if errors else None,
                "outage_ground_truth_distance_m": outage_dist_m,
            }
        })

    except WebSocketDisconnect:
        logger.info("Replay WebSocket disconnected")
    except Exception as exc:
        logger.exception("Replay WebSocket error: %s", exc)
        try:
            await websocket.send_json({"event": "error", "message": str(exc)})
        except Exception:
            pass


# ============================================================
# STATIC FILES
#
# IMPORTANT:
# Keep this LAST.
# ============================================================

if WEB_DIR.exists():

    app.mount(

        "/",

        StaticFiles(
            directory=str(WEB_DIR),
            html=False,
        ),

        name="web",
    )

else:

    logger.warning(
        "WEB directory not found: %s",
        WEB_DIR,
    )
