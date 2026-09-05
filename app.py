"""FastAPI LAN gateway for the live IDR navigation engine.

Start from this directory with:

    python -m uvicorn app:app --host 0.0.0.0 --port 8000

Android should use the laptop's LAN address, for example
``http://192.168.1.42:8000``; localhost on the phone is the phone itself.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

from navigation_engine import NavigationEngine


DEFAULT_MAP = Path(__file__).resolve().parent / "Data" / "roads.geojson"
MAP_PATH = os.getenv("IDR_ROADS_GEOJSON") or (
    str(DEFAULT_MAP) if DEFAULT_MAP.exists() else None
)
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "speed_model.joblib"
MODEL_PATH = os.getenv("IDR_SPEED_MODEL") or (
    str(DEFAULT_MODEL) if DEFAULT_MODEL.exists() else None
)
engine = NavigationEngine(map_path=MAP_PATH, model_path=MODEL_PATH)
engine_lock = RLock()
WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(
    title="Intelligent Dead Reckoning API",
    version="1.0.0-live",
    description="LAN sensor gateway and continuous GNSS/INS navigation state.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IMUPayload(BaseModel):
    timestamp: float = Field(description="Monotonic seconds from the sensor clock")
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    mx: float | None = None
    my: float | None = None
    mz: float | None = None


class GNSSPayload(BaseModel):
    timestamp: float = Field(description="Monotonic seconds from the sensor clock")
    latitude: float
    longitude: float
    speed: float | None = Field(default=None, description="m/s")
    accuracy: float = Field(default=10.0, description="horizontal accuracy in metres")
    altitude: float | None = None


class AlignmentPayload(BaseModel):
    forward_phone: list[float] = Field(min_length=3, max_length=3)
    up_phone: list[float] = Field(min_length=3, max_length=3)


def _error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


@app.get("/")
def root() -> FileResponse:
    """Open the presentation-ready dashboard at the base server URL."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api")
def api_info() -> dict[str, Any]:
    return {
        "service": "IDR live navigation gateway",
        "dashboard": "/dashboard",
        "health": "/health",
        "sensor_http": ["/sensor/imu", "/sensor/gnss"],
        "sensor_websocket": "/ws/sensor",
        "navigation_state": "/navigation/state",
    }


@app.get("/dashboard", include_in_schema=False)
def dashboard() -> FileResponse:
    """Self-contained browser demo dashboard served by the same LAN API."""
    return FileResponse(WEB_DIR / "index.html")

@app.get("/navigate", include_in_schema=False)
def navigate() -> FileResponse:
    """Stitch Navigate screen."""
    return FileResponse(WEB_DIR / "livenavigation.html")

@app.get("/outage", include_in_schema=False)
def outage() -> FileResponse:
    """Stitch Outage Mode screen."""
    return FileResponse(WEB_DIR / "gnssoutage.html")

@app.get("/calibration", include_in_schema=False)
def calibration() -> FileResponse:
    """Stitch Calibration screen."""
    return FileResponse(WEB_DIR / "sensorcaliberation.html")

@app.get("/pipeline", include_in_schema=False)
def pipeline() -> FileResponse:
    """Stitch Fusion Pipeline screen."""
    return FileResponse(WEB_DIR / "fusionpipeline.html")

@app.get("/settings", include_in_schema=False)
def settings() -> FileResponse:
    return FileResponse(WEB_DIR / "setting.html")

@app.get("/health")
def health() -> dict[str, Any]:
    with engine_lock:
        return {
            "ok": True,
            "service": "idr",
            "api_version": app.version,
            "engine": "ready",
            "map_status": engine.map_status,
            "map_error": engine.map_error,
            "speed_model": (
                "AVAILABLE"
                if engine.accel_model.model is not None
                else ("NOT_CONFIGURED" if MODEL_PATH is None else "UNAVAILABLE")
            ),
            "accepted_imu": engine.accepted_imu,
            "accepted_gnss": engine.accepted_gnss,
        }


@app.get("/navigation/state")
def navigation_state() -> dict[str, Any]:
    with engine_lock:
        return engine.state_snapshot()


@app.post("/sensor/imu")
def sensor_imu(payload: IMUPayload) -> dict[str, Any]:
    try:
        mag = None
        if payload.mx is not None and payload.my is not None and payload.mz is not None:
            mag = [payload.mx, payload.my, payload.mz]
        with engine_lock:
            return engine.process_imu(
                timestamp=payload.timestamp,
                accel=[payload.ax, payload.ay, payload.az],
                gyro=[payload.gx, payload.gy, payload.gz],
                mag=mag,
            )
    except ValueError as exc:
        raise _error(exc) from exc


@app.post("/sensor/gnss")
def sensor_gnss(payload: GNSSPayload) -> dict[str, Any]:
    try:
        with engine_lock:
            return engine.process_gnss(
                timestamp=payload.timestamp,
                latitude=payload.latitude,
                longitude=payload.longitude,
                speed_mps=payload.speed,
                accuracy_m=payload.accuracy,
                altitude_m=payload.altitude,
            )
    except ValueError as exc:
        raise _error(exc) from exc


@app.post("/sensor")
def sensor_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Single endpoint for generic external-IMU/dataset clients."""
    try:
        event_type = payload.get("type")
        if event_type == "imu":
            return sensor_imu(IMUPayload.model_validate(payload))
        if event_type == "gnss":
            return sensor_gnss(GNSSPayload.model_validate(payload))
        raise HTTPException(status_code=422, detail="type must be 'imu' or 'gnss'")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


@app.post("/config/alignment")
def configure_alignment(payload: AlignmentPayload) -> dict[str, Any]:
    try:
        with engine_lock:
            engine.orientation.set_vehicle_alignment(
                payload.forward_phone,
                payload.up_phone,
            )
            engine.forward_phone = payload.forward_phone
            engine.up_phone = payload.up_phone
            return {
                "ok": True,
                "mounting_calibrated": engine.orientation.vehicle_calibrated,
                "forward_phone": payload.forward_phone,
                "up_phone": payload.up_phone,
            }
    except (TypeError, ValueError) as exc:
        raise _error(exc) from exc


@app.post("/demo/reset")
def demo_reset() -> dict[str, Any]:
    """Reset the live engine before a replay or a physical demonstration."""
    with engine_lock:
        engine.reset()
        return engine.state_snapshot()


@app.websocket("/ws/sensor")
async def sensor_websocket(websocket: WebSocket) -> None:
    """Bidirectional low-latency sensor stream.

    Each received JSON message is the same shape as /sensor with a ``type``
    field. The server replies with the latest navigation state.
    """
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            try:
                event_type = payload.get("type")
                if event_type == "imu":
                    result = sensor_imu(IMUPayload.model_validate(payload))
                elif event_type == "gnss":
                    result = sensor_gnss(GNSSPayload.model_validate(payload))
                else:
                    raise ValueError("type must be 'imu' or 'gnss'")
                await websocket.send_json(result)
            except Exception as exc:
                await websocket.send_json({"ok": False, "error": str(exc)})
    except WebSocketDisconnect:
        return