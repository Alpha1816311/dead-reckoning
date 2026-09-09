from __future__ import annotations

import os
import logging
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

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
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Intelligent Dead Reckoning API",
    version="2.0.0-live",
    description=(
        "Live Android GNSS + IMU Intelligent Dead Reckoning "
        "Navigation System"
    ),
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


class AlignmentPayload(BaseModel):

    forward_phone: list[float] = Field(
        min_length=3,
        max_length=3,
    )

    up_phone: list[float] = Field(
        min_length=3,
        max_length=3,
    )


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
def calibration():

    return get_page("sensorcaliberation.html")


@app.get("/pipeline", include_in_schema=False)
def pipeline():

    return get_page("fusionpipeline.html")


@app.get("/settings", include_in_schema=False)
def settings():

    return get_page("setting.html")


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

            "websocket": "/ws/sensor",
        },

        "configuration": {

            "map_path": MAP_PATH,

            "model_path": MODEL_PATH,

            "web_directory": str(WEB_DIR),
        },
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
        }


# ============================================================
# LIVE NAVIGATION STATE
# ============================================================

@app.get("/navigation/state")
def navigation_state():

    with engine_lock:

        return engine.state_snapshot()


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

        return result

    except ValueError as exc:

        logger.warning(
            "IMU rejected: %s",
            exc,
        )

        raise api_error(
            422,
            str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "IMU processing failed"
        )

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

        return result

    except ValueError as exc:

        logger.warning(
            "GNSS rejected: %s",
            exc,
        )

        raise api_error(
            422,
            str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "GNSS processing failed"
        )

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

            engine.orientation.set_vehicle_alignment(

                payload.forward_phone,

                payload.up_phone,
            )

            engine.forward_phone = (
                payload.forward_phone
            )

            engine.up_phone = (
                payload.up_phone
            )

            return {

                "ok": True,

                "mounting_calibrated":
                    engine.orientation.vehicle_calibrated,

                "forward_phone":
                    payload.forward_phone,

                "up_phone":
                    payload.up_phone,
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
# WEBSOCKET
# ============================================================

@app.websocket("/ws/sensor")
async def sensor_websocket(
    websocket: WebSocket,
):

    await websocket.accept()

    logger.info(
        "WebSocket client connected"
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

                else:

                    raise ValueError(
                        "type must be 'imu', "
                        "'gnss', or 'state'"
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

                await websocket.send_json({

                    "ok": False,

                    "error": str(exc),
                })

    except WebSocketDisconnect:

        logger.info(
            "WebSocket client disconnected"
        )


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