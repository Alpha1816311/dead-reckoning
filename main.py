import os
import json
import warnings
import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

# Initialize FastAPI Application
app = FastAPI(
    title="Intelligent Dead Reckoning (IDR) Engine API",
    version="1.0.0",
    description="Live backend engine for processing real-time IMU and GNSS telemetry streams."
)

# Enable CORS for Vercel Web Dashboard and Android Client Requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State for Live Tracking Session
session_state = {
    "current_x": 0.0,
    "current_y": 0.0,
    "current_speed": 0.0,
    "current_heading": 0.0,
    "trajectory_history": []
}

# Request Payload Schema from Android/Web Simulators
class SensorPayload(BaseModel):
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    dt: float = 0.01
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_speed: Optional[float] = None

@app.get("/")
def health_check():
    return {
        "status": "online",
        "system": "IDR Engine API",
        "version": "1.0.0"
    }

@app.post("/api/telemetry")
def process_telemetry(data: SensorPayload):
    """
    Receives live IMU/GNSS data from Member 2's Android App or Web UI,
    calculates velocity & heading updates, and tracks position.
    """
    # 1. Accelerometer Magnitude & Gravity Removal Estimation
    acc_mag = np.sqrt(data.accel_x**2 + data.accel_y**2 + data.accel_z**2) - 9.81
    if abs(acc_mag) < 0.15:
        acc_mag = 0.0

    # 2. Velocity Estimation
    estimated_speed = max(0.0, session_state["current_speed"] + acc_mag * data.dt)
    if data.gps_speed is not None and data.gps_speed > 0:
        estimated_speed = 0.7 * data.gps_speed + 0.3 * estimated_speed

    # 3. Orientation / Heading Update
    session_state["current_heading"] = (session_state["current_heading"] + np.degrees(data.gyro_z * data.dt)) % 360.0
    rad_heading = np.radians(session_state["current_heading"])

    # 4. Dead Reckoning Integration Step
    dx = estimated_speed * np.sin(rad_heading) * data.dt
    dy = estimated_speed * np.cos(rad_heading) * data.dt

    session_state["current_x"] += dx
    session_state["current_y"] += dy
    session_state["current_speed"] = estimated_speed

    point = {
        "x": round(session_state["current_x"], 3),
        "y": round(session_state["current_y"], 3),
        "speed": round(session_state["current_speed"], 2),
        "heading": round(session_state["current_heading"], 1)
    }

    session_state["trajectory_history"].append(point)
    if len(session_state["trajectory_history"]) > 1000:
        session_state["trajectory_history"].pop(0)

    return {"status": "success", "current_position": point}

@app.get("/api/position")
def get_current_position():
    """Endpoint for Vercel Web UI to fetch real-time position."""
    return {
        "x": session_state["current_x"],
        "y": session_state["current_y"],
        "speed": session_state["current_speed"],
        "heading": session_state["current_heading"],
        "history": session_state["trajectory_history"][-100:]
    }

@app.post("/api/reset")
def reset_session():
    """Resets positioning tracking state."""
    global session_state
    session_state = {
        "current_x": 0.0,
        "current_y": 0.0,
        "current_speed": 0.0,
        "current_heading": 0.0,
        "trajectory_history": []
    }
    return {"status": "reset_successful"}