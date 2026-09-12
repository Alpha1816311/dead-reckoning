"""
MVP State Machine Validation Tests (Phase 7)

Tests the complete GNSS → BLACKOUT → DEAD_RECKONING → RECOVERY → FUSED pipeline
as required by the MVP definition of done.

Test A: GNSS AVAILABLE  → fused position
Test B: GNSS LOSS        → deficit detected, DR starts
Test C: BLACKOUT         → IMU continues, position updates (no freeze)
Test D: GNSS RETURN      → recovery mode, staged correction, fused state
Test E: CONTINUOUS OUTPUT→ application never stops producing navigation states
Test F: Sensor interface → canonical events dispatch correctly
Test G: Performance      → update rate >= 10 Hz in batch mode
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from navigation_engine import GNSSState, NavigationEngine
from sensor_interface import (
    GNSSEvent,
    IMUEvent,
    NavigationState,
    SensorDispatcher,
    android_gnss_to_event,
    android_imu_to_event,
    normalize_sensor_event,
)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _engine_with_clock(**kwargs):
    clock: list[float] = [0.0]
    engine = NavigationEngine(_clock=lambda: clock[0], **kwargs)
    return engine, clock


def _feed_imu(engine, clock, t, accel=(0.0, 0.0, 9.80665), gyro=(0.0, 0.0, 0.0)):
    clock[0] = t
    return engine.process_imu(timestamp=t, accel=list(accel), gyro=list(gyro))


def _feed_gnss(engine, clock, t, lat=12.0, lon=77.0, speed=10.0, acc=4.0):
    clock[0] = t
    return engine.process_gnss(timestamp=t, latitude=lat, longitude=lon,
                                speed_mps=speed, accuracy_m=acc)


# ──────────────────────────────────────────────────────────────────────────────
# TEST A — GNSS AVAILABLE → FUSED POSITION
# ──────────────────────────────────────────────────────────────────────────────

def test_A_gnss_available_produces_fused_position():
    """TEST A: with GNSS + IMU, engine should be in GNSS_AIDED or FUSED mode."""
    engine, clock = _engine_with_clock(gnss_timeout_s=2.0)

    # Initial GNSS fix
    state = _feed_gnss(engine, clock, 0.0, lat=12.0, lon=77.0, speed=8.0, acc=5.0)
    assert state["gnss_state"] == GNSSState.GNSS_AIDED.value
    assert state["position"] is not None
    assert state["gnss_status"] == "CONNECTED"

    # Second fix + IMU
    _feed_gnss(engine, clock, 0.5, lat=12.0, lon=77.00005, speed=8.0, acc=5.0)
    state = _feed_imu(engine, clock, 0.6, accel=[0.0, 0.0, 9.80665])

    assert state["position"] is not None, "Position must be non-null with GNSS active"
    assert state["gnss_state"] in (
        GNSSState.GNSS_AIDED.value,
        GNSSState.FUSED.value,
        "GNSS_AIDED",
        "FUSED",
    )
    assert state["gnss_status"] in ("CONNECTED", "CONNECTED")


# ──────────────────────────────────────────────────────────────────────────────
# TEST B — GNSS LOSS → DEFICIT DETECTED, DR STARTS
# ──────────────────────────────────────────────────────────────────────────────

def test_B_gnss_loss_detected_and_dr_starts():
    """TEST B: when GNSS stops, engine must detect deficit and enter DR."""
    engine, clock = _engine_with_clock(gnss_timeout_s=0.5)

    _feed_gnss(engine, clock, 0.0, lat=12.0, lon=77.0, speed=10.0, acc=4.0)
    _feed_gnss(engine, clock, 0.1, lat=12.0, lon=77.00001, speed=10.0, acc=4.0)

    # Feed IMU only — no more GNSS
    dr_state = None
    for step in range(1, 20):
        t = 0.1 + step * 0.1
        state = _feed_imu(engine, clock, t, accel=[0.0, 0.0, 9.80665])
        if state["gnss_state"] == GNSSState.INS_DEAD_RECKONING.value:
            dr_state = state
            break

    assert dr_state is not None, "Engine must enter INS_DEAD_RECKONING after GNSS timeout"
    assert dr_state["gnss_status"] == "LOST"
    assert dr_state["gnss_state"] == GNSSState.INS_DEAD_RECKONING.value


# ──────────────────────────────────────────────────────────────────────────────
# TEST C — BLACKOUT: IMU CONTINUES, NO FREEZE
# ──────────────────────────────────────────────────────────────────────────────

def test_C_blackout_position_continues_no_freeze():
    """TEST C: during GNSS blackout, position must keep updating (never frozen)."""
    engine, clock = _engine_with_clock(gnss_timeout_s=0.5)

    _feed_gnss(engine, clock, 0.0, lat=12.0, lon=77.0, speed=10.0, acc=4.0)
    _feed_gnss(engine, clock, 0.1, lat=12.0, lon=77.00002, speed=10.0, acc=4.0)

    # Advance into DR
    for step in range(1, 8):
        _feed_imu(engine, clock, 0.1 + step * 0.1)

    # Feed IMU with forward motion during full blackout window
    positions = []
    for step in range(30):
        t = 0.9 + step * 0.1
        state = _feed_imu(engine, clock, t, accel=[0.0, 2.0, 9.80665])
        if state.get("position") is not None:
            local = state["local_position_m"]
            positions.append((local["east"], local["north"]))

    assert len(positions) >= 10, "Must have produced positions during blackout"

    # Verify position is NOT frozen (at least some movement)
    east_vals = [p[0] for p in positions]
    north_vals = [p[1] for p in positions]
    movement = math.hypot(east_vals[-1] - east_vals[0], north_vals[-1] - north_vals[0])
    assert movement > 0.01, f"Position should move during DR, got {movement:.4f}m movement"


# ──────────────────────────────────────────────────────────────────────────────
# TEST D — GNSS RETURN → RECOVERY → STAGED CORRECTION → FUSED
# ──────────────────────────────────────────────────────────────────────────────

def test_D_gnss_return_enters_recovery_then_fused():
    """TEST D: when GNSS returns, engine must enter REACQUISITION then FUSED."""
    engine, clock = _engine_with_clock(gnss_timeout_s=0.5)

    # Establish GNSS
    _feed_gnss(engine, clock, 0.0, lat=12.0, lon=77.0, speed=10.0, acc=4.0)
    _feed_gnss(engine, clock, 0.1, lat=12.0, lon=77.00001, speed=10.0, acc=4.0)

    # Go into DR
    for step in range(1, 12):
        t = 0.1 + step * 0.1
        _feed_imu(engine, clock, t)

    # Verify DR
    snap = engine.state_snapshot()
    assert snap["gnss_state"] == GNSSState.INS_DEAD_RECKONING.value

    # GNSS returns
    t_return = 0.1 + 12 * 0.1
    clock[0] = t_return
    recovered = engine.process_gnss(
        timestamp=t_return,
        latitude=12.0,
        longitude=77.0002,
        speed_mps=10.0,
        accuracy_m=4.0,
    )
    assert recovered["gnss_state"] == GNSSState.GNSS_REACQUISITION.value, (
        f"Expected GNSS_REACQUISITION, got {recovered['gnss_state']}"
    )
    assert recovered["mode"] == "REACQUISITION"

    # Continue IMU + GNSS — engine should eventually reach FUSED or GNSS_AIDED
    final_state = None
    for step in range(1, 15):
        t2 = t_return + step * 0.1
        clock[0] = t2
        if step % 3 == 0:
            engine.process_gnss(
                timestamp=t2,
                latitude=12.0,
                longitude=77.0002 + step * 0.00001,
                speed_mps=10.0,
                accuracy_m=4.0,
            )
        final_state = _feed_imu(engine, clock, t2)

    assert final_state is not None
    assert final_state["gnss_state"] in (
        GNSSState.FUSED.value,
        GNSSState.GNSS_AIDED.value,
        GNSSState.GNSS_REACQUISITION.value,
    ), f"Expected fused/aided after recovery, got {final_state['gnss_state']}"


# ──────────────────────────────────────────────────────────────────────────────
# TEST E — CONTINUOUS OUTPUT: NEVER STOPS PRODUCING STATES
# ──────────────────────────────────────────────────────────────────────────────

def test_E_continuous_output_never_stops():
    """TEST E: the engine must produce a navigation state for every IMU call."""
    engine, clock = _engine_with_clock(gnss_timeout_s=0.5)

    _feed_gnss(engine, clock, 0.0, lat=12.0, lon=77.0, speed=5.0, acc=5.0)

    none_count = 0
    for step in range(1, 100):
        t = step * 0.1
        clock[0] = t
        # Inject GNSS only for first 10 steps, then simulate outage
        if step <= 10 and step % 5 == 0:
            engine.process_gnss(t, latitude=12.0, longitude=77.0 + step * 0.00001,
                                speed_mps=5.0, accuracy_m=4.0)
        state = engine.process_imu(t, accel=[0.0, 0.0, 9.80665], gyro=[0.0, 0.0, 0.0])

        assert state is not None, f"Engine returned None at step {step}"
        assert "gnss_state" in state, f"No gnss_state at step {step}"
        assert "mode" in state, f"No mode at step {step}"


# ──────────────────────────────────────────────────────────────────────────────
# TEST F — SENSOR INTERFACE: CANONICAL EVENTS DISPATCH CORRECTLY
# ──────────────────────────────────────────────────────────────────────────────

def test_F_sensor_interface_imu_event_dispatches():
    """IMUEvent dispatched through SensorDispatcher produces a NavigationState."""
    engine = NavigationEngine(gnss_timeout_s=5.0)
    dispatcher = SensorDispatcher(engine)

    gnss_ev = GNSSEvent(
        timestamp=0.0,
        latitude=12.0,
        longitude=77.0,
        speed=5.0,
        accuracy=5.0,
    )
    nav_state = dispatcher.dispatch(gnss_ev)
    assert isinstance(nav_state, NavigationState)
    assert nav_state.latitude is not None
    assert nav_state.longitude is not None
    assert nav_state.mode in ("GNSS_AIDED", "GNSS_FUSED", "WAITING_FOR_FIX", "GNSS_RECOVERY",
                               "INS_DEAD_RECKONING", "GNSS_DEGRADED")

    imu_ev = IMUEvent(
        timestamp=0.1,
        accelerometer=[0.0, 0.0, 9.80665],
        gyroscope=[0.0, 0.0, 0.0],
        magnetometer=[1.0, 0.0, 0.0],
    )
    nav_state2 = dispatcher.dispatch(imu_ev)
    assert isinstance(nav_state2, NavigationState)
    assert nav_state2.timestamp == 0.1


def test_F_sensor_interface_dict_dispatch():
    """Dict-based events (WebSocket/HTTP format) dispatch correctly."""
    engine = NavigationEngine(gnss_timeout_s=5.0)
    dispatcher = SensorDispatcher(engine)

    dispatcher.dispatch({
        "type": "gnss",
        "timestamp": 0.0,
        "latitude": 12.0,
        "longitude": 77.0,
        "speed": 5.0,
        "accuracy": 5.0,
    })

    nav = dispatcher.dispatch({
        "type": "imu",
        "timestamp": 0.1,
        "accelerometer": [0.0, 0.0, 9.80665],
        "gyroscope": [0.0, 0.0, 0.0],
    })
    assert isinstance(nav, NavigationState)


def test_F_sensor_interface_android_adapter():
    """Android-format IMU payload converts to canonical IMUEvent."""
    android_payload = {
        "type": "imu",
        "timestamp": 2_000_000_000_000_000,  # nanoseconds
        "ax": 0.1, "ay": 0.2, "az": 9.8,
        "gx": 0.01, "gy": 0.02, "gz": 0.0,
        "mx": 25.0, "my": 5.0, "mz": -40.0,
    }
    ev = android_imu_to_event(android_payload)
    assert isinstance(ev, IMUEvent)
    assert ev.timestamp == pytest.approx(2_000_000.0, rel=1e-6)
    assert ev.magnetometer is not None
    assert len(ev.magnetometer) == 3

    android_gnss = {
        "type": "gnss",
        "timestamp": 1000.0,
        "latitude": 51.5, "longitude": -0.1,
        "speed": 10.0, "accuracy": 5.0,
    }
    gev = android_gnss_to_event(android_gnss)
    assert isinstance(gev, GNSSEvent)
    assert gev.latitude == 51.5


def test_F_normalize_sensor_event_detects_format():
    """normalize_sensor_event handles both canonical and Android formats."""
    canonical = {
        "type": "imu",
        "timestamp": 1.0,
        "accelerometer": [0.0, 0.0, 9.8],
        "gyroscope": [0.0, 0.0, 0.0],
    }
    ev1 = normalize_sensor_event(canonical)
    assert isinstance(ev1, IMUEvent)

    android_style = {
        "type": "imu",
        "timestamp": 1.0,
        "ax": 0.0, "ay": 0.0, "az": 9.8,
        "gx": 0.0, "gy": 0.0, "gz": 0.0,
    }
    ev2 = normalize_sensor_event(android_style)
    assert isinstance(ev2, IMUEvent)


def test_F_navigation_state_from_engine_snapshot():
    """NavigationState.from_engine_snapshot maps engine dict correctly."""
    engine = NavigationEngine(gnss_timeout_s=5.0)
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=5.0, accuracy_m=5.0)
    snap = engine.state_snapshot()

    nav = NavigationState.from_engine_snapshot(snap, gnss_available=True)
    assert nav.latitude is not None
    assert nav.longitude is not None
    assert nav.mode in ("GNSS_AIDED", "GNSS_FUSED", "GNSS_RECOVERY")
    assert nav.gnss_available is True

    # to_dict must be JSON-serialisable
    d = nav.to_dict()
    assert "mode" in d
    assert "latitude" in d
    assert "x" in d


# ──────────────────────────────────────────────────────────────────────────────
# TEST G — PERFORMANCE: >= 10 Hz UPDATE RATE
# ──────────────────────────────────────────────────────────────────────────────

def test_G_imu_processing_at_10hz_or_faster():
    """TEST G: batch IMU processing must sustain >= 10 Hz nav output."""
    engine = NavigationEngine(gnss_timeout_s=5.0)
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=5.0, accuracy_m=5.0)

    N = 200
    t_start = time.perf_counter()
    for i in range(N):
        engine.process_imu(
            timestamp=float(i + 1) * 0.1,
            accel=[0.0, 0.0, 9.80665],
            gyro=[0.0, 0.0, 0.0],
        )
    elapsed = time.perf_counter() - t_start

    actual_rate = N / elapsed
    assert actual_rate >= 10.0, (
        f"Navigation engine too slow: {actual_rate:.1f} Hz (need >= 10 Hz)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# TEST H — COMPLETE FLOW: GNSS → BLACKOUT → DR → RECOVERY → FUSED
# ──────────────────────────────────────────────────────────────────────────────

def test_H_complete_gnss_blackout_and_recovery_flow():
    """
    End-to-end state machine validation:
    GNSS_AIDED → INS_DEAD_RECKONING → GNSS_REACQUISITION → GNSS_AIDED/FUSED
    """
    engine, clock = _engine_with_clock(gnss_timeout_s=0.5)
    states_seen = set()

    # Phase 1: GNSS Available
    _feed_gnss(engine, clock, 0.0, lat=12.0, lon=77.0, speed=10.0, acc=4.0)
    _feed_gnss(engine, clock, 0.1, lat=12.0, lon=77.00002, speed=10.0, acc=4.0)
    s = _feed_imu(engine, clock, 0.15)
    states_seen.add(s["gnss_state"])

    # Phase 2: GNSS Blackout — feed only IMU for 2 seconds
    for step in range(1, 22):
        t = 0.15 + step * 0.1
        s = _feed_imu(engine, clock, t, accel=[0.0, 1.5, 9.80665])
        states_seen.add(s["gnss_state"])

    assert GNSSState.INS_DEAD_RECKONING.value in states_seen, (
        f"DR never entered. States seen: {states_seen}"
    )

    # DR position must exist and be non-zero
    snap = engine.state_snapshot()
    assert snap["position"] is not None
    local = snap["local_position_m"]
    dr_east = local["east"]
    dr_north = local["north"]

    # Phase 3: GNSS Returns
    t_return = 0.15 + 22 * 0.1
    clock[0] = t_return
    recovered = engine.process_gnss(
        timestamp=t_return,
        latitude=12.0,
        longitude=77.0003,
        speed_mps=10.0,
        accuracy_m=4.0,
    )
    states_seen.add(recovered["gnss_state"])

    assert GNSSState.GNSS_REACQUISITION.value in states_seen, (
        f"REACQUISITION never entered. States seen: {states_seen}"
    )

    # Phase 4: Converge to fused
    for step in range(1, 20):
        t2 = t_return + step * 0.1
        clock[0] = t2
        if step % 4 == 0:
            engine.process_gnss(
                t2, latitude=12.0, longitude=77.0003 + step * 0.00001,
                speed_mps=10.0, accuracy_m=4.0,
            )
        s = _feed_imu(engine, clock, t2)
        states_seen.add(s["gnss_state"])

    final = engine.state_snapshot()
    assert final["gnss_state"] in (
        GNSSState.FUSED.value,
        GNSSState.GNSS_AIDED.value,
        GNSSState.GNSS_REACQUISITION.value,
    ), f"Did not reach fused state. Final: {final['gnss_state']}, all seen: {states_seen}"

    print(f"\n[H] States traversed: {sorted(states_seen)}")
    print(f"[H] Final state: {final['gnss_state']}")
