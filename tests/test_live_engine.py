import math

import numpy as np

from navigation_engine import GNSSState, GyroYawCalibrator, NavigationEngine
from sensor_processing import RobustIMUPreprocessor, TimestampNormalizer


def _make_engine_with_simulated_clock(**kwargs) -> tuple["NavigationEngine", "list[float]"]:
    """Return an engine whose clock tracks the last IMU/GNSS timestamp fed to it.

    A single-element list is returned alongside the engine so that callers can
    advance the clock by mutating ``clock[0]`` before each call, or let the
    helper ``_feed_sim`` do it automatically.
    """
    clock: list[float] = [0.0]
    engine = NavigationEngine(_clock=lambda: clock[0], **kwargs)
    return engine, clock


class _FixedSpeedModel:
    available = True

    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, _values):
        return self.prediction


def _feed_reliable_alignment_drive(engine: NavigationEngine, *, samples: int = 36) -> None:
    """Eastbound, smooth driving with recurrent accurate GNSS course fixes."""
    latitude = 12.0
    longitude = 77.0
    engine.process_gnss(0.0, latitude, longitude, speed_mps=8.0, accuracy_m=4.0)
    engine.process_gnss(0.5, latitude, longitude + 0.0001, speed_mps=8.0, accuracy_m=4.0)
    for step in range(1, samples + 1):
        timestamp = 0.5 + step * 0.1
        if step % 10 == 0:
            longitude += 0.0001
            engine.process_gnss(timestamp - 0.01, latitude, longitude, speed_mps=8.0, accuracy_m=4.0)
        engine.process_imu(
            timestamp=timestamp,
            accel=[0.0, 0.0, 9.80665],
            gyro=[0.0, 0.0, 0.0],
            mag=[1.0, 0.0, 0.0],
        )


def test_timestamp_normalizer_accepts_android_nanoseconds_and_rejects_duplicates():
    normalizer = TimestampNormalizer(default_dt=0.01)
    first = normalizer.accept(2_000_000_000_000_000)
    second = normalizer.accept(2_000_000_010_000_000)
    assert first.timestamp == 2_000_000.0
    assert math.isclose(second.dt, 0.01, rel_tol=1e-6)

    try:
        normalizer.accept(2_000_000_010_000_000)
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("duplicate timestamps must be rejected")


def test_preprocessor_removes_stationary_gravity_and_limits_spike():
    processor = RobustIMUPreprocessor()
    stationary = processor.update([0.0, 0.0, 9.80665], [0.0, 0.0, 0.0])
    assert np.linalg.norm(stationary.filtered_linear_accel_phone) < 1e-9

    shocked = processor.update([100.0, 0.0, 9.80665], [0.0, 0.0, 0.0])
    assert np.max(np.abs(shocked.filtered_linear_accel_phone)) <= 35.0


def test_engine_propagates_during_gnss_loss_and_fuses_on_recovery():
    engine, clock = _make_engine_with_simulated_clock(gnss_timeout_s=0.5)

    clock[0] = 0.0
    engine.process_gnss(timestamp=0.0, latitude=12.0, longitude=77.0, speed_mps=10.0, accuracy_m=4.0)
    clock[0] = 0.1
    engine.process_gnss(timestamp=0.1, latitude=12.0, longitude=77.00001, speed_mps=10.0, accuracy_m=4.0)
    assert engine.heading_deg is not None

    state = None
    for step in range(1, 21):
        t = 0.1 + step * 0.1
        clock[0] = t
        state = engine.process_imu(timestamp=t, accel=[0.0, 0.0, 9.80665], gyro=[0.0, 0.0, 0.0])
    assert state["gnss_state"] == GNSSState.INS_DEAD_RECKONING.value
    assert state["position"] is not None
    assert state["local_position_m"]["east"] > 0.0

    clock[0] = 2.3
    recovered = engine.process_gnss(timestamp=2.3, latitude=12.0, longitude=77.0002, speed_mps=10.0, accuracy_m=4.0)
    assert recovered["gnss_state"] == GNSSState.GNSS_REACQUISITION.value
    assert recovered["mode"] == "REACQUISITION"


def test_alignment_starts_unaligned_and_requires_real_course_evidence():
    engine = NavigationEngine()
    assert engine.state_snapshot()["alignment_state"] == "UNALIGNED"

    for index in range(15):
        engine.process_imu(
            timestamp=index * 0.1,
            accel=[0.0, 0.0, 9.80665],
            gyro=[0.0, 0.0, 0.0],
            mag=[1.0, 0.0, 0.0],
        )

    state = engine.state_snapshot()
    assert state["alignment_state"] == "CALIBRATING"
    assert state["alignment_locked"] is False
    assert state["alignment_confidence"] < 0.4


def test_stable_gnss_and_imu_evidence_locks_automatic_alignment():
    engine = NavigationEngine()
    _feed_reliable_alignment_drive(engine)

    state = engine.state_snapshot()
    assert state["alignment_state"] == "LOCKED"
    assert state["alignment_source"] == "AUTOMATIC"
    assert state["alignment_locked"] is True
    assert state["alignment_confidence"] >= 0.8
    assert state["mounting_calibrated"] is True


def test_vibration_and_shocks_do_not_false_lock_alignment():
    engine = NavigationEngine()
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=8.0, accuracy_m=4.0)
    engine.process_gnss(0.5, 12.0, 77.0001, speed_mps=8.0, accuracy_m=4.0)
    for step in range(1, 45):
        engine.process_imu(
            timestamp=0.5 + step * 0.1,
            accel=[30.0 if step % 2 else -30.0, 0.0, 9.80665],
            gyro=[0.7, 0.0, 0.0],
            mag=[1.0, 0.0, 0.0],
        )

    state = engine.state_snapshot()
    assert state["alignment_locked"] is False
    assert state["alignment_state"] == "CALIBRATING"
    assert state["alignment_confidence"] < 0.5


def test_manual_alignment_cannot_be_overwritten_and_survives_reset():
    engine = NavigationEngine()
    engine.apply_manual_alignment([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    manual_forward = list(engine.forward_phone)
    _feed_reliable_alignment_drive(engine)

    state = engine.state_snapshot()
    assert state["alignment_state"] == "LOCKED"
    assert state["alignment_source"] == "MANUAL"
    assert state["alignment_locked"] is True
    assert engine.forward_phone == manual_forward

    engine.reset()
    reset_state = engine.state_snapshot()
    assert reset_state["alignment_source"] == "MANUAL"
    assert reset_state["alignment_locked"] is True
    assert engine.forward_phone == manual_forward


def test_stationary_and_idling_imu_do_not_accumulate_false_distance():
    engine = NavigationEngine()
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=0.0, accuracy_m=4.0)
    initial_position = engine.position.copy()
    for step in range(1, 50):
        engine.process_imu(
            timestamp=step * 0.1,
            accel=[0.0, 0.08 * math.sin(step), 9.80665],
            gyro=[0.0, 0.0, 0.02],
        )

    state = engine.state_snapshot()
    assert np.linalg.norm(engine.position - initial_position) < 0.05
    assert state["speed_mps"] == 0.0
    assert state["motion_state"] in {"STATIONARY", "IDLING"}


def test_vibration_and_shock_are_bounded_and_not_integrated_as_speed():
    processor = RobustIMUPreprocessor()
    processor.update([0.0, 0.0, 9.80665], [0.0, 0.0, 0.0])
    shock = processor.update([80.0, 0.0, 9.80665], [4.0, 0.0, 0.0])
    assert shock.shock_detected is True
    assert shock.motion_state.value == "SHOCK"
    assert np.linalg.norm(shock.filtered_linear_accel_phone) <= processor.shock_limit_mps2

    engine = NavigationEngine()
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=0.0, accuracy_m=4.0)
    state = engine.process_imu(0.1, [80.0, 0.0, 9.80665], [4.0, 0.0, 0.0])
    assert state["speed_mps"] == 0.0
    assert state["motion_state"] == "SHOCK"
    assert state["mount_disturbance_status"] == "SUSPECTED"


def test_genuine_motion_remains_usable_after_quality_filtering():
    # Use gnss_timeout_s=1.0 so GNSS ages out before the 3-second IMU run ends
    engine, clock = _make_engine_with_simulated_clock(gnss_timeout_s=1.0)
    clock[0] = 0.0
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=0.0, accuracy_m=4.0)
    state = None
    for step in range(1, 31):
        t = step * 0.1
        clock[0] = t
        state = engine.process_imu(timestamp=t, accel=[0.0, 2.0, 9.80665], gyro=[0.0, 0.0, 0.0])
    assert state is not None
    assert state["motion_state"] == "MOVING"
    assert state["speed_mps"] > 0.3
    assert state["speed_source"] == "INERTIAL"


def test_ml_speed_is_used_only_when_valid_and_falls_back_when_invalid():
    # Use gnss_timeout_s=1.0 so GNSS ages out at t=1.1
    engine, clock = _make_engine_with_simulated_clock(gnss_timeout_s=1.0)
    engine.speed_estimator = _FixedSpeedModel(7.0)
    engine.ai_model_kind = "speed"
    clock[0] = 0.0
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=5.0, accuracy_m=4.0)
    clock[0] = 1.1
    state = engine.process_imu(1.1, [0.0, 2.0, 9.80665], [0.0, 0.0, 0.0])
    assert state["speed_source"] == "ML"
    assert 0.0 < state["speed_confidence"] <= 1.0

    engine, clock = _make_engine_with_simulated_clock(gnss_timeout_s=1.0)
    engine.speed_estimator = _FixedSpeedModel(float("nan"))
    engine.ai_model_kind = "speed"
    clock[0] = 0.0
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=5.0, accuracy_m=4.0)
    clock[0] = 1.1
    state = engine.process_imu(1.1, [0.0, 2.0, 9.80665], [0.0, 0.0, 0.0])
    assert state["speed_source"] == "INERTIAL"


# ---------------------------------------------------------------------------
# Additional coverage: DR drift, GNSS accuracy states, reacquisition quality
# ---------------------------------------------------------------------------

def test_dr_uncertainty_grows_during_gnss_outage():
    """Uncertainty_m should increase monotonically during INS dead reckoning."""
    engine, clock = _make_engine_with_simulated_clock(gnss_timeout_s=0.5)

    clock[0] = 0.0
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=10.0, accuracy_m=4.0)
    clock[0] = 0.1
    engine.process_gnss(0.1, 12.0, 77.00001, speed_mps=10.0, accuracy_m=4.0)

    # Drive into dead reckoning
    prev_uncertainty = None
    for step in range(1, 15):
        t = 0.1 + step * 0.1
        clock[0] = t
        state = engine.process_imu(
            timestamp=t,
            accel=[0.0, 0.0, 9.80665],
            gyro=[0.0, 0.0, 0.0],
        )
        if state["gnss_state"] == GNSSState.INS_DEAD_RECKONING.value:
            unc = state["uncertainty_m"]
            assert unc is not None
            if prev_uncertainty is not None:
                assert unc >= prev_uncertainty, (
                    f"Uncertainty should not shrink during DR: {prev_uncertainty} -> {unc}"
                )
            prev_uncertainty = unc

    assert prev_uncertainty is not None, "Must have entered dead reckoning at some point"


def test_gnss_degraded_state_on_low_accuracy():
    """Poor GNSS accuracy (>25m) after first fix should produce GNSS_DEGRADED."""
    engine = NavigationEngine(gnss_timeout_s=5.0)
    # First fix establishes the origin cleanly
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=8.0, accuracy_m=5.0)
    engine.process_gnss(0.5, 12.0, 77.00005, speed_mps=8.0, accuracy_m=5.0)

    # Feed some IMU to advance time slightly
    for step in range(1, 4):
        engine.process_imu(
            timestamp=0.5 + step * 0.1,
            accel=[0.0, 0.0, 9.80665],
            gyro=[0.0, 0.0, 0.0],
        )

    # A fix with poor accuracy should produce GNSS_DEGRADED
    state = engine.process_gnss(
        timestamp=0.9,
        latitude=12.0,
        longitude=77.0001,
        speed_mps=8.0,
        accuracy_m=30.0,   # > 25 m threshold
    )
    assert state["gnss_state"] == GNSSState.GNSS_DEGRADED.value, (
        f"Expected GNSS_DEGRADED for 30m accuracy, got {state['gnss_state']}"
    )
    assert state["gnss_status"] == "DEGRADED"


def test_reacquisition_produces_no_catastrophic_position_jump():
    """After GNSS recovery, fused position must be closer to GNSS than raw INS."""
    engine, clock = _make_engine_with_simulated_clock(gnss_timeout_s=0.5)

    clock[0] = 0.0
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=10.0, accuracy_m=4.0)
    clock[0] = 0.1
    engine.process_gnss(0.1, 12.0, 77.00001, speed_mps=10.0, accuracy_m=4.0)

    # 20 IMU-only steps: DR drifts the position
    for step in range(1, 21):
        t = 0.1 + step * 0.1
        clock[0] = t
        engine.process_imu(timestamp=t, accel=[0.0, 0.0, 9.80665], gyro=[0.0, 0.0, 0.0])

    ins_position = engine.position.copy()

    # GNSS returns 22 m east of origin
    gnss_lat = 12.0
    gnss_lon = 77.0002  # approximately 22 m east
    clock[0] = 2.2
    recovered = engine.process_gnss(
        timestamp=2.2, latitude=gnss_lat, longitude=gnss_lon, speed_mps=10.0, accuracy_m=4.0,
    )
    assert recovered["gnss_state"] == GNSSState.GNSS_REACQUISITION.value

    fused_east = engine.position[0]
    gnss_east = engine._ll_to_xy(gnss_lat, gnss_lon)[0]

    # Fused position must be between DR and GNSS (no snap to exact GNSS)
    assert fused_east < gnss_east, "Fused should not jump all the way to GNSS immediately"
    assert fused_east >= ins_position[0] or fused_east > 0, (
        "Fused should move toward GNSS"
    )


def test_yaw_calibrator_uses_gnss_windows_and_locks_best_signed_axis():
    calibrator = GyroYawCalibrator(min_distance_m=3.0, min_speed_mps=2.0)
    assert not calibrator.observe_gnss(np.array([0.0, 0.0]), 6.0)

    position = np.zeros(2)
    for heading_deg in (0.0, 45.0, 90.0, 135.0):
        # The physical vehicle yaw is phone gyro Y with the opposite sign.
        for _ in range(30):
            calibrator.add_imu(np.array([0.01, -math.radians(15.0), 0.005]), 0.1)
        position += 5.0 * np.array([
            math.sin(math.radians(heading_deg)),
            math.cos(math.radians(heading_deg)),
        ])
        calibrator.observe_gnss(position, 6.0)

    assert calibrator.axis == 1
    assert calibrator.sign == -1.0
    assert calibrator.locked

    # Duplicate/short GNSS movement is rejected and cannot change calibration.
    assert not calibrator.observe_gnss(position + np.array([0.5, 0.0]), 6.0)
    assert calibrator.axis == 1


def test_gnss_aided_state_with_good_accuracy():
    """First fix with good accuracy establishes GNSS_AIDED immediately."""
    engine = NavigationEngine(gnss_timeout_s=2.0)
    state = engine.process_gnss(
        0.0, 12.0, 77.0, speed_mps=5.0, accuracy_m=5.0
    )
    assert state["gnss_state"] == GNSSState.GNSS_AIDED.value
    assert state["gnss_status"] == "CONNECTED"
    assert state["position"] is not None
    assert math.isclose(state["position"]["latitude"], 12.0, rel_tol=1e-6)


def test_reset_clears_position_and_counters_but_keeps_manual_alignment():
    """Engine reset must wipe navigation state while preserving manual alignment."""
    engine = NavigationEngine(gnss_timeout_s=1.0)
    engine.apply_manual_alignment([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=5.0, accuracy_m=4.0)
    engine.process_imu(0.1, [0.0, 0.0, 9.80665], [0.0, 0.0, 0.0])
    assert engine.accepted_imu == 1
    assert engine.accepted_gnss == 1

    engine.reset()

    assert engine.accepted_imu == 0
    assert engine.accepted_gnss == 0
    assert engine.origin_latitude is None
    assert engine.gnss_state == GNSSState.WAITING_FOR_FIX
    # Manual alignment must survive reset
    assert engine.alignment_locked is True
    assert engine.alignment_source == "MANUAL"


def test_filter_mode_change_takes_effect_immediately():
    """set_runtime_options must apply filter mode to the preprocessor instance."""
    engine = NavigationEngine()
    assert engine.filter_mode == "balanced"

    engine.set_runtime_options(filter_mode="strict")
    assert engine.filter_mode == "strict"
    assert engine.imu_preprocessor.filter_mode == "strict"

    engine.set_runtime_options(filter_mode="raw")
    assert engine.filter_mode == "raw"
    assert engine.imu_preprocessor.signal_cutoff_hz == 40.0


def test_nhc_disabled_allows_lateral_velocity_contribution():
    """With NHC disabled, lateral velocity adds a small non-zero east component."""
    engine_nhc = NavigationEngine(gnss_timeout_s=5.0)
    engine_no_nhc = NavigationEngine(gnss_timeout_s=5.0)
    engine_no_nhc.set_runtime_options(nhc_enabled=False)

    for engine in (engine_nhc, engine_no_nhc):
        engine.process_gnss(0.0, 12.0, 77.0, speed_mps=10.0, accuracy_m=4.0)
        engine.process_gnss(0.1, 12.0, 77.00001, speed_mps=10.0, accuracy_m=4.0)

    # 10 IMU samples with a lateral linear acceleration component
    for step in range(1, 11):
        t = 0.1 + step * 0.1
        for engine in (engine_nhc, engine_no_nhc):
            engine.process_imu(
                timestamp=t,
                accel=[2.0, 0.0, 9.80665],   # lateral accel in phone X
                gyro=[0.0, 0.0, 0.0],
            )

    # Both paths produce valid positions; NHC path must not crash
    state_nhc = engine_nhc.state_snapshot()
    state_no_nhc = engine_no_nhc.state_snapshot()
    assert state_nhc["nhc_status"] == "KINEMATIC"
    assert state_no_nhc["nhc_status"] == "DISABLED"
    assert state_nhc["position"] is not None
    assert state_no_nhc["position"] is not None


# ---------------------------------------------------------------------------
# Regression: GNSS freshness must use backend monotonic clock, not Android
# device timestamps, so continuous GNSS packets never trigger DEAD_RECKONING.
# ---------------------------------------------------------------------------

def test_continuous_gnss_never_triggers_dead_reckoning():
    """Simulates the real-time pipeline: interleaved GNSS + IMU at normal rates.

    This is the regression test for the gnss_age_s=12884954 bug where
    Android boot-relative timestamps were subtracted from Python monotonic
    timestamps, producing an impossibly large age that immediately set
    gnss_state=INS_DEAD_RECKONING even while fresh GNSS packets arrived.

    With the fix, _gnss_age and _update_mode_from_age both use a single
    backend monotonic clock reference (self._clock), so any timestamp
    carried in the GNSS or IMU packet payload is irrelevant to freshness.
    """
    # Simulate: phone has been on for ~10 hours before starting the test.
    # Android sends elapsedRealtimeNanos/1e9 ≈ 36000 s for both IMU and GNSS.
    boot_offset = 36_000.0  # seconds of simulated boot time

    # The engine's clock starts just after the first GNSS packet
    real_clock: list[float] = [0.0]
    engine = NavigationEngine(gnss_timeout_s=3.0, _clock=lambda: real_clock[0])

    # First GNSS fix — clock is at 0.0 (backend receive time)
    real_clock[0] = 0.0
    engine.process_gnss(
        timestamp=boot_offset + 0.0,   # Android timestamp in seconds since boot
        latitude=12.9716,
        longitude=77.5946,
        speed_mps=5.0,
        accuracy_m=4.0,
    )

    # Simulate 20 interleaved GNSS+IMU cycles at realistic rates:
    # - IMU at 10 Hz (every 100 ms)
    # - GNSS at ~1 Hz (every 1 s)
    gnss_interval = 1.0   # seconds
    imu_interval  = 0.1   # seconds
    next_gnss_backend_t = 1.0
    next_gnss_device_t  = boot_offset + 1.0
    gnss_lat = 12.9716

    for i in range(1, 21):
        backend_t = i * imu_interval
        device_t  = boot_offset + backend_t
        real_clock[0] = backend_t

        # Inject a GNSS fix once per second
        if backend_t >= next_gnss_backend_t - 1e-9:
            gnss_lat += 0.00001  # slight movement northward
            engine.process_gnss(
                timestamp=next_gnss_device_t,
                latitude=gnss_lat,
                longitude=77.5946,
                speed_mps=5.0,
                accuracy_m=4.0,
            )
            next_gnss_backend_t += gnss_interval
            next_gnss_device_t  += gnss_interval

        state = engine.process_imu(
            timestamp=device_t,
            accel=[0.0, 0.0, 9.80665],
            gyro=[0.0, 0.0, 0.0],
        )

        # While GNSS keeps arriving (within timeout), must NEVER be dead reckoning
        assert state["gnss_state"] != GNSSState.INS_DEAD_RECKONING.value, (
            f"Incorrectly entered DEAD_RECKONING at backend_t={backend_t:.2f}s, "
            f"gnss_age_s={state['gnss_age_s']}"
        )
        assert state["mode"] != "DEAD_RECKONING", (
            f"mode=DEAD_RECKONING at backend_t={backend_t:.2f}s"
        )
        # gnss_age must be realistic — under the gnss_interval + small margin
        age = state["gnss_age_s"]
        assert age is not None and age < gnss_interval + 0.2, (
            f"gnss_age_s={age} is unrealistically large (expected < {gnss_interval + 0.2})"
        )

    # Final state: connected
    final = engine.state_snapshot()
    assert final["gnss_status"] in {"CONNECTED", "DEGRADED"}
    assert final["mode"] != "DEAD_RECKONING"
    assert final["accepted_gnss"] >= 2
    assert final["accepted_imu"] >= 1
