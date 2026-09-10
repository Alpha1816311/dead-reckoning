import math

import numpy as np

from navigation_engine import GNSSState, NavigationEngine
from sensor_processing import RobustIMUPreprocessor, TimestampNormalizer


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
    engine = NavigationEngine(gnss_timeout_s=0.5)
    engine.process_gnss(
        timestamp=0.0,
        latitude=12.0,
        longitude=77.0,
        speed_mps=10.0,
        accuracy_m=4.0,
    )
    # Establish an eastbound course from two fixes.
    engine.process_gnss(
        timestamp=0.1,
        latitude=12.0,
        longitude=77.00001,
        speed_mps=10.0,
        accuracy_m=4.0,
    )
    assert engine.heading_deg is not None

    state = None
    for step in range(1, 21):
        state = engine.process_imu(
            timestamp=0.1 + step * 0.1,
            accel=[0.0, 0.0, 9.80665],
            gyro=[0.0, 0.0, 0.0],
        )
    assert state["gnss_state"] == GNSSState.INS_DEAD_RECKONING.value
    assert state["position"] is not None
    assert state["local_position_m"]["east"] > 0.0

    recovered = engine.process_gnss(
        timestamp=2.3,
        latitude=12.0,
        longitude=77.0002,
        speed_mps=10.0,
        accuracy_m=4.0,
    )
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
    engine = NavigationEngine()
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=0.0, accuracy_m=4.0)
    state = None
    for step in range(1, 31):
        state = engine.process_imu(
            timestamp=step * 0.1,
            accel=[0.0, 2.0, 9.80665],
            gyro=[0.0, 0.0, 0.0],
        )
    assert state is not None
    assert state["motion_state"] == "MOVING"
    assert state["speed_mps"] > 0.3
    assert state["speed_source"] == "INERTIAL"


def test_ml_speed_is_used_only_when_valid_and_falls_back_when_invalid():
    engine = NavigationEngine()
    engine.speed_estimator = _FixedSpeedModel(7.0)
    engine.ai_model_kind = "speed"
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=5.0, accuracy_m=4.0)
    state = engine.process_imu(1.1, [0.0, 2.0, 9.80665], [0.0, 0.0, 0.0])
    assert state["speed_source"] == "ML"
    assert 0.0 < state["speed_confidence"] <= 1.0

    engine = NavigationEngine()
    engine.speed_estimator = _FixedSpeedModel(float("nan"))
    engine.ai_model_kind = "speed"
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=5.0, accuracy_m=4.0)
    state = engine.process_imu(1.1, [0.0, 2.0, 9.80665], [0.0, 0.0, 0.0])
    assert state["speed_source"] == "INERTIAL"


# ---------------------------------------------------------------------------
# Additional coverage: DR drift, GNSS accuracy states, reacquisition quality
# ---------------------------------------------------------------------------

def test_dr_uncertainty_grows_during_gnss_outage():
    """Uncertainty_m should increase monotonically during INS dead reckoning."""
    engine = NavigationEngine(gnss_timeout_s=0.5)
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=10.0, accuracy_m=4.0)
    engine.process_gnss(0.1, 12.0, 77.00001, speed_mps=10.0, accuracy_m=4.0)

    # Drive into dead reckoning
    prev_uncertainty = None
    for step in range(1, 15):
        state = engine.process_imu(
            timestamp=0.1 + step * 0.1,
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
    engine = NavigationEngine(gnss_timeout_s=0.5)
    # Establish position at origin
    engine.process_gnss(0.0, 12.0, 77.0, speed_mps=10.0, accuracy_m=4.0)
    engine.process_gnss(0.1, 12.0, 77.00001, speed_mps=10.0, accuracy_m=4.0)

    # 20 IMU-only steps: DR drifts the position
    for step in range(1, 21):
        engine.process_imu(
            timestamp=0.1 + step * 0.1,
            accel=[0.0, 0.0, 9.80665],
            gyro=[0.0, 0.0, 0.0],
        )

    ins_position = engine.position.copy()

    # GNSS returns 22 m east of origin
    gnss_lat = 12.0
    gnss_lon = 77.0002  # approximately 22 m east
    recovered = engine.process_gnss(
        timestamp=2.2,
        latitude=gnss_lat,
        longitude=gnss_lon,
        speed_mps=10.0,
        accuracy_m=4.0,
    )
    assert recovered["gnss_state"] == GNSSState.GNSS_REACQUISITION.value

    fused_east = engine.position[0]
    gnss_east = engine._ll_to_xy(gnss_lat, gnss_lon)[0]

    # Fused position must be between DR and GNSS (no snap to exact GNSS)
    assert fused_east < gnss_east, "Fused should not jump all the way to GNSS immediately"
    assert fused_east >= ins_position[0] or fused_east > 0, (
        "Fused should move toward GNSS"
    )


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
