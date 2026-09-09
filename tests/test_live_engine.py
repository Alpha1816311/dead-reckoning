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
