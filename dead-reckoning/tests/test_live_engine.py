import math

import numpy as np

from navigation_engine import GNSSState, NavigationEngine
from sensor_processing import RobustIMUPreprocessor, TimestampNormalizer


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
