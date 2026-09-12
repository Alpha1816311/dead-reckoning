from fastapi.testclient import TestClient

from app import app, engine


client = TestClient(app)


def setup_function():
    engine.reset()
    engine.set_runtime_options(
        filter_mode="balanced",
        nhc_enabled=True,
        map_matching_enabled=True,
        profile="phone",
    )


def test_health_and_sensor_contract():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True

    gnss = client.post(
        "/sensor/gnss",
        json={
            "timestamp": 1.0,
            "latitude": 12.0,
            "longitude": 77.0,
            "speed": 0.0,
            "accuracy": 5.0,
        },
    )
    assert gnss.status_code == 200
    assert gnss.json()["gnss_status"] == "CONNECTED"

    imu = client.post(
        "/sensor/imu",
        json={
            "timestamp": 1.01,
            "ax": 0.0,
            "ay": 0.0,
            "az": 9.80665,
            "gx": 0.0,
            "gy": 0.0,
            "gz": 0.0,
        },
    )
    assert imu.status_code == 200
    assert imu.json()["imu_status"] == "ACTIVE"
    assert "position" in imu.json()


def test_runtime_configuration_can_be_read_and_applied_to_live_engine():
    initial = client.get("/config")
    assert initial.status_code == 200
    assert initial.json()["configuration"]["filter_mode"] == "balanced"

    changed = client.patch(
        "/config",
        json={
            "filter_mode": "strict",
            "nhc_enabled": False,
            "map_matching_enabled": False,
            "gnss_timeout_s": 0.5,
        },
    )
    assert changed.status_code == 200
    configuration = changed.json()["configuration"]
    assert configuration["filter_mode"] == "strict"
    assert configuration["nhc_enabled"] is False
    assert configuration["map_matching_enabled"] is False
    assert configuration["gnss_timeout_s"] == 0.5

    # The endpoint mutates the shared engine rather than returning UI-only state.
    assert engine.imu_preprocessor.filter_mode == "strict"
    assert engine.nhc_enabled is False
    assert engine.gnss_timeout_s == 0.5
    state = client.get("/navigation/state").json()
    assert state["filter_mode"] == "strict"
    assert state["nhc_status"] == "DISABLED"
    assert state["gnss_timeout_s"] == 0.5


def test_runtime_configuration_rejects_invalid_values_and_unknown_settings():
    invalid_type = client.patch("/config", json={"nhc_enabled": "yes"})
    assert invalid_type.status_code == 422

    invalid_range = client.patch("/config", json={"gnss_timeout_s": 0.01})
    assert invalid_range.status_code == 422

    unknown = client.patch("/config", json={"map_path": "C:/secret.geojson"})
    assert unknown.status_code == 422


def test_malformed_sensor_is_rejected():
    response = client.post(
        "/sensor/imu",
        json={
            "timestamp": 1.0,
            "ax": "not-a-number",
            "ay": 0.0,
            "az": 9.8,
            "gx": 0.0,
            "gy": 0.0,
            "gz": 0.0,
        },
    )
    assert response.status_code == 422

    generic = client.post("/sensor", json={"type": "imu", "timestamp": 1.0})
    assert generic.status_code == 422


def test_non_finite_timestamps_are_rejected_by_http_and_websocket():
    response = client.post(
        "/sensor/imu",
        json={
            "timestamp": "NaN",
            "ax": 0.0, "ay": 0.0, "az": 9.80665,
            "gx": 0.0, "gy": 0.0, "gz": 0.0,
        },
    )
    assert response.status_code == 422

    with client.websocket_connect("/ws/sensor") as websocket:
        websocket.send_json(
            {
                "type": "gnss", "timestamp": "Infinity",
                "latitude": 12.0, "longitude": 77.0,
            }
        )
        acknowledgement = websocket.receive_json()
        assert acknowledgement["ok"] is False
        assert "timestamp must be finite" in acknowledgement["error"]


def test_websocket_sensor_acknowledges_live_packets():
    with client.websocket_connect("/ws/sensor") as websocket:
        websocket.send_json(
            {
                "type": "gnss",
                "timestamp": 1.0,
                "latitude": 12.0,
                "longitude": 77.0,
                "speed": 0.0,
                "accuracy": 5.0,
            }
        )
        gnss_ack = websocket.receive_json()
        assert gnss_ack["ok"] is True
        assert gnss_ack["data"]["gnss_status"] == "CONNECTED"

        websocket.send_json(
            {
                "type": "imu",
                "timestamp": 1.01,
                "ax": 0.0,
                "ay": 0.0,
                "az": 9.80665,
                "gx": 0.0,
                "gy": 0.0,
                "gz": 0.0,
            }
        )
        imu_ack = websocket.receive_json()
        assert imu_ack["ok"] is True
        assert imu_ack["data"]["imu_status"] == "ACTIVE"


def test_dashboard_and_track_are_available():
    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert "INTELLIGENT DEAD RECKONING" in dashboard.text

    client.post(
        "/sensor/gnss",
        json={"timestamp": 1.0, "latitude": 12.0, "longitude": 77.0, "speed": 8.0},
    )
    state = client.get("/navigation/state")
    assert state.status_code == 200
    assert isinstance(state.json()["track"], list)


# ──────────────────────────────────────────────────────────────────────────────
# NEW MVP LIVE PIPELINE TESTS: blackout gate, telemetry, session logging
# ──────────────────────────────────────────────────────────────────────────────

def test_live_telemetry_endpoint_returns_connection_and_nav_state():
    """GET /api/live_telemetry must return telemetry + navigation dicts."""
    r = client.get("/api/live_telemetry")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    tel = d["telemetry"]
    assert "ws_clients_connected" in tel
    assert "imu_rate_hz" in tel
    assert "gnss_rate_hz" in tel
    assert "gnss_blackout_active" in tel
    assert "total_imu_events" in tel
    assert "total_gnss_events" in tel
    assert "navigation" in d


def test_gnss_blackout_start_stop_suppresses_gnss_from_engine():
    """
    When blackout is active, POST /sensor/gnss must NOT reach the engine,
    so gnss_state remains INS_DEAD_RECKONING (not GNSS_AIDED).
    """
    from app import live_telemetry
    live_telemetry.stop_blackout()   # Ensure clean state

    # Establish position first
    client.post("/sensor/gnss", json={
        "timestamp": 100.0, "latitude": 12.0, "longitude": 77.0, "speed": 5.0, "accuracy": 4.0
    })
    client.post("/sensor/gnss", json={
        "timestamp": 100.5, "latitude": 12.0, "longitude": 77.0001, "speed": 5.0, "accuracy": 4.0
    })

    # Activate blackout
    r_start = client.post("/api/blackout/start", json={})
    assert r_start.status_code == 200
    assert r_start.json()["blackout_active"] is True

    # GNSS during blackout must be suppressed
    sup_before = client.get("/api/live_telemetry").json()["telemetry"]["total_gnss_suppressed"]

    # Send a GNSS packet — it should be suppressed
    g = client.post("/sensor/gnss", json={
        "timestamp": 150.0, "latitude": 12.0, "longitude": 77.0002, "speed": 5.0, "accuracy": 4.0
    })
    assert g.status_code == 200   # Returns engine state, not error

    sup_after = client.get("/api/live_telemetry").json()["telemetry"]["total_gnss_suppressed"]
    assert sup_after > sup_before, "Suppression counter must have incremented"

    # Stop blackout
    r_stop = client.post("/api/blackout/stop")
    assert r_stop.status_code == 200
    assert r_stop.json()["blackout_active"] is False

    # GNSS now reaches engine
    r_gnss = client.post("/sensor/gnss", json={
        "timestamp": 160.0, "latitude": 12.0, "longitude": 77.0003, "speed": 5.0, "accuracy": 4.0
    })
    assert r_gnss.status_code == 200
    assert r_gnss.json()["gnss_status"] in ("CONNECTED", "DEGRADED")


def test_blackout_status_endpoint():
    """GET /api/blackout/status returns blackout state."""
    from app import live_telemetry
    live_telemetry.stop_blackout()

    r = client.get("/api/blackout/status")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert d["blackout_active"] is False

    live_telemetry.start_blackout()
    r2 = client.get("/api/blackout/status")
    assert r2.json()["blackout_active"] is True
    live_telemetry.stop_blackout()


def test_websocket_blackout_control_via_event_type():
    """Blackout can be activated/deactivated through the /ws/sensor event protocol."""
    from app import live_telemetry
    live_telemetry.stop_blackout()

    with client.websocket_connect("/ws/sensor") as ws:
        ws.send_json({"type": "blackout_start"})
        ack = ws.receive_json()
        assert ack["ok"] is True
        assert ack["data"]["blackout"] is True

        ws.send_json({"type": "blackout_stop"})
        ack2 = ws.receive_json()
        assert ack2["ok"] is True
        assert ack2["data"]["blackout"] is False


def test_imu_events_increment_telemetry_counter():
    """IMU events sent via HTTP must increment the telemetry total_imu_events counter."""
    from app import live_telemetry
    before = live_telemetry.total_imu

    client.post("/sensor/imu", json={
        "timestamp": 200.0,
        "ax": 0.0, "ay": 0.0, "az": 9.80665,
        "gx": 0.0, "gy": 0.0, "gz": 0.0,
    })

    assert live_telemetry.total_imu > before


def test_gnss_events_increment_telemetry_counter_when_not_blocked():
    """GNSS events sent via HTTP must increment gnss counter when blackout is off."""
    from app import live_telemetry
    live_telemetry.stop_blackout()
    before = live_telemetry.total_gnss

    client.post("/sensor/gnss", json={
        "timestamp": 210.0, "latitude": 12.0, "longitude": 77.0, "accuracy": 5.0
    })

    assert live_telemetry.total_gnss > before


def test_session_log_endpoints():
    """Session log can be started and stopped via REST API."""
    import tempfile, os
    from app import live_telemetry

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        tmp_path = f.name

    try:
        r_start = client.post(f"/api/session/start?path={tmp_path}")
        assert r_start.status_code == 200

        # Send an event — should be logged
        client.post("/sensor/imu", json={
            "timestamp": 300.0,
            "ax": 0.1, "ay": 0.0, "az": 9.80665,
            "gx": 0.0, "gy": 0.0, "gz": 0.0,
        })

        r_stop = client.post("/api/session/stop")
        assert r_stop.status_code == 200

        # File should have at least one line
        content = open(tmp_path, encoding="utf-8").read()
        assert len(content) > 0
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
