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
