from fastapi.testclient import TestClient

from app import app, engine


client = TestClient(app)


def setup_function():
    engine.reset()


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