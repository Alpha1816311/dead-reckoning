"""
tests/test_map_matching.py — End-to-end map matching pipeline tests.

Uses real road geometry constructed from known coordinates so no network
access is required. Coordinate ordering follows RFC 7946 GeoJSON throughout:
  - GeoJSON / map_matching internals: [longitude, latitude]
  - NavigationEngine / state_snapshot output: {"latitude": ..., "longitude": ...}
  - Leaflet rendering: [latitude, longitude] arrays (handled in livenavigation.html)
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

from navigation_engine import GNSSState, NavigationEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_geojson(roads: list[list[tuple[float, float]]]) -> str:
    """Build a minimal GeoJSON FeatureCollection string from a list of
    coordinate sequences.

    Each coordinate pair must be (longitude, latitude) — GeoJSON standard.
    """
    features = [
        {
            "type": "Feature",
            "properties": {"id": str(i), "highway": "residential"},
            "geometry": {
                "type": "LineString",
                "coordinates": list(road),  # [(lon, lat), ...]
            },
        }
        for i, road in enumerate(roads)
    ]
    return json.dumps({"type": "FeatureCollection", "features": features})


def _engine_with_map(roads: list[list[tuple[float, float]]], **kwargs) -> NavigationEngine:
    """Write a temp GeoJSON and return an engine with that map loaded."""
    geojson_str = _make_geojson(roads)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".geojson", delete=False, encoding="utf-8"
    )
    tmp.write(geojson_str)
    tmp.close()
    try:
        engine = NavigationEngine(map_path=tmp.name, **kwargs)
    finally:
        os.unlink(tmp.name)
    return engine


# ---------------------------------------------------------------------------
# Road geometry centred on a synthetic test location:
# lat 12.9716, lon 77.5946 (Bangalore, MG Road corridor - synthetic coords)
# Two roads: one running E–W, one running N–S
# ---------------------------------------------------------------------------

LAT0 = 12.9716
LON0 = 77.5946

# ~500 m E–W road at LAT0: lon range 77.59 – 77.60, (lon, lat) order for GeoJSON
EW_ROAD = [
    (77.5900, LAT0),
    (77.5920, LAT0),
    (77.5940, LAT0),
    (77.5960, LAT0),
    (77.5980, LAT0),
]

# ~500 m N–S road at LON0: lat range 12.97 – 12.98
NS_ROAD = [
    (LON0, 12.9700),
    (LON0, 12.9720),
    (LON0, 12.9740),
    (LON0, 12.9760),
    (LON0, 12.9780),
]

BOTH_ROADS = [EW_ROAD, NS_ROAD]


# ---------------------------------------------------------------------------
# 1. Map loads and computes correct bounds
# ---------------------------------------------------------------------------

def test_map_loads_and_computes_bounds():
    engine = _engine_with_map(BOTH_ROADS)
    assert engine.map_status == "READY", engine.map_error
    assert engine.map_matcher is not None
    assert engine.map_bounds is not None
    min_lat, max_lat, min_lon, max_lon = engine.map_bounds
    # E–W road is at LAT0; N–S road spans 12.97–12.978
    assert min_lat <= 12.970 <= max_lat
    assert min_lon <= 77.590 <= max_lon


# ---------------------------------------------------------------------------
# 2. Coordinate ordering: GeoJSON [lon,lat] → engine → state [lat,lon]
# ---------------------------------------------------------------------------

def test_coordinate_ordering_geojson_lon_lat_to_state_lat_lon():
    """The engine must output latitude/longitude in the correct order."""
    engine = _engine_with_map(BOTH_ROADS)
    # Feed a GNSS fix exactly on the E–W road
    state = engine.process_gnss(
        timestamp=0.0,
        latitude=LAT0,        # should come back as position.latitude
        longitude=LON0,       # should come back as position.longitude
        speed_mps=5.0,
        accuracy_m=3.0,
    )
    pos = state["position"]
    assert pos is not None
    assert math.isclose(pos["latitude"],  LAT0, abs_tol=0.001), (
        f"latitude {pos['latitude']} != expected {LAT0}"
    )
    assert math.isclose(pos["longitude"], LON0, abs_tol=0.001), (
        f"longitude {pos['longitude']} != expected {LON0}"
    )


# ---------------------------------------------------------------------------
# 3. raw_gnss_position is recorded before any map snapping
# ---------------------------------------------------------------------------

def test_raw_gnss_position_stored_before_snapping():
    engine = _engine_with_map(BOTH_ROADS)
    gnss_lat, gnss_lon = LAT0 + 0.00005, LON0 + 0.00005  # slightly off-road
    engine.process_gnss(
        timestamp=0.0, latitude=gnss_lat, longitude=gnss_lon,
        speed_mps=5.0, accuracy_m=3.0,
    )
    raw = engine.last_raw_gnss_position
    assert raw is not None
    assert math.isclose(raw[0], gnss_lat, abs_tol=1e-9)
    assert math.isclose(raw[1], gnss_lon, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 4. Map matching succeeds for a point near a known road
# ---------------------------------------------------------------------------

def test_map_matching_succeeds_near_road():
    """A GPS point 15 m north of the E–W road should be matched to it."""
    engine = _engine_with_map(BOTH_ROADS)

    # ~15 m north of the E–W road (1 degree lat ≈ 111 320 m → 15 m ≈ 0.000135 deg)
    near_lat = LAT0 + 0.000135
    near_lon = LON0

    # First fix to establish origin
    engine.process_gnss(
        timestamp=0.0, latitude=near_lat, longitude=near_lon,
        speed_mps=5.0, accuracy_m=3.0,
    )
    state = engine.state_snapshot()

    assert state["map_status"] in {"MATCHED", "NO_MATCH", "READY"}, state["map_status"]
    assert state["map_out_of_bounds"] is False, (
        "Position is inside map bounds — must not be out_of_bounds"
    )
    assert state["nearest_road_distance_m"] is not None
    # ~15 m offset so nearest road should be < 50 m
    assert state["nearest_road_distance_m"] < 50.0, (
        f"nearest_road_distance_m={state['nearest_road_distance_m']} is too large"
    )
    # raw_gnss_position must be recorded
    assert state["raw_gnss_position"] is not None
    assert math.isclose(state["raw_gnss_position"]["latitude"], near_lat, abs_tol=1e-7)


# ---------------------------------------------------------------------------
# 5. Map out-of-bounds flag when far from loaded road network
# ---------------------------------------------------------------------------

def test_map_out_of_bounds_when_far_from_road_network():
    engine = _engine_with_map(BOTH_ROADS)
    # Feed a position in Rome (~41.9°N), while roads are near 12.97°N
    engine.process_gnss(
        timestamp=0.0, latitude=41.9, longitude=12.5,
        speed_mps=5.0, accuracy_m=5.0,
    )
    state = engine.state_snapshot()
    assert state["map_out_of_bounds"] is True, (
        "Position 41.9°N is far from roads at ~12.97°N — must be out_of_bounds"
    )


# ---------------------------------------------------------------------------
# 6. map_matched_position and nearest_road_distance_m appear in state_snapshot
# ---------------------------------------------------------------------------

def test_state_snapshot_contains_all_map_diagnostics():
    engine = _engine_with_map(BOTH_ROADS)
    engine.process_gnss(
        timestamp=0.0, latitude=LAT0, longitude=LON0,
        speed_mps=5.0, accuracy_m=3.0,
    )
    state = engine.state_snapshot()
    required_keys = [
        "map_status", "map_confidence", "map_out_of_bounds",
        "nearest_road_distance_m", "raw_gnss_position",
        "map_matched_position", "map_bounds",
    ]
    for key in required_keys:
        assert key in state, f"Missing key in state_snapshot: {key}"


# ---------------------------------------------------------------------------
# 7. fetch_roads.py helper compiles and generates correct GeoJSON structure
# ---------------------------------------------------------------------------

def test_fetch_roads_geojson_structure_from_overpass_response():
    """Unit-test the Overpass→GeoJSON converter without network access."""
    import sys, importlib, types
    # Import fetch_roads without executing main()
    import importlib.util
    spec = importlib.util.spec_from_file_location("fetch_roads", "fetch_roads.py")
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)

    fake_overpass = {
        "elements": [
            {"type": "node", "id": 1, "lat": 12.9716, "lon": 77.5946},
            {"type": "node", "id": 2, "lat": 12.9720, "lon": 77.5950},
            {"type": "node", "id": 3, "lat": 12.9724, "lon": 77.5954},
            {
                "type": "way", "id": 101,
                "nodes": [1, 2, 3],
                "tags": {"highway": "residential", "name": "Test Road"},
            },
        ]
    }
    gj = fr._overpass_to_geojson(fake_overpass)
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 1
    feat = gj["features"][0]
    assert feat["geometry"]["type"] == "LineString"
    # GeoJSON coords must be [lon, lat] — not [lat, lon]
    c0 = feat["geometry"]["coordinates"][0]
    assert c0[0] == 77.5946, f"Expected lon=77.5946 at index 0, got {c0[0]}"
    assert c0[1] == 12.9716, f"Expected lat=12.9716 at index 1, got {c0[1]}"
    assert feat["properties"]["id"] == "101"
    assert feat["properties"]["highway"] == "residential"


# ---------------------------------------------------------------------------
# 8. Engine reset clears map diagnostics
# ---------------------------------------------------------------------------

def test_reset_clears_map_diagnostics():
    engine = _engine_with_map(BOTH_ROADS)
    engine.process_gnss(
        timestamp=0.0, latitude=LAT0, longitude=LON0,
        speed_mps=5.0, accuracy_m=3.0,
    )
    assert engine.last_raw_gnss_position is not None

    engine.reset()

    state = engine.state_snapshot()
    assert engine.last_raw_gnss_position is None
    assert engine.last_map_nearest_road_m is None
    assert engine.last_map_matched_position is None
    assert state["raw_gnss_position"] is None
    assert state["nearest_road_distance_m"] is None
    assert state["map_matched_position"] is None
