"""
Map Matching Module for Vehicle Dead Reckoning
================================================

Input:
    - Estimated latitude / longitude
    - Optional vehicle heading and speed
    - Road geometry

Output:
    - Map-matched latitude / longitude
    - Matched road segment
    - Lateral error
    - Heading error
    - Confidence

Designed to be independent and callable from another Python module.

Dependencies:
    pip install numpy shapely pyproj
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

import numpy as np
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from pyproj import Transformer


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class VehicleState:
    latitude: float
    longitude: float

    # Optional vehicle information
    heading: Optional[float] = None   # degrees, 0 = North
    speed: Optional[float] = None     # m/s


@dataclass
class MatchedState:
    latitude: float
    longitude: float

    road_id: Optional[str]

    lateral_error: float
    heading_error: float

    confidence: float

    # Distance along the selected road segment
    along_track: float


@dataclass
class RoadSegment:
    road_id: str
    geometry: LineString


# ============================================================
# GEOMETRY UTILITIES
# ============================================================

def wrap_angle(angle):
    """
    Wrap angle to [-180, 180].
    """
    return (angle + 180.0) % 360.0 - 180.0


def heading_difference(h1, h2):
    """
    Smallest absolute difference between two headings.
    """
    return abs(wrap_angle(h1 - h2))


def bearing_between(p1, p2):
    """
    Calculate bearing between two WGS84 points.

    Returns:
        bearing in degrees
        0 = North
        90 = East
    """

    lat1 = math.radians(p1[1])
    lat2 = math.radians(p2[1])

    dlon = math.radians(p2[0] - p1[0])

    x = math.sin(dlon) * math.cos(lat2)

    y = (
        math.cos(lat1) * math.sin(lat2)
        -
        math.sin(lat1)
        * math.cos(lat2)
        * math.cos(dlon)
    )

    bearing = math.degrees(math.atan2(x, y))

    return bearing % 360.0


# ============================================================
# MAP MATCHER
# ============================================================

class VehicleMapMatcher:

    def __init__(
        self,
        road_segments: List[RoadSegment],
        search_radius=50.0,
        heading_threshold=75.0,
        use_nonholonomic=True
    ):
        """
        Parameters
        ----------
        road_segments:
            List of RoadSegment objects.

        search_radius:
            Search radius around DR position in meters.

        heading_threshold:
            Maximum allowed heading difference.

        use_nonholonomic:
            Enable vehicle forward-motion constraints.
        """

        self.road_segments = road_segments

        self.search_radius = search_radius
        self.heading_threshold = heading_threshold
        self.use_nonholonomic = use_nonholonomic

        # ----------------------------------------------------
        # Coordinate transformation
        # WGS84 -> Web Mercator
        # ----------------------------------------------------

        self.to_xy = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:3857",
            always_xy=True
        )

        self.to_ll = Transformer.from_crs(
            "EPSG:3857",
            "EPSG:4326",
            always_xy=True
        )

        # Convert roads to projected coordinates
        self.projected_roads = []

        for road in road_segments:

            projected = self._project_linestring(
                road.geometry
            )

            self.projected_roads.append(
                RoadSegment(
                    road_id=road.road_id,
                    geometry=projected
                )
            )

        # Spatial index
        self.tree = STRtree(
            [r.geometry for r in self.projected_roads]
        )

    # ========================================================
    # COORDINATE TRANSFORMATION
    # ========================================================

    def _project_point(self, latitude, longitude):

        x, y = self.to_xy.transform(
            longitude,
            latitude
        )

        return Point(x, y)

    def _project_linestring(self, line):

        coords = []

        for lon, lat in line.coords:

            x, y = self.to_xy.transform(
                lon,
                lat
            )

            coords.append((x, y))

        return LineString(coords)

    def _unproject_point(self, point):

        lon, lat = self.to_ll.transform(
            point.x,
            point.y
        )

        return lat, lon

    # ========================================================
    # ROAD HEADING
    # ========================================================

    def _road_heading(self, road, distance):

        line = road.geometry

        total_length = line.length

        d1 = max(0.0, distance - 2.0)
        d2 = min(total_length, distance + 2.0)

        p1 = line.interpolate(d1)
        p2 = line.interpolate(d2)

        # Convert projected coordinates back to lat/lon
        lat1, lon1 = self._unproject_point(p1)
        lat2, lon2 = self._unproject_point(p2)

        return bearing_between(
            (lon1, lat1),
            (lon2, lat2)
        )

    # ========================================================
    # CANDIDATE SEARCH
    # ========================================================

    def _find_candidates(self, point):

        search_area = point.buffer(
            self.search_radius
        )

        indices = self.tree.query(
            search_area
        )

        candidates = []

        for idx in indices:

            road = self.projected_roads[idx]

            distance = point.distance(
                road.geometry
            )

            if distance <= self.search_radius:

                projection = road.geometry.project(
                    point
                )

                snapped = road.geometry.interpolate(
                    projection
                )

                candidates.append(
                    (
                        road,
                        distance,
                        projection,
                        snapped
                    )
                )

        return candidates

    # ========================================================
    # NON-HOLONOMIC CONSTRAINT
    # ========================================================

    def _passes_vehicle_constraint(
        self,
        vehicle_heading,
        road_heading
    ):
        """
        Vehicle generally travels along the road direction.

        A large heading mismatch indicates that the candidate
        road is probably incorrect.

        The reverse direction of a road is also considered.
        """

        if vehicle_heading is None:
            return True

        error = heading_difference(
            vehicle_heading,
            road_heading
        )

        # Road geometry may be digitized in the opposite
        # direction to vehicle travel.
        reverse_error = heading_difference(
            vehicle_heading,
            (road_heading + 180.0) % 360.0
        )

        best_error = min(
            error,
            reverse_error
        )

        return best_error <= self.heading_threshold

    # ========================================================
    # CANDIDATE SCORING
    # ========================================================

    def _score_candidate(
        self,
        distance,
        vehicle_heading,
        road_heading
    ):

        # ----------------------------------------------------
        # Position score
        # ----------------------------------------------------

        position_score = math.exp(
            -(distance ** 2) /
            (2 * 15.0 ** 2)
        )

        # ----------------------------------------------------
        # Heading score
        # ----------------------------------------------------

        if vehicle_heading is None:

            heading_score = 1.0
            heading_error = 0.0

        else:

            error_forward = heading_difference(
                vehicle_heading,
                road_heading
            )

            error_reverse = heading_difference(
                vehicle_heading,
                (road_heading + 180.0) % 360.0
            )

            heading_error = min(
                error_forward,
                error_reverse
            )

            heading_score = math.exp(
                -(heading_error ** 2) /
                (2 * 30.0 ** 2)
            )

        # ----------------------------------------------------
        # Combined score
        # ----------------------------------------------------

        score = (
            0.65 * position_score
            +
            0.35 * heading_score
        )

        return score, heading_error

    # ========================================================
    # SINGLE POINT MATCH
    # ========================================================

    def match(self, state: VehicleState):

        point = self._project_point(
            state.latitude,
            state.longitude
        )

        candidates = self._find_candidates(
            point
        )

        if not candidates:

            return MatchedState(
                latitude=state.latitude,
                longitude=state.longitude,
                road_id=None,
                lateral_error=float("inf"),
                heading_error=float("inf"),
                confidence=0.0,
                along_track=0.0
            )

        best = None

        for (
            road,
            distance,
            projection,
            snapped
        ) in candidates:

            along_track = road.geometry.project(
                point
            )

            road_heading = self._road_heading(
                road,
                along_track
            )

            # ------------------------------------------------
            # Vehicle non-holonomic constraint
            # ------------------------------------------------

            if self.use_nonholonomic:

                if not self._passes_vehicle_constraint(
                    state.heading,
                    road_heading
                ):
                    continue

            # ------------------------------------------------
            # Score
            # ------------------------------------------------

            score, heading_error = self._score_candidate(
                distance,
                state.heading,
                road_heading
            )

            if best is None or score > best["score"]:

                best = {
                    "road": road,
                    "distance": distance,
                    "projection": projection,
                    "snapped": snapped,
                    "score": score,
                    "heading_error": heading_error
                }

        # No candidate survived vehicle constraint
        if best is None:

            return MatchedState(
                latitude=state.latitude,
                longitude=state.longitude,
                road_id=None,
                lateral_error=float("inf"),
                heading_error=float("inf"),
                confidence=0.0,
                along_track=0.0
            )

        # ----------------------------------------------------
        # Convert snapped point back to WGS84
        # ----------------------------------------------------

        lat, lon = self._unproject_point(
            best["snapped"]
        )

        return MatchedState(
            latitude=lat,
            longitude=lon,
            road_id=best["road"].road_id,
            lateral_error=best["distance"],
            heading_error=best["heading_error"],
            confidence=best["score"],
            along_track=best["projection"]
        )

    # ========================================================
    # TRAJECTORY MATCHING
    # ========================================================

    def match_trajectory(
        self,
        trajectory: List[VehicleState]
    ):

        results = []

        previous_road = None

        for state in trajectory:

            result = self.match(state)

            # ------------------------------------------------
            # Temporal road continuity
            # ------------------------------------------------

            if (
                previous_road is not None
                and result.road_id is not None
            ):

                # Penalize sudden jumps to unrelated roads.
                if result.road_id != previous_road:

                    result.confidence *= 0.75

            if result.road_id is not None:

                previous_road = result.road_id

            results.append(result)

        return results


# ============================================================
# GEOJSON LOADER
# ============================================================

def load_roads_from_geojson(path):

    """
    Load LineString road geometry from a GeoJSON file.

    Example GeoJSON:
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "id": "road_001"
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [...]
                    }
                }
            ]
        }
    """

    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    roads = []

    for i, feature in enumerate(
        data["features"]
    ):

        geometry = feature["geometry"]

        if geometry["type"] != "LineString":
            continue

        road_id = str(
            feature.get(
                "properties",
                {}
            ).get(
                "id",
                i
            )
        )

        line = LineString(
            geometry["coordinates"]
        )

        roads.append(
            RoadSegment(
                road_id=road_id,
                geometry=line
            )
        )

    return roads


# ============================================================
# SIMPLE FUNCTION INTERFACE
# ============================================================

def map_match(
    latitude,
    longitude,
    roads,
    heading=None,
    speed=None
):
    """
    Simple interface for another Python module.

    Parameters
    ----------
    latitude:
        Estimated latitude.

    longitude:
        Estimated longitude.

    roads:
        List[RoadSegment]

    heading:
        Vehicle heading in degrees.

    speed:
        Vehicle speed in m/s.

    Returns
    -------
    MatchedState
    """

    matcher = VehicleMapMatcher(
        roads
    )

    state = VehicleState(
        latitude=latitude,
        longitude=longitude,
        heading=heading,
        speed=speed
    )

    return matcher.match(state)


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    # Example road geometry
    road1 = RoadSegment(
        road_id="road_001",
        geometry=LineString([
            (77.5900, 12.9700),
            (77.5910, 12.9710),
            (77.5920, 12.9720)
        ])
    )

    road2 = RoadSegment(
        road_id="road_002",
        geometry=LineString([
            (77.5900, 12.9720),
            (77.5910, 12.9720),
            (77.5920, 12.9720)
        ])
    )

    roads = [
        road1,
        road2
    ]

    result = map_match(
        latitude=12.9710,
        longitude=77.5912,
        roads=roads,
        heading=45.0,
        speed=12.0
    )

    print("\nMAP MATCH RESULT")
    print("---------------------------")
    print("Latitude       :", result.latitude)
    print("Longitude      :", result.longitude)
    print("Road ID        :", result.road_id)
    print("Lateral Error  :", result.lateral_error, "m")
    print("Heading Error  :", result.heading_error, "deg")
    print("Confidence     :", result.confidence)