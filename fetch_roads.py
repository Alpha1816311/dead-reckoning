#!/usr/bin/env python3
"""
fetch_roads.py — Download real OSM road data for any location.

Queries the free Overpass API for driveable roads within a bounding box
and writes them to Data/roads.geojson in the exact LineString format that
map_matching.load_roads_from_geojson() and VehicleMapMatcher expect.

GeoJSON coordinate order is always [longitude, latitude] per RFC 7946.
The Leaflet front-end and NavigationEngine._xy_to_ll() both handle this
correctly — do NOT swap the order here.

USAGE
-----
# By centre point + radius (recommended):
    python fetch_roads.py --lat 12.9716 --lon 77.5946 --radius 1500

# By explicit bounding box (south, west, north, east):
    python fetch_roads.py --bbox 12.96 77.58 12.98 77.61

# Save to a different file:
    python fetch_roads.py --lat 12.9716 --lon 77.5946 --radius 1000 --out Data/roads.geojson

# Dry-run (just print the Overpass query, do not download):
    python fetch_roads.py --lat 12.9716 --lon 77.5946 --dry-run

DEPENDENCIES
------------
Only Python standard library + requests:
    pip install requests

ROAD TYPES FETCHED
------------------
motorway, trunk, primary, secondary, tertiary, unclassified, residential,
service, living_street, road  (all standard OSM driveable highway tags)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# OSM highway tags that represent driveable roads for vehicle navigation
# ---------------------------------------------------------------------------
DRIVEABLE_HIGHWAY_TAGS = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "service", "living_street", "road",
    "motorway_link", "trunk_link", "primary_link", "secondary_link",
    "tertiary_link",
]


def _bbox_from_centre(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """Return (south, west, north, east) bounding box from centre + radius in metres."""
    lat_deg = radius_m / 111_320.0
    lon_deg = radius_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return lat - lat_deg, lon - lon_deg, lat + lat_deg, lon + lon_deg


def _build_overpass_query(bbox: tuple[float, float, float, float]) -> str:
    south, west, north, east = bbox
    bbox_str = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"
    tag_filter = "|".join(DRIVEABLE_HIGHWAY_TAGS)
    return (
        f'[out:json][timeout:60];\n'
        f'way["highway"~"^({tag_filter})$"]({bbox_str});\n'
        f'(._;>;);\n'
        f'out body;'
    )


def _overpass_to_geojson(data: dict) -> dict:
    """Convert Overpass JSON response to a GeoJSON FeatureCollection.

    GeoJSON coordinates: [longitude, latitude] — RFC 7946 standard.
    This matches what load_roads_from_geojson() iterates as (lon, lat).
    """
    # Build node lookup: node_id → (lon, lat)
    nodes: dict[int, tuple[float, float]] = {}
    for element in data.get("elements", []):
        if element["type"] == "node":
            nodes[element["id"]] = (element["lon"], element["lat"])

    features = []
    for element in data.get("elements", []):
        if element["type"] != "way":
            continue
        node_refs = element.get("nodes", [])
        # Build coordinate list [lon, lat] — GeoJSON standard
        coords = [nodes[n] for n in node_refs if n in nodes]
        if len(coords) < 2:
            continue  # degenerate segment — skip
        tags = element.get("tags", {})
        props = {
            "id": str(element["id"]),
            "name": tags.get("name", ""),
            "highway": tags.get("highway", ""),
            "oneway": tags.get("oneway", "no"),
            "maxspeed": tags.get("maxspeed", ""),
        }
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "LineString",
                "coordinates": coords,  # [lon, lat] per RFC 7946
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def fetch(
    bbox: tuple[float, float, float, float],
    output_path: str | Path = "Data/roads.geojson",
    dry_run: bool = False,
) -> Path | None:
    """Download roads and write GeoJSON.  Returns the output Path, or None on dry-run."""
    query = _build_overpass_query(bbox)
    south, west, north, east = bbox

    print(f"Bounding box  : lat {south:.5f}–{north:.5f}, lon {west:.5f}–{east:.5f}")
    print(f"Road types    : {len(DRIVEABLE_HIGHWAY_TAGS)} driveable highway classes")

    if dry_run:
        print("\n--- Overpass query (dry-run, not sent) ---")
        print(query)
        return None

    try:
        import requests
    except ImportError:
        print("ERROR: 'requests' is not installed.  Run: pip install requests", file=sys.stderr)
        sys.exit(1)

    _OVERPASS_MIRRORS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]
    headers = {
        "User-Agent": "IDR-NavigationEngine/1.0 (dead-reckoning research)",
        "Accept": "application/json",
    }
    resp = None
    for overpass_url in _OVERPASS_MIRRORS:
        print(f"Querying      : {overpass_url}")
        try:
            resp = requests.post(overpass_url, data={"data": query}, headers=headers, timeout=90)
            resp.raise_for_status()
            break  # success
        except Exception as exc:
            print(f"  → failed ({exc}), trying next mirror…")
            resp = None
    if resp is None:
        print("ERROR: all Overpass mirrors failed. Try again later.", file=sys.stderr)
        sys.exit(1)

    raw = resp.json()
    ways = sum(1 for e in raw.get("elements", []) if e["type"] == "way")
    nodes = sum(1 for e in raw.get("elements", []) if e["type"] == "node")
    print(f"Response      : {ways} ways, {nodes} nodes")

    geojson = _overpass_to_geojson(raw)
    n_features = len(geojson["features"])
    print(f"LineStrings   : {n_features}")

    if n_features == 0:
        print("WARNING: no roads found for this area — check the bounding box.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(geojson, f, separators=(",", ":"))  # compact, no indent

    size_kb = out.stat().st_size / 1024
    print(f"Saved         : {out}  ({size_kb:.1f} KB, {n_features} features)")
    print()
    print("Next step: restart the backend and check /navigation/state for")
    print('  "map_status": "READY"  and  "map_out_of_bounds": false')
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OSM roads for a location and save as Data/roads.geojson.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--lat", type=float, metavar="LAT",
        help="Centre latitude (requires --lon and --radius).",
    )
    group.add_argument(
        "--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"),
        help="Explicit bounding box: south west north east.",
    )
    parser.add_argument("--lon", type=float, metavar="LON")
    parser.add_argument(
        "--radius", type=float, default=1000.0, metavar="METRES",
        help="Radius in metres around --lat/--lon (default: 1000).",
    )
    parser.add_argument(
        "--out", default="Data/roads.geojson", metavar="PATH",
        help="Output GeoJSON path (default: Data/roads.geojson).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the Overpass query without downloading.",
    )
    args = parser.parse_args()

    if args.bbox:
        bbox = tuple(args.bbox)  # (south, west, north, east)
    else:
        if args.lon is None:
            parser.error("--lat requires --lon")
        bbox = _bbox_from_centre(args.lat, args.lon, args.radius)

    fetch(bbox, output_path=args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
