#!/usr/bin/env python3
"""
3DEP -> contour lines pipeline for Dew Diligent (version 2).

WHAT CHANGED FROM VERSION 1
Version 1 asked USGS's tile-listing service (TNM Access) for 1-meter elevation tiles. For the
Gulfport test area that service returned zero tiles, so the job stopped. Version 2 instead asks
USGS's 3DEP elevation ImageServer for just the rectangle we want, already clipped, as a single
GeoTIFF. That service uses 1-meter data where it exists and coarser data where it doesn't, and
it avoids downloading huge 10 km x 10 km tiles.

TWO DIFFERENT "METERS" (easy to mix up)
- The elevation data has a 1-meter GRID: each pixel covers 1 m x 1 m of ground (--cell-m).
- The CONTOUR INTERVAL is the vertical spacing between lines: 1 FOOT by default (--interval-ft).
  USGS stores elevation in meters, so the script converts: 1 ft = 0.3048 m.

IMPORTANT HONESTY NOTE
Where USGS has no 1-meter data, this service quietly fills in with coarser data (about 10 m).
Contour lines drawn at 1-foot spacing from 10 m data look precise but are NOT. The script prints
what the service says about its source data so it can be checked in the log. Treat any area that
isn't backed by 1-meter lidar as unreliable at 1-foot spacing.

DEPENDENCIES
    pip install requests
    GDAL command-line tools on the host (gdal_contour) -- installed via apt-get in the workflow.

USAGE
    python3 3dep_contours.py --bbox -89.10 30.37 -89.08 30.39 --interval-ft 1 --out contours.geojson
"""

import argparse
import json
import math
import os
import subprocess
import sys

import requests

IMAGE_SERVER = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
FEET_PER_METER = 3.28084
NODATA = -9999
MAX_PIXELS_PER_SIDE = 4000  # stay well under the service's per-request size limit
HEADERS = {"User-Agent": "dew-diligent-contours/2.0"}


def pixel_size_for(bbox, cell_m):
    """How many pixels wide/tall the bbox is at the requested cell size (in meters)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2.0
    width_m = (max_lon - min_lon) * 111320.0 * math.cos(math.radians(mid_lat))
    height_m = (max_lat - min_lat) * 110574.0
    return max(1, round(width_m / cell_m)), max(1, round(height_m / cell_m))


def describe_source(bbox):
    """
    Ask the service what data sits under the middle of the area, and print it, so the log shows
    whether we're really looking at 1-meter lidar or something coarser. Diagnostic only: any
    failure here is reported but does not stop the job.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    params = {
        "geometry": f"{(min_lon + max_lon) / 2},{(min_lat + max_lat) / 2}",
        "geometryType": "esriGeometryPoint",
        "sr": 4326,
        "returnGeometry": "false",
        "returnCatalogItems": "true",
        "f": "json",
    }
    try:
        resp = requests.get(f"{IMAGE_SERVER}/identify", params=params, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        items = (data.get("catalogItems") or {}).get("features", [])
        print(f"  source check: {len(items)} source dataset(s) under the area center", file=sys.stderr)
        for item in items[:5]:
            print("   - " + json.dumps(item.get("attributes", {}))[:400], file=sys.stderr)
        print(f"  elevation value at center (meters): {data.get('value')}", file=sys.stderr)
    except Exception as err:  # diagnostic only
        print(f"  (source check skipped: {err})", file=sys.stderr)


def export_dem(bbox, cell_m, out_path):
    """Download a clipped GeoTIFF of elevation (meters, float) for the bbox in lon/lat."""
    width_px, height_px = pixel_size_for(bbox, cell_m)
    if width_px > MAX_PIXELS_PER_SIDE or height_px > MAX_PIXELS_PER_SIDE:
        raise RuntimeError(
            f"Area is {width_px} x {height_px} pixels at {cell_m} m; the limit per request is "
            f"{MAX_PIXELS_PER_SIDE}. Use a smaller --bbox (about 4 km per side at 1 m). "
            "Covering a whole county means splitting it into pieces, which is the next step."
        )
    min_lon, min_lat, max_lon, max_lat = bbox
    params = {
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{width_px},{height_px}",
        "format": "tiff",
        "pixelType": "F32",
        "noData": NODATA,
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    print(f"  requesting {width_px} x {height_px} pixel elevation raster...", file=sys.stderr)
    resp = requests.get(f"{IMAGE_SERVER}/exportImage", params=params, headers=HEADERS, timeout=300)
    resp.raise_for_status()
    body = resp.content
    # A GeoTIFF starts with "II*\0" or "MM\0*". Anything else is an error message in disguise.
    if body[:4] not in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError(
            "USGS did not return a GeoTIFF. What it sent instead (first 800 characters): "
            + body[:800].decode("utf-8", errors="replace")
        )
    with open(out_path, "wb") as f:
        f.write(body)
    print(f"  saved {len(body) / 1_000_000:.1f} MB to {out_path}", file=sys.stderr)
    return out_path


def generate_contours(dem_path, interval_ft, out_path, nodata=None):
    """
    Run GDAL's own `gdal_contour` (the standard, battle-tested tool) at the given interval in
    FEET, converted to meters because the elevation values are in meters. Output is GeoJSON in
    the raster's coordinate system (lon/lat here, which is what the web map expects).
    """
    interval_m = interval_ft / FEET_PER_METER
    raw_path = out_path + ".raw.geojson"
    cmd = ["gdal_contour", "-a", "elev_m", "-i", str(interval_m)]
    if nodata is not None:
        cmd += ["-snodata", str(nodata)]
    cmd += [dem_path, raw_path, "-f", "GeoJSON"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdal_contour failed: {result.stderr}")

    with open(raw_path) as f:
        geojson = json.load(f)
    os.remove(raw_path)

    # Add a feet-based elevation next to the meters value, since the app shows feet.
    for feature in geojson.get("features", []):
        elev_m = feature["properties"].get("elev_m")
        if elev_m is not None:
            feature["properties"]["elev_ft"] = round(elev_m * FEET_PER_METER, 1)

    with open(out_path, "w") as f:
        json.dump(geojson, f)
    return geojson


def main():
    parser = argparse.ArgumentParser(description="Generate precise-interval contours from USGS 3DEP elevation data.")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"), required=True)
    parser.add_argument("--interval-ft", type=float, default=1.0, help="Contour interval in FEET (default: 1)")
    parser.add_argument("--cell-m", type=float, default=1.0, help="Elevation grid size in meters (default: 1)")
    parser.add_argument("--out", default="contours.geojson", help="Output GeoJSON path")
    parser.add_argument("--workdir", default="./3dep_work", help="Scratch directory")
    args = parser.parse_args()

    bbox = tuple(args.bbox)
    os.makedirs(args.workdir, exist_ok=True)
    dem_path = os.path.join(args.workdir, "dem.tif")

    print(f"Area {bbox}", file=sys.stderr)
    describe_source(bbox)
    export_dem(bbox, args.cell_m, dem_path)

    print(f"Generating {args.interval_ft}-foot contours...", file=sys.stderr)
    geojson = generate_contours(dem_path, args.interval_ft, args.out, nodata=NODATA)
    count = len(geojson.get("features", []))
    print(f"Wrote {count} contour line(s) to {args.out}", file=sys.stderr)
    if count == 0:
        raise RuntimeError("No contour lines were produced. Check the source check lines above.")


if __name__ == "__main__":
    main()
