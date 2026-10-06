#!/usr/bin/env python3
"""
3DEP -> contour lines pipeline for Dew Diligent.

WHAT THIS DOES
Generates precise-interval contour lines (default 1 foot) for a given bounding box, sourced
directly from USGS 3DEP 1-meter elevation data. This is the real computation that a static
HTML/JS map cannot do in-browser — it needs to run on a server, on a schedule or on demand,
with output published somewhere the app can fetch (see PRECISE_CONTOUR_TILE_URL /
AJD_DATA_URL-style hooks in the app itself).

WHY THIS BEATS CHASING STATE-BY-STATE CONTOUR PRODUCTS (e.g. MARIS's 1'/2' county set)
3DEP's baseline nationwide LiDAR acquisition is complete as of 2026 — 1-meter bare-earth
elevation, standardized, public domain, one federal source, the same everywhere. Instead of
negotiating with 50 different state GIS offices for 50 different contour products in 50
different formats, this script generates contours ourselves, consistently, from one source,
for any county or bounding box in the country.

DATA SOURCE
USGS hosts 3DEP 1-meter DEM tiles as Cloud-Optimized GeoTIFFs (COGs) in a public, no-auth-
required AWS S3 bucket:
    https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/<project>/TIFF/*.tif
The National Map's TNM Access API (https://tnmaccess.nationalmap.gov/api/v1/products) is the
practical way to discover which 1m DEM tiles cover a given bounding box, since project/tile
naming varies by region and acquisition project.

DEPENDENCIES
    pip install requests
    Plus GDAL's command-line tools on the host (gdal_contour, gdalbuildvrt) — installed via
    apt-get (see the GitHub Actions workflow) or your system's package manager, not pip.

USAGE
    python3 3dep_contours.py --bbox -89.10 30.35 -89.05 30.40 --interval-ft 1 --out contours.geojson

WHAT A REAL DEPLOYMENT NEEDS ON TOP OF THIS SCRIPT
- A scheduler (cron / GitHub Actions / AWS Lambda on a timer) to run this per-county or
  per-region and keep output current.
- A place to publish the output (S3/Cloudflare R2 + a CDN, or converted to vector tiles with
  a tool like tippecanoe for performance at scale — raw GeoJSON is fine for one county, too
  heavy for statewide 1' contours all at once).
- The app's PRECISE_CONTOUR_TILE_URL / a new fetch hook pointed at wherever that output lands.
"""

import argparse
import json
import os
import subprocess
import sys

import requests

TNM_ACCESS_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
FEET_PER_METER = 3.28084


def find_1m_dem_tiles(bbox):
    """
    Query USGS's TNM Access API for 1-meter DEM (3DEP) tiles intersecting the bounding box.
    bbox = (min_lon, min_lat, max_lon, max_lat)
    Returns a list of download URLs (COG GeoTIFFs).
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    params = {
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "datasets": "Digital Elevation Model (DEM) 1 meter",
        "max": 100,
        "outputFormat": "JSON",
    }
    headers = {"User-Agent": "dew-diligent-contours/1.0"}
    resp = requests.get(TNM_ACCESS_API, params=params, headers=headers, timeout=60)
    print(f"  request: {resp.url}", file=sys.stderr)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("items", [])
    print(f"  USGS reported total={data.get('total')}, items returned={len(items)}", file=sys.stderr)

    # Keep only the GeoTIFF files (filtering here instead of in the request, so a wrong
    # format label can't silently exclude everything).
    tif_items = [i for i in items if (i.get("downloadURL") or "").lower().endswith(".tif")]
    if not tif_items:
        raise RuntimeError(
            "No 1m 3DEP GeoTIFF tiles found for this bbox. Raw USGS response (first 1500 "
            "characters) so we can see what it actually said: " + json.dumps(data)[:1500]
        )

    # Several survey projects can overlap the same spot. Use only the newest project so we
    # don't mosaic different surveys together.
    def project_of(url):
        parts = url.split("/Projects/")
        return parts[1].split("/")[0] if len(parts) > 1 else "unknown"

    newest = max(tif_items, key=lambda i: i.get("publicationDate") or "")
    project = project_of(newest["downloadURL"])
    chosen = [i["downloadURL"] for i in tif_items if project_of(i["downloadURL"]) == project]
    print(f"  using newest project '{project}' ({len(chosen)} tile(s))", file=sys.stderr)
    return chosen


def download_tile(url, dest_dir):
    """Download a single DEM tile to dest_dir, streaming so large files don't blow up memory."""
    local_path = os.path.join(dest_dir, os.path.basename(url.split("?")[0]))
    if os.path.exists(local_path):
        return local_path  # skip re-download on local re-runs
    print(f"  downloading {url}", file=sys.stderr)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    return local_path


def mosaic_tiles(tile_paths, dest_dir):
    """Build a single virtual mosaic (VRT) from multiple DEM tiles via gdalbuildvrt."""
    vrt_path = os.path.join(dest_dir, "mosaic.vrt")
    cmd = ["gdalbuildvrt", vrt_path] + tile_paths
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdalbuildvrt failed: {result.stderr}")
    return vrt_path


def generate_contours(dem_path, interval_ft, out_path):
    """
    Run contour extraction on a DEM (or VRT mosaic) at the given interval (in feet), via
    GDAL's own `gdal_contour` utility — battle-tested, handles edge cases like flat areas
    and nodata correctly, rather than reinventing contour math from scratch.
    """
    interval_m = interval_ft / FEET_PER_METER
    raw_path = out_path + ".raw.geojson"
    cmd = [
        "gdal_contour",
        "-a", "elev_m",
        "-i", str(interval_m),
        dem_path,
        raw_path,
        "-f", "GeoJSON",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdal_contour failed: {result.stderr}")

    with open(raw_path) as f:
        geojson = json.load(f)
    os.remove(raw_path)

    # Add a feet-based elevation property alongside the meters value gdal_contour wrote,
    # since the app displays everything in feet.
    for feature in geojson.get("features", []):
        elev_m = feature["properties"].get("elev_m")
        if elev_m is not None:
            feature["properties"]["elev_ft"] = round(elev_m * FEET_PER_METER, 1)

    with open(out_path, "w") as f:
        json.dump(geojson, f)

    return geojson


def main():
    parser = argparse.ArgumentParser(description="Generate precise-interval contours from USGS 3DEP data.")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"), required=True)
    parser.add_argument("--interval-ft", type=float, default=1.0, help="Contour interval in feet (default: 1)")
    parser.add_argument("--out", default="contours.geojson", help="Output GeoJSON path")
    parser.add_argument("--workdir", default="./3dep_work", help="Scratch directory for downloaded tiles")
    args = parser.parse_args()

    bbox = tuple(args.bbox)
    os.makedirs(args.workdir, exist_ok=True)

    print(f"Finding 3DEP 1m DEM tiles for bbox {bbox}...", file=sys.stderr)
    tile_urls = find_1m_dem_tiles(bbox)
    print(f"Found {len(tile_urls)} tile(s).", file=sys.stderr)

    local_paths = [download_tile(url, args.workdir) for url in tile_urls]

    if len(local_paths) == 1:
        dem_source = local_paths[0]
    else:
        print("Mosaicking tiles...", file=sys.stderr)
        dem_source = mosaic_tiles(local_paths, args.workdir)

    print(f"Generating {args.interval_ft}' contours...", file=sys.stderr)
    geojson = generate_contours(dem_source, args.interval_ft, args.out)
    print(f"Wrote {len(geojson.get('features', []))} contour lines to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
