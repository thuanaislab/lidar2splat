#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
import sys
from typing import Optional, Tuple, Dict, Any


def read_origin_lat_lon(origin_file_path: str) -> Tuple[float, float]:
    with open(origin_file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    # Accept formats like "lat, lon" or "lat lon"
    parts = re.split(r"[\s,]+", first_line)
    if len(parts) < 2:
        raise ValueError(f"Could not parse lat/lon from origin file: {origin_file_path}")
    lat = float(parts[0])
    lon = float(parts[1])
    return lat, lon


def parse_tile_coordinates_from_filename(file_path: str) -> Optional[Tuple[float, float]]:
    """
    Attempt to parse easting_northing from filenames like 717500_4224900.laz
    Returns (easting, northing) if detected, else None.
    """
    base = os.path.basename(file_path)
    name, _ext = os.path.splitext(base)
    m = re.match(r"^(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$", name)
    if m:
        try:
            easting = float(m.group(1))
            northing = float(m.group(2))
            return easting, northing
        except Exception:
            return None
    return None


def get_laz_header_via_laspy(file_path: str) -> Optional[Dict[str, Any]]:
    try:
        import laspy  # type: ignore
    except Exception:
        return None

    try:
        with laspy.open(file_path) as laz:
            header = laz.header
            try:
                crs = header.parse_crs()
            except Exception:
                crs = None

            try:
                point_count = header.point_count
            except Exception:
                point_count = None

            info: Dict[str, Any] = {
                "source": "laspy",
                "point_count": point_count,
                "scales": getattr(header, "scales", None),
                "offsets": getattr(header, "offsets", None),
                "mins": getattr(header, "mins", None),
                "maxs": getattr(header, "maxs", None),
                "version": str(getattr(header, "version", "")),
                "point_format": str(getattr(header, "point_format", "")),
                "crs": None,
                "crs_epsg": None,
                "crs_name": None,
                "crs_is_projected": None,
                "crs_units": None,
            }

            if crs is not None:
                info["crs"] = str(crs)
                try:
                    info["crs_epsg"] = crs.to_epsg()
                except Exception:
                    info["crs_epsg"] = None
                try:
                    info["crs_name"] = crs.name
                except Exception:
                    info["crs_name"] = None
                try:
                    info["crs_is_projected"] = crs.is_projected
                except Exception:
                    info["crs_is_projected"] = None
                try:
                    axis_info = crs.axis_info if hasattr(crs, "axis_info") else []
                    units = sorted({getattr(a, "unit_name", None) for a in axis_info if getattr(a, "unit_name", None)})
                    info["crs_units"] = units or None
                except Exception:
                    info["crs_units"] = None

            return info
    except Exception as e:
        return {"source": "laspy", "error": str(e)}


def get_laz_info_via_pdal(file_path: str) -> Optional[Dict[str, Any]]:
    import subprocess
    try:
        completed = subprocess.run(
            ["pdal", "info", "--summary", "--metadata", file_path],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return None
        # PDAL outputs JSON
        data = json.loads(completed.stdout)
        # Extract some useful bits
        summary = data.get("summary", {})
        metadata = data.get("metadata", {})
        srs = metadata.get("srs", {})
        out = {
            "source": "pdal",
            "summary": summary,
            "srs": srs,
        }
        # Normalize CRS info
        if isinstance(srs, dict):
            out["crs_wkt"] = srs.get("wkt")
            out["crs_proj4"] = srs.get("proj4")
            out["crs_horizontal"] = srs.get("horizontal")
        return out
    except Exception:
        return None


def compute_utm_zone(lon_deg: float) -> int:
    return int(math.floor((lon_deg + 180.0) / 6.0) + 1)


def project_latlon_to_candidates(lat: float, lon: float) -> Dict[str, Tuple[float, float]]:
    """
    Project lat/lon to plausible UTM candidates for quick comparison.
    Returns mapping name -> (easting, northing).
    Includes WGS84 UTM and NAD83 UTM for the computed zone.
    """
    try:
        from pyproj import CRS, Transformer  # type: ignore
    except Exception:
        return {}

    zone = compute_utm_zone(lon)
    is_northern = lat >= 0.0

    candidates: Dict[str, int] = {}
    # WGS84 UTM
    wgs84_epsg = 32600 + zone if is_northern else 32700 + zone
    candidates[f"WGS84_UTM_zone_{zone}{'N' if is_northern else 'S'}_{wgs84_epsg}"] = wgs84_epsg
    # NAD83 UTM (North only defined like 269xx)
    if is_northern:
        nad83_epsg = 26900 + zone
        candidates[f"NAD83_UTM_zone_{zone}N_{nad83_epsg}"] = nad83_epsg

    results: Dict[str, Tuple[float, float]] = {}

    for name, epsg in candidates.items():
        try:
            crs_src = CRS.from_epsg(4326)
            crs_dst = CRS.from_epsg(epsg)
            transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)
            easting, northing = transformer.transform(lon, lat)
            results[name] = (easting, northing)
        except Exception:
            continue

    return results


def format_number(n: Optional[float]) -> str:
    if n is None:
        return "-"
    try:
        if abs(n) >= 1000:
            return f"{n:,.3f}"
        return f"{n:.6f}"
    except Exception:
        return str(n)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect LiDAR (LAS/LAZ) file CRS, bounds, and compare against origin lat/lon.")
    parser.add_argument("--file", required=True, help="Path to LAS/LAZ file")
    parser.add_argument("--origin", default=os.path.join(os.getcwd(), "SFM", "origin_point.txt"), help="Path to origin_point.txt containing 'lat, lon'")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")
    args = parser.parse_args()

    laz_path = os.path.abspath(args.file)
    origin_path = os.path.abspath(args.origin)

    if not os.path.exists(laz_path):
        print(f"ERROR: File not found: {laz_path}", file=sys.stderr)
        return 2
    if not os.path.exists(origin_path):
        print(f"ERROR: Origin file not found: {origin_path}", file=sys.stderr)
        return 2

    try:
        lat, lon = read_origin_lat_lon(origin_path)
    except Exception as e:
        print(f"ERROR: Failed to read origin lat/lon: {e}", file=sys.stderr)
        return 2

    filename_tile_xy = parse_tile_coordinates_from_filename(laz_path)

    laspy_info = get_laz_header_via_laspy(laz_path)
    pdal_info = None if laspy_info and laspy_info.get("error") else get_laz_info_via_pdal(laz_path)

    utm_candidates = project_latlon_to_candidates(lat, lon)

    report: Dict[str, Any] = {
        "file": laz_path,
        "origin_file": origin_path,
        "origin_lat": lat,
        "origin_lon": lon,
        "filename_tile_xy": filename_tile_xy,
        "laspy": laspy_info,
        "pdal": pdal_info,
        "utm_candidates_from_origin": utm_candidates,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    # Human readable output
    print("LiDAR file inspection")
    print("----------------------")
    print(f"File: {laz_path}")
    print(f"Origin (lat, lon): {format_number(lat)}, {format_number(lon)}")

    if filename_tile_xy:
        print(f"Filename tile XY: E={format_number(filename_tile_xy[0])}, N={format_number(filename_tile_xy[1])}")

    if laspy_info is not None:
        if "error" in laspy_info and laspy_info["error"]:
            print(f"laspy: ERROR {laspy_info['error']}")
        else:
            print("laspy header:")
            print(f"  Point count: {laspy_info.get('point_count')}")
            mins = laspy_info.get("mins")
            maxs = laspy_info.get("maxs")
            if mins is not None and maxs is not None:
                try:
                    print(f"  Bounds X: {format_number(mins[0])} .. {format_number(maxs[0])}")
                    print(f"  Bounds Y: {format_number(mins[1])} .. {format_number(maxs[1])}")
                    print(f"  Bounds Z: {format_number(mins[2])} .. {format_number(maxs[2])}")
                except Exception:
                    pass
            if laspy_info.get("crs_epsg") or laspy_info.get("crs_name"):
                print("  CRS (laspy):")
                print(f"    EPSG: {laspy_info.get('crs_epsg')}")
                print(f"    Name: {laspy_info.get('crs_name')}")
                print(f"    Projected: {laspy_info.get('crs_is_projected')}")
                print(f"    Units: {laspy_info.get('crs_units')}")
            else:
                print("  CRS (laspy): not embedded or could not be parsed")
    else:
        print("laspy not available (install with: pip install 'laspy[lazrs]')")

    if pdal_info is not None:
        print("PDAL info:")
        srs = pdal_info.get("srs") or {}
        if srs:
            print(f"  Has SRS: yes")
            if "horizontal" in srs and isinstance(srs["horizontal"], dict):
                horiz = srs["horizontal"]
                auth = horiz.get("authority")
                code = horiz.get("code")
                name = horiz.get("name")
                print(f"  Horizontal CRS: {auth}:{code} {name}")
            elif srs.get("proj4"):
                print(f"  PROJ4: {srs.get('proj4')}")
            elif srs.get("wkt"):
                print("  WKT available")
        else:
            print("  No SRS in metadata")
    else:
        print("PDAL not available or could not read metadata (optional). Install PDAL CLI for richer CRS details.")

    if utm_candidates:
        print("UTM candidates from origin:")
        for name, (e, n) in utm_candidates.items():
            print(f"  {name}: E={format_number(e)}, N={format_number(n)}")
        if filename_tile_xy:
            print("Comparison to filename tile:")
            fe, fn = filename_tile_xy
            for name, (e, n) in utm_candidates.items():
                de = e - fe
                dn = n - fn
                dist = math.hypot(de, dn)
                print(f"  {name}: dE={format_number(de)}, dN={format_number(dn)}, dist={format_number(dist)}")

    # Heuristic note
    if laspy_info and laspy_info.get("mins") is not None and laspy_info.get("maxs") is not None:
        mins = laspy_info.get("mins")
        maxs = laspy_info.get("maxs")
        try:
            x_mid = 0.5 * (mins[0] + maxs[0])
            y_mid = 0.5 * (mins[1] + maxs[1])
            if abs(x_mid) > 10000 and abs(y_mid) > 1000000:
                print("Heuristic: Coordinates look like projected meters (possibly UTM).")
            else:
                print("Heuristic: Coordinates may be geographic (degrees) or a local system.")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main()) 