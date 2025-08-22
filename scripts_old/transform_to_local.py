#!/usr/bin/env python3

import argparse
import os
import re
import sys
from typing import Tuple, Optional


def read_origin_lat_lon(origin_file_path: str) -> Tuple[float, float]:
    with open(origin_file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    parts = re.split(r"[\s,]+", first_line)
    if len(parts) < 2:
        raise ValueError(f"Could not parse lat/lon from origin file: {origin_file_path}")
    lat = float(parts[0])
    lon = float(parts[1])
    return lat, lon


def transform_to_local(
    laz_path: str,
    origin_path: str,
    out_path: Optional[str],
    origin_alt: Optional[float],
    keep_crs: bool,
) -> str:
    import laspy  # type: ignore
    from pyproj import Transformer  # type: ignore

    laz_path = os.path.abspath(laz_path)
    origin_path = os.path.abspath(origin_path)

    if not os.path.exists(laz_path):
        raise FileNotFoundError(laz_path)
    if not os.path.exists(origin_path):
        raise FileNotFoundError(origin_path)

    lat, lon = read_origin_lat_lon(origin_path)

    las = laspy.read(laz_path)

    crs = None
    try:
        crs = las.header.parse_crs()
    except Exception:
        crs = None

    if crs is None or not crs.is_projected:
        raise RuntimeError(
            "Input file CRS is missing or not projected. This script expects a projected CRS (e.g., UTM)."
        )

    transformer = Transformer.from_crs(4326, crs, always_xy=True)
    origin_e, origin_n = transformer.transform(lon, lat)

    # Prepare output path
    if out_path is None:
        base, ext = os.path.splitext(laz_path)
        if ext.lower() not in (".las", ".laz"):
            ext = ".laz"
        out_path = f"{base}_local{ext}"

    # Compute local coordinates
    x_local = las.x - origin_e
    y_local = las.y - origin_n
    if origin_alt is not None:
        z_local = las.z - origin_alt
    else:
        z_local = las.z

    # Update header scales/offsets for clean integer encoding
    try:
        # Keep original scales; set offsets to zero for local frame
        las.header.offsets = (0.0, 0.0, 0.0)
    except Exception:
        # Some laspy versions require list
        las.header.offsets = [0.0, 0.0, 0.0]

    # Optionally strip CRS VLRs since this is a local frame now
    if not keep_crs:
        try:
            from laspy.vlrs.known import (
                GeoKeyDirectoryVlr,
                GeoAsciiParamsVlr,
                GeoDoubleParamsVlr,
                WktCoordinateSystemVlr,
            )

            filtered_vlrs = []
            for vlr in list(las.header.vlrs):
                if isinstance(
                    vlr,
                    (
                        GeoKeyDirectoryVlr,
                        GeoAsciiParamsVlr,
                        GeoDoubleParamsVlr,
                        WktCoordinateSystemVlr,
                    ),
                ):
                    continue
                filtered_vlrs.append(vlr)
            las.header.vlrs = filtered_vlrs
        except Exception:
            # If anything fails, leave VLRs as-is
            pass

    # Assign transformed coordinates
    las.x = x_local
    las.y = y_local
    las.z = z_local

    # Write output (LAZ if extension is .laz and lazrs installed)
    las.write(out_path)

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transform LAS/LAZ to a local meter frame relative to origin_point.txt (lat, lon)."
    )
    parser.add_argument("--file", required=True, help="Path to input LAS/LAZ file")
    parser.add_argument(
        "--origin",
        default=os.path.join(os.getcwd(), "SFM", "origin_point.txt"),
        help="Path to origin_point.txt containing 'lat, lon'",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for transformed file (defaults to <input>_local.laz)",
    )
    parser.add_argument(
        "--origin-alt",
        type=float,
        default=None,
        help="Optional altitude at origin to subtract from Z (meters)",
    )
    parser.add_argument(
        "--keep-crs",
        action="store_true",
        help="Keep original CRS metadata in output (by default it is stripped, since frame becomes local)",
    )

    args = parser.parse_args()

    try:
        out_path = transform_to_local(
            laz_path=args.file,
            origin_path=args.origin,
            out_path=args.out,
            origin_alt=args.origin_alt,
            keep_crs=args.keep_crs,
        )
        print(f"Wrote local-frame file: {out_path}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main()) 