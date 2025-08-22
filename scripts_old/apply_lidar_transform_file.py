#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np


def _load_transform_matrix(json_path: str) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    M = data.get("matrix_4x4")
    if M is not None:
        M = np.asarray(M, dtype=float)
        if M.shape != (4, 4):
            raise ValueError("matrix_4x4 must be 4x4")
        return M

    # Fallback: reconstruct from origin, scale, R, t, z_offset
    required = ("origin_e", "origin_n", "scale", "R_2x2", "t_xy", "z_offset")
    if not all(k in data for k in required):
        raise ValueError("Transform JSON missing matrix_4x4 and components to reconstruct it")
    origin_e = float(data["origin_e"]) 
    origin_n = float(data["origin_n"]) 
    s = float(data["scale"]) 
    R = np.asarray(data["R_2x2"], dtype=float)
    t = np.asarray(data["t_xy"], dtype=float)
    z_off = float(data["z_offset"]) 

    Mxy = s * R
    v = -Mxy @ np.array([origin_e, origin_n]) + t
    M = np.eye(4, dtype=float)
    M[0, 0:2] = Mxy[0]
    M[1, 0:2] = Mxy[1]
    M[0, 3] = float(v[0])
    M[1, 3] = float(v[1])
    M[2, 2] = 1.0
    M[2, 3] = float(z_off)
    return M


def apply_transform_to_las(
    in_path: str,
    out_path: Optional[str],
    transform_json: str,
    keep_crs: bool,
) -> str:
    import laspy  # type: ignore

    in_path = os.path.abspath(in_path)
    transform_json = os.path.abspath(transform_json)
    if out_path is None:
        base, ext = os.path.splitext(in_path)
        if ext.lower() not in (".las", ".laz"):
            ext = ".laz"
        out_path = f"{base}_applied{ext}"

    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)
    if not os.path.exists(transform_json):
        raise FileNotFoundError(transform_json)

    M = _load_transform_matrix(transform_json)
    A = M[:3, :3]
    b = M[:3, 3]

    las = laspy.read(in_path)

    # Transform all points
    P = np.column_stack((np.asarray(las.x, dtype=np.float64),
                         np.asarray(las.y, dtype=np.float64),
                         np.asarray(las.z, dtype=np.float64)))
    P_out = (P @ A.T) + b

    # Prepare output header for local frame
    try:
        las.header.scales = (0.001, 0.001, 0.001)
    except Exception:
        las.header.scales = [0.001, 0.001, 0.001]

    try:
        las.header.offsets = (0.0, 0.0, 0.0)
    except Exception:
        las.header.offsets = [0.0, 0.0, 0.0]

    las.x = P_out[:, 0]
    las.y = P_out[:, 1]
    las.z = P_out[:, 2]

    # Optionally strip CRS VLRs since this is a local frame now
    if not keep_crs:
        try:
            from laspy.vlrs.known import (
                GeoKeyDirectoryVlr,
                GeoAsciiParamsVlr,
                GeoDoubleParamsVlr,
                WktCoordinateSystemVlr,
            )  # type: ignore

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
            pass

    las.write(out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply a saved 4x4 transform JSON to an entire LAS/LAZ (LiDAR→target local frame)."
    )
    parser.add_argument("--file", required=True, help="Input LAS/LAZ in LiDAR CRS (e.g., UTM)")
    parser.add_argument("--transform-json", required=True, help="Transform JSON with matrix_4x4")
    parser.add_argument("--out", default=None, help="Output LAS/LAZ (defaults to <input>_applied.laz)")
    parser.add_argument("--keep-crs", action="store_true", help="Keep CRS VLRs in output")

    args = parser.parse_args()

    try:
        out = apply_transform_to_las(
            in_path=args.file,
            out_path=args.out,
            transform_json=args.transform_json,
            keep_crs=args.keep_crs,
        )
        print(f"Wrote transformed file: {out}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main()) 