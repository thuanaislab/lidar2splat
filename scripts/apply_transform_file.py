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
    if M is None:
        raise ValueError("matrix_4x4 missing in transform JSON")
    M = np.asarray(M, dtype=float)
    if M.shape != (4, 4):
        raise ValueError("matrix_4x4 must be 4x4")
    return M


def _apply_to_ply(in_path: str, out_path: Optional[str], M: np.ndarray) -> str:
    from plyfile import PlyData  # type: ignore

    in_path = os.path.abspath(in_path)
    if out_path is None:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}_applied.ply"
    out_path = os.path.abspath(out_path)

    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    ply = PlyData.read(in_path)
    if "vertex" not in ply:
        raise RuntimeError("PLY missing 'vertex' element")
    v = ply["vertex"]
    x = np.asarray(v.data["x"], dtype=np.float64)
    y = np.asarray(v.data["y"], dtype=np.float64)
    z = np.asarray(v.data["z"], dtype=np.float64)
    P = np.column_stack((x, y, z))
    P_out = (P @ M[:3, :3].T) + M[:3, 3]

    # Preserve original dtype
    v.data["x"] = P_out[:, 0].astype(v.data.dtype["x"])
    v.data["y"] = P_out[:, 1].astype(v.data.dtype["y"])
    v.data["z"] = P_out[:, 2].astype(v.data.dtype["z"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ply.write(out_path)
    return out_path


def _apply_to_las(in_path: str, out_path: Optional[str], M: np.ndarray, keep_crs: bool) -> str:
    import laspy  # type: ignore

    in_path = os.path.abspath(in_path)
    if out_path is None:
        base, ext = os.path.splitext(in_path)
        if ext.lower() not in (".las", ".laz"):
            ext = ".laz"
        out_path = f"{base}_applied{ext}"

    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    A = M[:3, :3]
    b = M[:3, 3]

    las = laspy.read(in_path)

    P = np.column_stack((np.asarray(las.x, dtype=np.float64),
                         np.asarray(las.y, dtype=np.float64),
                         np.asarray(las.z, dtype=np.float64)))
    P_out = (P @ A.T) + b

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

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    las.write(out_path)
    return out_path


def apply_transform(in_path: str, out_path: Optional[str], transform_json: str, keep_crs: bool) -> str:
    in_path = os.path.abspath(in_path)
    transform_json = os.path.abspath(transform_json)

    if not os.path.exists(transform_json):
        raise FileNotFoundError(transform_json)

    M = _load_transform_matrix(transform_json)
    ext = os.path.splitext(in_path)[1].lower()
    if ext == ".ply":
        return _apply_to_ply(in_path, out_path, M)
    elif ext in (".las", ".laz"):
        return _apply_to_las(in_path, out_path, M, keep_crs)
    else:
        raise ValueError(f"Unsupported input extension: {ext}. Expected .las, .laz, or .ply")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply transform JSON to LAS/LAZ or PLY")
    parser.add_argument("--file", required=True)
    parser.add_argument("--transform-json", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--keep-crs", action="store_true", help="LAS/LAZ only: preserve CRS VLRs")
    args = parser.parse_args()

    try:
        out = apply_transform(args.file, args.out, args.transform_json, args.keep_crs)
        print(f"Wrote transformed file: {out}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main()) 