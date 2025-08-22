#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Tuple


def apply_transform_to_point(
    x: float, y: float, z: float, transform_json_path: str
) -> Tuple[float, float, float]:
    with open(transform_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = data.get("matrix_4x4")
    if m is None:
        raise ValueError("matrix_4x4 not found in transform JSON")

    # Homogeneous multiply
    X = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]
    Y = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]
    Z = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]
    return X, Y, Z


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply saved LiDAR→local transform to a single UTM point"
    )
    parser.add_argument("--x", type=float, required=True, help="LiDAR UTM X/Easting")
    parser.add_argument("--y", type=float, required=True, help="LiDAR UTM Y/Northing")
    parser.add_argument("--z", type=float, required=True, help="LiDAR Z")
    parser.add_argument(
        "--transform-json",
        required=True,
        help="Path to JSON created by align_lidar_z.py",
    )

    args = parser.parse_args()

    X, Y, Z = apply_transform_to_point(args.x, args.y, args.z, args.transform_json)
    print(f"local_x local_y local_z: {X:.3f} {Y:.3f} {Z:.3f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2) 