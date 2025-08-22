#!/usr/bin/env python3

import argparse
import json
import os
import sys
import math
from typing import Tuple, Optional

import numpy as np


def _load_transform_matrix(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    M = np.asarray(data.get("matrix_4x4"), dtype=float)
    if M.shape != (4, 4):
        raise ValueError("matrix_4x4 must be a 4x4 matrix in the transform JSON")
    A = M[:3, :3]
    b = M[:3, 3]
    return A, b


def _read_ply_xyz_all(ply_path: str) -> np.ndarray:
    from plyfile import PlyData  # type: ignore

    if not os.path.exists(ply_path):
        raise FileNotFoundError(ply_path)
    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise RuntimeError("PLY missing 'vertex' element")
    v = ply["vertex"]
    x = np.asarray(v["x"], dtype=np.float64)
    y = np.asarray(v["y"], dtype=np.float64)
    z = np.asarray(v["z"], dtype=np.float64)
    pts = np.stack([x, y, z], axis=1)
    return pts.astype(np.float32, copy=False)


def _compute_crop_bounds_from_points(points_xyz: np.ndarray, crop_frac: float,
                                     center_xy: Optional[Tuple[float, float]] = None) -> Tuple[float, float, float, float]:
    if points_xyz.shape[0] == 0:
        raise ValueError("No points provided to compute crop bounds")
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    cx = (xmin + xmax) * 0.5 if center_xy is None else float(center_xy[0])
    cy = (ymin + ymax) * 0.5 if center_xy is None else float(center_xy[1])
    # side scale so that area ratio ~= crop_frac
    side_scale = math.sqrt(float(crop_frac))
    half_w = (xmax - xmin) * 0.5 * side_scale
    half_h = (ymax - ymin) * 0.5 * side_scale
    return cx - half_w, cx + half_w, cy - half_h, cy + half_h


def _crop_points_xy(points_xyz: np.ndarray, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    if points_xyz.shape[0] == 0:
        return points_xyz
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    return points_xyz[mask]


def _transform_and_crop_las(las_path: str, A: np.ndarray, b: np.ndarray,
                            bounds: Tuple[float, float, float, float]) -> np.ndarray:
    import laspy  # type: ignore

    las_path = os.path.abspath(las_path)
    if not os.path.exists(las_path):
        raise FileNotFoundError(las_path)

    xmin, xmax, ymin, ymax = bounds
    out_chunks: list[np.ndarray] = []
    with laspy.open(las_path) as fh:
        for chunk_points in fh.chunk_iterator(2_000_000):
            h = fh.header
            xs = chunk_points.X * h.scales[0] + h.offsets[0]
            ys = chunk_points.Y * h.scales[1] + h.offsets[1]
            zs = chunk_points.Z * h.scales[2] + h.offsets[2]
            P = np.column_stack((xs, ys, zs)).astype(np.float64, copy=False)
            P_out = (P @ A.T) + b
            x = P_out[:, 0]
            y = P_out[:, 1]
            mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
            if np.any(mask):
                out_chunks.append(P_out[mask].astype(np.float32, copy=False))
    if len(out_chunks) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(out_chunks, axis=0)


def _serve(points_target: np.ndarray, points_lidar: np.ndarray, port: int, point_size: float) -> None:
    import viser  # type: ignore
    import time

    server = viser.ViserServer(port=port)

    target_colors = np.tile(np.array([[1.0, 0.15, 0.15]], dtype=np.float32), (points_target.shape[0], 1))
    lidar_colors = np.tile(np.array([[0.2, 0.4, 1.0]], dtype=np.float32), (points_lidar.shape[0], 1))

    server.add_point_cloud(name="target_red", points=points_target, colors=target_colors, point_size=point_size)
    server.add_point_cloud(name="lidar_blue", points=points_lidar, colors=lidar_colors, point_size=point_size)

    print(f"Viser running on http://localhost:{port} (use SSH port-forwarding if remote)")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def _parse_center(center_str: Optional[str]) -> Optional[Tuple[float, float]]:
    if center_str is None:
        return None
    parts = center_str.split(",")
    if len(parts) != 2:
        raise ValueError("--crop-center must be 'x,y'")
    return float(parts[0]), float(parts[1])


def _parse_bounds(bounds_str: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if bounds_str is None:
        return None
    parts = bounds_str.split(",")
    if len(parts) != 4:
        raise ValueError("--crop-bounds must be 'xmin,xmax,ymin,ymax'")
    xmin, xmax, ymin, ymax = map(float, parts)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid crop bounds: ensure xmin<xmax and ymin<ymax")
    return xmin, xmax, ymin, ymax


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize target (red) vs transformed LiDAR (blue) with Viser (cropped)")
    parser.add_argument("--file", required=True, help="Path to input LAS/LAZ")
    parser.add_argument("--transform-json", required=True, help="Transform JSON with 4x4 matrix_4x4")
    parser.add_argument("--target-ply", required=True, help="Target PLY path")
    parser.add_argument("--crop-frac", type=float, default=0.05, help="Crop area fraction (e.g., 0.05 ≈ 1/20 area)")
    parser.add_argument("--crop-center", type=str, default=None, help="Optional crop center 'x,y' in target coords")
    parser.add_argument("--crop-bounds", type=str, default=None, help="Manual bounds 'xmin,xmax,ymin,ymax' (overrides frac/center)")
    parser.add_argument("--point-size", type=float, default=0.8, help="Point size in viewer")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve Viser on")
    args = parser.parse_args()

    A, b = _load_transform_matrix(args.transform_json)
    target_all = _read_ply_xyz_all(args.target_ply)

    manual_bounds = _parse_bounds(args.crop_bounds)
    if manual_bounds is not None:
        bounds = manual_bounds
    else:
        center_xy = _parse_center(args.crop_center)
        bounds = _compute_crop_bounds_from_points(target_all, args.crop_frac, center_xy)

    target_pts = _crop_points_xy(target_all, bounds)
    lidar_pts = _transform_and_crop_las(args.file, A, b, bounds)

    if target_pts.shape[0] == 0 and lidar_pts.shape[0] == 0:
        raise RuntimeError("Crop resulted in zero points for both target and LiDAR. Consider increasing --crop-frac or changing bounds.")

    print(f"Crop bounds: xmin={bounds[0]:.3f}, xmax={bounds[1]:.3f}, ymin={bounds[2]:.3f}, ymax={bounds[3]:.3f}")
    print(f"Target points (cropped): {target_pts.shape[0]} | LiDAR points (cropped): {lidar_pts.shape[0]}")

    _serve(target_pts, lidar_pts, args.port, args.point_size)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2) 