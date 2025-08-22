#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree


def _load_matrix(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    M = np.asarray(d["matrix_4x4"], dtype=float)
    if M.shape != (4, 4):
        raise ValueError("matrix_4x4 must be 4x4")
    A = M[:3, :3]
    b = M[:3, 3]
    return A, b


def _read_ply_xyz(ply_path: str, max_points: int = 4_000_000) -> np.ndarray:
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
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(42).choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts


def verify_alignment(las_path: str, transform_json: str, target_ply: str,
                     chunk_size: int = 2_000_000, max_samples: int = 3_000_000, nn_radius: float = 0.8) -> dict:
    import laspy  # type: ignore
    las_path = os.path.abspath(las_path)
    transform_json = os.path.abspath(transform_json)
    target_ply = os.path.abspath(target_ply)
    if not os.path.exists(las_path):
        raise FileNotFoundError(las_path)
    if not os.path.exists(transform_json):
        raise FileNotFoundError(transform_json)
    if not os.path.exists(target_ply):
        raise FileNotFoundError(target_ply)
    A, b = _load_matrix(transform_json)
    target = _read_ply_xyz(target_ply)
    tree = cKDTree(target[:, :2])
    target_z = target[:, 2]
    rng = np.random.default_rng(123)
    xy_residuals = []
    z_residuals = []
    total_seen = 0
    with laspy.open(las_path) as fh:
        for chunk_points in fh.chunk_iterator(chunk_size):
            h = fh.header
            xs = chunk_points.X * h.scales[0] + h.offsets[0]
            ys = chunk_points.Y * h.scales[1] + h.offsets[1]
            zs = chunk_points.Z * h.scales[2] + h.offsets[2]
            P = np.column_stack((xs, ys, zs))
            P_out = (P @ A.T) + b
            remaining = max_samples - total_seen
            if remaining <= 0:
                break
            take = min(remaining, P_out.shape[0])
            if take < P_out.shape[0]:
                sel = rng.choice(P_out.shape[0], size=take, replace=False)
                Q = P_out[sel]
            else:
                Q = P_out
            dists, idx = tree.query(Q[:, :2], k=1, workers=-1)
            mask = dists <= nn_radius
            if np.any(mask):
                d = dists[mask]
                z_err = Q[mask, 2] - target_z[idx[mask]]
                xy_residuals.append(d)
                z_residuals.append(z_err)
                total_seen += int(mask.sum())
            if total_seen >= max_samples:
                break
    if total_seen == 0:
        raise RuntimeError("No matches within NN radius. Increase --nn-radius.")
    xy = np.concatenate(xy_residuals, axis=0)
    z = np.concatenate(z_residuals, axis=0)
    res3 = np.sqrt(xy ** 2 + z ** 2)
    stats = {
        "num_matches": int(total_seen),
        "nn_radius_m": float(nn_radius),
        "xy_rmse_m": float(math.sqrt(np.mean(xy ** 2))),
        "xy_median_m": float(np.median(xy)),
        "xy_p90_m": float(np.percentile(xy, 90)),
        "xy_p95_m": float(np.percentile(xy, 95)),
        "z_rmse_m": float(math.sqrt(np.mean(z ** 2))),
        "z_median_abs_m": float(np.median(np.abs(z))),
        "z_p90_abs_m": float(np.percentile(np.abs(z), 90)),
        "z_p95_abs_m": float(np.percentile(np.abs(z), 95)),
        "rmse_3d_m": float(math.sqrt(np.mean(res3 ** 2))),
        "median_3d_m": float(np.median(res3)),
        "p90_3d_m": float(np.percentile(res3, 90)),
        "p95_3d_m": float(np.percentile(res3, 95)),
    }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify alignment using NN stats")
    parser.add_argument("--file", required=True)
    parser.add_argument("--transform-json", required=True)
    parser.add_argument("--target-ply", required=True)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--max-samples", type=int, default=3_000_000)
    parser.add_argument("--nn-radius", type=float, default=0.8)
    args = parser.parse_args()
    try:
        stats = verify_alignment(args.file, args.transform_json, args.target_ply, args.chunk_size, args.max_samples, args.nn_radius)
        print(json.dumps(stats, indent=2))
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main()) 