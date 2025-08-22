#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


def _read_ply_xyz(ply_path: str, max_points: int = 2000000) -> np.ndarray:
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
        # Uniform random subsample for speed
        idx = np.random.default_rng(42).choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts


def _read_origin_lat_lon(origin_file_path: str) -> Tuple[float, float]:
    import re

    with open(origin_file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    parts = re.split(r"[\s,]+", first_line)
    if len(parts) < 2:
        raise ValueError(f"Could not parse lat/lon from origin file: {origin_file_path}")
    lat = float(parts[0])
    lon = float(parts[1])
    return lat, lon


def _to_local_from_las(laz_path: str, origin_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    import laspy  # type: ignore
    from pyproj import Transformer  # type: ignore

    laz_path = os.path.abspath(laz_path)
    if not os.path.exists(laz_path):
        raise FileNotFoundError(laz_path)
    if not os.path.exists(origin_path):
        raise FileNotFoundError(origin_path)

    lat, lon = _read_origin_lat_lon(origin_path)

    las = laspy.read(laz_path)

    crs = None
    try:
        crs = las.header.parse_crs()
    except Exception:
        crs = None

    if crs is None or not crs.is_projected:
        raise RuntimeError(
            "Input file CRS is missing or not projected. Expected a projected CRS (e.g., UTM)."
        )

    transformer = Transformer.from_crs(4326, crs, always_xy=True)
    origin_e, origin_n = transformer.transform(lon, lat)

    x_local = np.asarray(las.x, dtype=np.float64) - origin_e
    y_local = np.asarray(las.y, dtype=np.float64) - origin_n
    z = np.asarray(las.z, dtype=np.float64)

    meta = {
        "epsg": getattr(crs, "to_epsg", lambda: None)(),
        "wkt": getattr(crs, "to_wkt", lambda: None)(),
        "origin_e": float(origin_e),
        "origin_n": float(origin_n),
    }
    return x_local, y_local, z, meta


def compute_z_alignment(
    lidar_xy_local: np.ndarray,
    lidar_z: np.ndarray,
    target_xyz_local: np.ndarray,
    nn_radius: float = 1.5,
    max_pairs: int = 2000000,
) -> Tuple[float, dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute robust Z offset to align LiDAR to target local frame.

    Returns (z_offset, stats_dict, lidar_xy_sample, lidar_z_sample, match_idx)
    where z_offset should be ADDED to LiDAR z to best match target z.
    """
    if lidar_xy_local.shape[0] != lidar_z.shape[0]:
        raise ValueError("XY and Z arrays must have the same length")

    # Subsample lidar for speed if needed
    num_lidar = lidar_xy_local.shape[0]
    if num_lidar > max_pairs:
        idx = np.random.default_rng(123).choice(num_lidar, size=max_pairs, replace=False)
        xy_sample = lidar_xy_local[idx]
        z_sample = lidar_z[idx]
    else:
        xy_sample = lidar_xy_local
        z_sample = lidar_z

    # Build KDTree on target XY
    target_xy = target_xyz_local[:, :2]
    target_z = target_xyz_local[:, 2]
    tree = cKDTree(target_xy)

    # Nearest neighbor search
    dists, nn_idx = tree.query(xy_sample, k=1, workers=-1)

    # Filter by radius
    mask = dists <= nn_radius
    if not np.any(mask):
        raise RuntimeError(
            f"No LiDAR points have a target neighbor within {nn_radius} m in XY."
        )

    matched_lidar_z = z_sample[mask]
    matched_target_z = target_z[nn_idx[mask]]
    matched_d_xy = dists[mask]

    # Differences target - lidar
    dz = matched_target_z - matched_lidar_z

    # Robust offset via median
    z_offset = float(np.median(dz))

    # Error metrics after applying offset
    residuals = (matched_lidar_z + z_offset) - matched_target_z
    rmse = float(math.sqrt(float(np.mean(residuals ** 2))))
    mae = float(np.mean(np.abs(residuals)))
    med_abs = float(np.median(np.abs(residuals)))
    p90 = float(np.percentile(np.abs(residuals), 90))
    p95 = float(np.percentile(np.abs(residuals), 95))

    stats = {
        "num_lidar_sampled": int(xy_sample.shape[0]),
        "num_pairs": int(mask.sum()),
        "pair_fraction": float(mask.mean()),
        "nn_radius_m": float(nn_radius),
        "z_offset": z_offset,
        "rmse_m": rmse,
        "mae_m": mae,
        "median_abs_error_m": med_abs,
        "p90_abs_error_m": p90,
        "p95_abs_error_m": p95,
        "mean_xy_nn_dist_m": float(np.mean(matched_d_xy)),
        "median_xy_nn_dist_m": float(np.median(matched_d_xy)),
    }

    # Return sampled arrays and indices of matches for potential reuse
    matched_idx = nn_idx[mask]
    return z_offset, stats, xy_sample[mask], matched_lidar_z, matched_idx


def _estimate_similarity_2d(A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate 2D similarity transform (scale s, rotation R 2x2, translation t) such that
    B ≈ s R A + t.
    """
    if A.shape[0] != B.shape[0] or A.shape[1] != 2 or B.shape[1] != 2:
        raise ValueError("A and B must be (N,2) with same N")
    n = A.shape[0]
    mu_A = np.mean(A, axis=0)
    mu_B = np.mean(B, axis=0)
    A_c = A - mu_A
    B_c = B - mu_B
    cov = (A_c.T @ B_c) / n
    U, S, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    var_A = np.sum(A_c ** 2) / n
    if var_A <= 1e-12:
        s = 1.0
    else:
        s = float(np.sum(S) / var_A)
    t = mu_B - s * (R @ mu_A)
    return s, R, t


def refine_xy_and_z(
    lidar_xy_local: np.ndarray,
    lidar_z: np.ndarray,
    target_xyz_local: np.ndarray,
    init_z_offset: float,
    nn_radius_xy: float = 1.0,
    z_gate: float = 3.0,
    max_pairs: int = 2000000,
    max_iters: int = 8,
) -> Tuple[float, float, np.ndarray, np.ndarray, dict]:
    """
    Refine 2D similarity on XY (s, R_z, t) and Z offset using ICP-style iterations.

    Returns (scale, theta_deg, R, t, stats)
    """
    # Subsample lidar
    num_lidar = lidar_xy_local.shape[0]
    if num_lidar > max_pairs:
        idx = np.random.default_rng(999).choice(num_lidar, size=max_pairs, replace=False)
        A_xy = lidar_xy_local[idx]
        A_z = lidar_z[idx]
    else:
        A_xy = lidar_xy_local
        A_z = lidar_z

    target_xy = target_xyz_local[:, :2]
    target_z = target_xyz_local[:, 2]
    tree = cKDTree(target_xy)

    s = 1.0
    R = np.eye(2)
    t = np.zeros(2)
    z_off = float(init_z_offset)

    last_obj = None
    for i in range(max_iters):
        # Transform LiDAR XY with current estimate
        A_xy_trans = (s * (A_xy @ R.T)) + t

        dists, nn_idx = tree.query(A_xy_trans, k=1, workers=-1)
        # Inlier mask by XY radius and Z gate
        z_res = (A_z + z_off) - target_z[nn_idx]
        mask = (dists <= nn_radius_xy) & (np.abs(z_res) <= z_gate)
        if mask.sum() < 1000:
            break

        A_in = A_xy[mask]
        B_in = target_xy[nn_idx[mask]]

        # Re-estimate similarity on inliers
        s_new, R_new, t_new = _estimate_similarity_2d(A_in, B_in)

        # Update Z offset robustly on inliers
        z_off_new = float(np.median(target_z[nn_idx[mask]] - A_z[mask]))

        # Objective: combined XY and Z residuals
        A_xy_trans_new = (s_new * (A_in @ R_new.T)) + t_new
        xy_res = np.linalg.norm(A_xy_trans_new - B_in, axis=1)
        z_res_in = (A_z[mask] + z_off_new) - target_z[nn_idx[mask]]
        obj = float(np.median(xy_res) + 0.5 * np.median(np.abs(z_res_in)))

        s, R, t, z_off = s_new, R_new, t_new, z_off_new

        if last_obj is not None and abs(last_obj - obj) < 1e-4:
            break
        last_obj = obj

    theta = math.degrees(math.atan2(R[1, 0], R[0, 0]))

    # Final stats on all accepted matches
    A_xy_trans = (s * (A_xy @ R.T)) + t
    dists, nn_idx = tree.query(A_xy_trans, k=1, workers=-1)
    z_res = (A_z + z_off) - target_z[nn_idx]
    mask = (dists <= nn_radius_xy) & (np.abs(z_res) <= z_gate)

    stats = {}
    if np.any(mask):
        xy_res = np.linalg.norm(A_xy_trans[mask] - target_xy[nn_idx[mask]], axis=1)
        z_res_in = z_res[mask]
        res3 = np.sqrt(xy_res ** 2 + z_res_in ** 2)
        stats = {
            "iters": i + 1,
            "num_pairs_xy": int(mask.sum()),
            "xy_rmse_m": float(math.sqrt(np.mean(xy_res ** 2))),
            "xy_median_m": float(np.median(xy_res)),
            "xy_p90_m": float(np.percentile(xy_res, 90)),
            "xy_p95_m": float(np.percentile(xy_res, 95)),
            "z_rmse_m": float(math.sqrt(np.mean(z_res_in ** 2))),
            "z_median_abs_m": float(np.median(np.abs(z_res_in))),
            "z_p90_abs_m": float(np.percentile(np.abs(z_res_in), 90)),
            "z_p95_abs_m": float(np.percentile(np.abs(z_res_in), 95)),
            "rmse_3d_m": float(math.sqrt(np.mean(res3 ** 2))),
            "median_3d_m": float(np.median(res3)),
            "p90_3d_m": float(np.percentile(res3, 90)),
            "p95_3d_m": float(np.percentile(res3, 95)),
            "scale": float(s),
            "theta_deg": float(theta),
            "tx": float(t[0]),
            "ty": float(t[1]),
            "z_offset": float(z_off),
        }

    return s, theta, R, t, z_off, stats


def write_aligned_laz(
    in_laz_path: str,
    out_laz_path: str,
    x_out: np.ndarray,
    y_out: np.ndarray,
    z_out: np.ndarray,
) -> None:
    import laspy  # type: ignore

    las = laspy.read(in_laz_path)

    # Set scales/offsets BEFORE assigning coordinates to ensure values fit int range
    try:
        # Choose millimeter precision by default
        las.header.scales = (0.001, 0.001, 0.001)
    except Exception:
        # Some laspy versions require list
        las.header.scales = [0.001, 0.001, 0.001]

    try:
        las.header.offsets = (0.0, 0.0, 0.0)
    except Exception:
        las.header.offsets = [0.0, 0.0, 0.0]

    # Assign transformed coordinates
    las.x = x_out
    las.y = y_out
    las.z = z_out

    # Optionally strip CRS since local frame is no longer a projected CRS
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

    # Write output
    las.write(out_laz_path)


def save_transform_json(
    json_path: str,
    origin_e: float,
    origin_n: float,
    z_offset: float,
    lidar_epsg: Optional[int],
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> None:
    # 4x4 homogeneous matrix that maps LiDAR UTM (x,y,z,1) to target local (X,Y,Z,1)
    # XY: apply origin subtraction, then similarity: X,Y = s*R*(x-origin) + t
    # Z: Z = z + z_offset
    Mxy = scale * R
    v = -Mxy @ np.array([origin_e, origin_n]) + t
    matrix = [
        [float(Mxy[0, 0]), float(Mxy[0, 1]), 0.0, float(v[0])],
        [float(Mxy[1, 0]), float(Mxy[1, 1]), 0.0, float(v[1])],
        [0.0, 0.0, 1.0, float(z_offset)],
        [0.0, 0.0, 0.0, 1.0],
    ]
    theta_deg = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    payload = {
        "lidar_epsg": int(lidar_epsg) if lidar_epsg is not None else None,
        "origin_e": float(origin_e),
        "origin_n": float(origin_n),
        "z_offset": float(z_offset),
        "scale": float(scale),
        "R_2x2": [[float(R[0, 0]), float(R[0, 1])], [float(R[1, 0]), float(R[1, 1])]],
        "theta_deg": float(theta_deg),
        "t_xy": [float(t[0]), float(t[1])],
        "matrix_4x4": matrix,
        "convention": {
            "XY": "X,Y = s*R*(x-origin_e, y-origin_n) + t",
            "Z": "Z = z + z_offset",
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Align LiDAR to target SFM local frame: coarse Z, then refine XY similarity and Z. "
            "Writes aligned LAZ, and a 4x4 transform JSON."
        )
    )
    parser.add_argument("--file", required=True, help="Path to input LAS/LAZ (projected CRS, e.g., UTM)")
    parser.add_argument(
        "--origin",
        default=os.path.join(os.getcwd(), "TRANSFORM", "sfm", "origin_point.txt"),
        help="Path to origin_point.txt containing 'lat, lon'",
    )
    parser.add_argument(
        "--target-ply",
        default=os.path.join(os.getcwd(), "TRANSFORM", "sfm", "sparse_transformed.ply"),
        help="Path to target SFM point cloud (PLY) in local meter frame",
    )
    parser.add_argument(
        "--out-laz",
        default=None,
        help="Output path for LAZ with local XY similarity and Z aligned (defaults to <input>_aligned.laz)",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Output JSON path for alignment stats and transform (defaults to <input>_align.json)",
    )
    parser.add_argument(
        "--out-align-txt",
        default=None,
        help="Optional output TXT with 'origin_e origin_n z_offset' (defaults to <input>_target_align.txt)",
    )
    parser.add_argument("--nn-radius", type=float, default=1.5, help="Coarse Z: max XY distance (m) to accept a match")
    parser.add_argument("--max-pairs", type=int, default=2000000, help="Max LiDAR points sampled for coarse Z")
    parser.add_argument("--xy-iters", type=int, default=8, help="Max ICP iterations for XY refine")
    parser.add_argument("--xy-radius", type=float, default=1.0, help="XY refine: NN radius (m)")
    parser.add_argument("--z-gate", type=float, default=3.0, help="XY refine: max |Z residual| to accept (m)")
    parser.add_argument("--max-pairs-xy", type=int, default=2000000, help="Max LiDAR points sampled for XY refine")

    args = parser.parse_args()

    # 1) Read and convert LiDAR to local XY using origin
    x_local, y_local, z_lidar, meta = _to_local_from_las(args.file, args.origin)
    lidar_epsg = meta.get("epsg")
    origin_e = meta["origin_e"]
    origin_n = meta["origin_n"]

    lidar_xy_local = np.stack([x_local, y_local], axis=1)

    # 2) Load target SFM local cloud (PLY)
    target_xyz = _read_ply_xyz(args.target_ply)

    # 3) Coarse Z alignment
    z_offset, z_stats, _, _, _ = compute_z_alignment(
        lidar_xy_local=lidar_xy_local,
        lidar_z=z_lidar,
        target_xyz_local=target_xyz,
        nn_radius=args.nn_radius,
        max_pairs=args.max_pairs,
    )

    # 4) Refine XY similarity and Z
    s, theta_deg, R, t, z_off_final, stats_xy = refine_xy_and_z(
        lidar_xy_local=lidar_xy_local,
        lidar_z=z_lidar,
        target_xyz_local=target_xyz,
        init_z_offset=z_offset,
        nn_radius_xy=args.xy_radius,
        z_gate=args.z_gate,
        max_pairs=args.max_pairs_xy,
        max_iters=args.xy_iters,
    )

    # 5) Compose final outputs
    Mxy = s * R
    xy_aligned = (lidar_xy_local @ Mxy.T) + t  # (N,2)
    z_aligned = z_lidar + z_off_final

    # 6) Write aligned LAZ
    out_laz = args.out_laz
    if out_laz is None:
        base, ext = os.path.splitext(args.file)
        if ext.lower() not in (".las", ".laz"):
            ext = ".laz"
        out_laz = f"{base}_aligned{ext}"

    write_aligned_laz(
        in_laz_path=args.file,
        out_laz_path=out_laz,
        x_out=xy_aligned[:, 0],
        y_out=xy_aligned[:, 1],
        z_out=z_aligned,
    )

    # 7) Save transform JSON and stats
    out_json = args.out_json
    if out_json is None:
        base, _ = os.path.splitext(args.file)
        out_json = f"{base}_align.json"

    save_transform_json(
        json_path=out_json,
        origin_e=origin_e,
        origin_n=origin_n,
        z_offset=z_off_final,
        lidar_epsg=lidar_epsg,
        scale=s,
        R=R,
        t=t,
    )

    # Merge stats
    all_stats = {"coarse_z": z_stats, "refine_xy": stats_xy}
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except Exception:
        existing = {}
    existing["stats"] = all_stats
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    # 8) Write simple align TXT file with origin_e origin_n z_offset
    out_align_txt = args.out_align_txt
    if out_align_txt is None:
        base, _ = os.path.splitext(args.file)
        out_align_txt = f"{base}_target_align.txt"
    try:
        with open(out_align_txt, "w", encoding="utf-8") as f:
            f.write(f"{origin_e} {origin_n} {z_off_final}\n")
    except Exception:
        pass

    # 9) Report
    print("Alignment complete")
    print(json.dumps(all_stats, indent=2))
    print(f"Scale: {s:.9f}, Theta(deg): {theta_deg:.6f}, t: [{t[0]:.4f}, {t[1]:.4f}], z_offset: {z_off_final:.4f}")
    print(f"Wrote aligned LAZ: {out_laz}")
    print(f"Wrote transform JSON: {out_json}")
    print(f"Wrote align TXT: {out_align_txt}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2) 