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
        raise RuntimeError("Input file CRS is missing or not projected. Expected a projected CRS (e.g., UTM).")
    transformer = Transformer.from_crs(4326, crs, always_xy=True)
    origin_e, origin_n = transformer.transform(lon, lat)
    x_local = np.asarray(las.x, dtype=np.float64) - origin_e
    y_local = np.asarray(las.y, dtype=np.float64) - origin_n
    z = np.asarray(las.z, dtype=np.float64)
    meta = {
        "epsg": getattr(crs, "to_epsg", lambda: None)(),
        "origin_e": float(origin_e),
        "origin_n": float(origin_n),
    }
    return x_local, y_local, z, meta


def compute_z_alignment(lidar_xy_local: np.ndarray, lidar_z: np.ndarray, target_xyz_local: np.ndarray,
                         nn_radius: float = 2.0, max_pairs: int = 3000000) -> Tuple[float, dict]:
    if lidar_xy_local.shape[0] != lidar_z.shape[0]:
        raise ValueError("XY and Z arrays must have the same length")
    num_lidar = lidar_xy_local.shape[0]
    if num_lidar > max_pairs:
        idx = np.random.default_rng(123).choice(num_lidar, size=max_pairs, replace=False)
        xy_sample = lidar_xy_local[idx]
        z_sample = lidar_z[idx]
    else:
        xy_sample = lidar_xy_local
        z_sample = lidar_z
    target_xy = target_xyz_local[:, :2]
    target_z = target_xyz_local[:, 2]
    tree = cKDTree(target_xy)
    dists, nn_idx = tree.query(xy_sample, k=1, workers=-1)
    mask = dists <= nn_radius
    if not np.any(mask):
        raise RuntimeError(f"No LiDAR points have a target neighbor within {nn_radius} m in XY.")
    dz = target_z[nn_idx[mask]] - z_sample[mask]
    z_offset = float(np.median(dz))
    residuals = (z_sample[mask] + z_offset) - target_z[nn_idx[mask]]
    stats = {
        "num_lidar_sampled": int(xy_sample.shape[0]),
        "num_pairs": int(mask.sum()),
        "pair_fraction": float(mask.mean()),
        "nn_radius_m": float(nn_radius),
        "z_offset": z_offset,
        "rmse_m": float(math.sqrt(float(np.mean(residuals ** 2)))),
        "mae_m": float(np.mean(np.abs(residuals))),
        "median_abs_error_m": float(np.median(np.abs(residuals))),
        "p90_abs_error_m": float(np.percentile(np.abs(residuals), 90)),
        "p95_abs_error_m": float(np.percentile(np.abs(residuals), 95)),
    }
    return z_offset, stats


def _estimate_similarity_2d(A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
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
    s = 1.0 if var_A <= 1e-12 else float(np.sum(S) / var_A)
    t = mu_B - s * (R @ mu_A)
    return s, R, t


def _compute_coarse_xy(lidar_xy_local: np.ndarray, target_xy: np.ndarray, nn_radius: float,
                        max_pairs: int = 1000000) -> Tuple[float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(321)
    n = lidar_xy_local.shape[0]
    take = min(n, max_pairs)
    if take < n:
        idx = rng.choice(n, size=take, replace=False)
        A = lidar_xy_local[idx]
    else:
        A = lidar_xy_local
    tree = cKDTree(target_xy)
    dists, nn_idx = tree.query(A, k=1, workers=-1)
    # Use the best-matching subset by distance (robust, no fixed radius gating)
    if A.shape[0] < 100:
        return 1.0, np.eye(2), np.zeros(2)
    # Keep up to 200k best pairs or 80th percentile, whichever is smaller
    k = min(200000, A.shape[0])
    order = np.argsort(dists)
    sel = order[:k]
    # Further trim by percentile to reduce extreme outliers
    thr = float(np.percentile(dists[sel], 80.0))
    mask = dists <= thr
    if np.count_nonzero(mask) < 100:
        mask = sel  # fall back to top-k if too few by percentile
    A_in = A[mask]
    B_in = target_xy[nn_idx[mask]]
    s, R, t = _estimate_similarity_2d(A_in, B_in)
    return s, R, t


def _parse_kml_polygon_coords(kml_path: str) -> np.ndarray:
    # Returns array of shape (M, 2) of (x,y) parsed in order from the first <coordinates> polygon found
    import re
    if not os.path.exists(kml_path):
        raise FileNotFoundError(kml_path)
    with open(kml_path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"<coordinates>(.*?)</coordinates>", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        raise RuntimeError("No <coordinates> section found in KML")
    coords_blob = m.group(1).strip()
    pts = []
    for token in coords_blob.replace("\n", " ").split():
        parts = token.split(",")
        if len(parts) < 2:
            continue
        x = float(parts[0])  # lon or x
        y = float(parts[1])  # lat or y
        pts.append((x, y))
    if len(pts) < 3:
        raise RuntimeError("KML polygon has fewer than 3 vertices")
    return np.asarray(pts, dtype=np.float64)


def _points_in_polygon(points_xy: np.ndarray, poly_xy: np.ndarray) -> np.ndarray:
    # Ray casting algorithm; vectorized over points
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    xv = poly_xy[:, 0]
    yv = poly_xy[:, 1]
    nvert = len(xv)
    inside = np.zeros(points_xy.shape[0], dtype=bool)
    j = nvert - 1
    for i in range(nvert):
        xi = xv[i]
        yi = yv[i]
        xj = xv[j]
        yj = yv[j]
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-20) + xi)
        inside ^= intersect
        j = i
    return inside


def refine_xy_and_z(lidar_xy_local: np.ndarray, lidar_z: np.ndarray, target_xyz_local: np.ndarray,
                     init_z_offset: float, nn_radius_xy: float = 0.8, z_gate: float = 2.5,
                     max_pairs: int = 3000000, max_iters: int = 12) -> Tuple[float, float, np.ndarray, np.ndarray, float, dict]:
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
        A_xy_trans = (s * (A_xy @ R.T)) + t
        dists, nn_idx = tree.query(A_xy_trans, k=1, workers=-1)
        z_res = (A_z + z_off) - target_z[nn_idx]
        mask = (dists <= nn_radius_xy) & (np.abs(z_res) <= z_gate)
        if mask.sum() < 1000:
            break
        A_in = A_xy[mask]
        B_in = target_xy[nn_idx[mask]]
        s_new, R_new, t_new = _estimate_similarity_2d(A_in, B_in)
        z_off_new = float(np.median(target_z[nn_idx[mask]] - A_z[mask]))
        A_xy_trans_new = (s_new * (A_in @ R_new.T)) + t_new
        xy_res = np.linalg.norm(A_xy_trans_new - B_in, axis=1)
        z_res_in = (A_z[mask] + z_off_new) - target_z[nn_idx[mask]]
        obj = float(np.median(xy_res) + 0.5 * np.median(np.abs(z_res_in)))
        s, R, t, z_off = s_new, R_new, t_new, z_off_new
        if last_obj is not None and abs(last_obj - obj) < 1e-4:
            break
        last_obj = obj
    theta = math.degrees(math.atan2(R[1, 0], R[0, 0]))
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


def write_aligned_laz(in_laz_path: str, out_laz_path: str, origin_e: float, origin_n: float,
                      Mxy: np.ndarray, t_xy: np.ndarray, z_off: float,
                      poly_target_xy: Optional[np.ndarray]) -> None:
    import laspy  # type: ignore
    las = laspy.read(in_laz_path)
    # Transform to target frame
    x_local = np.asarray(las.x, dtype=np.float64) - origin_e
    y_local = np.asarray(las.y, dtype=np.float64) - origin_n
    xy_local = np.stack([x_local, y_local], axis=1)
    xy_out = (xy_local @ Mxy.T) + t_xy
    z_out = np.asarray(las.z, dtype=np.float64) + z_off
    if poly_target_xy is not None:
        inside = _points_in_polygon(xy_out, poly_target_xy)
    else:
        inside = np.ones(xy_out.shape[0], dtype=bool)
    # Keep only inside points
    las.points = las.points[inside]
    try:
        las.header.scales = (0.001, 0.001, 0.001)
    except Exception:
        las.header.scales = [0.001, 0.001, 0.001]
    try:
        las.header.offsets = (0.0, 0.0, 0.0)
    except Exception:
        las.header.offsets = [0.0, 0.0, 0.0]
    las.x = xy_out[inside, 0]
    las.y = xy_out[inside, 1]
    las.z = z_out[inside]
    las.write(out_laz_path)


def save_transform_json(json_path: str, origin_e: float, origin_n: float, z_offset: float, lidar_epsg: Optional[int],
                        scale: float, R: np.ndarray, t: np.ndarray) -> None:
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
        "convention": {"XY": "X,Y = s*R*(x-origin_e, y-origin_n) + t", "Z": "Z = z + z_offset"},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _maybe_make_poly_in_target_xy(kml_path: Optional[str], kml_crs: Optional[str],
                                  lidar_epsg: Optional[int], origin_e: float, origin_n: float,
                                  coarse_Mxy: np.ndarray, coarse_t: np.ndarray) -> Optional[np.ndarray]:
    # Returns polygon vertices in target XY coordinates or None
    if kml_path is None:
        return None
    poly_in = _parse_kml_polygon_coords(kml_path)
    if kml_crs is None or kml_crs.strip().lower() == "target":
        # Already in target XY
        return poly_in
    # Transform from provided CRS to LiDAR projected CRS, then to LiDAR local, then into target XY via coarse transform
    from pyproj import CRS, Transformer  # type: ignore
    if lidar_epsg is None:
        raise RuntimeError("LiDAR EPSG is unknown; cannot transform KML polygon")
    src_crs = CRS.from_user_input(kml_crs)
    dst_crs = CRS.from_epsg(int(lidar_epsg))
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs, ys = transformer.transform(poly_in[:, 0], poly_in[:, 1])
    lidar_local = np.stack([np.asarray(xs) - origin_e, np.asarray(ys) - origin_n], axis=1)
    poly_target = (lidar_local @ coarse_Mxy.T) + coarse_t
    return poly_target


def _write_cropped_target_ply_from_file(ply_path: str, poly_target_xy: Optional[np.ndarray], out_path: str) -> None:
    from plyfile import PlyData, PlyElement  # type: ignore
    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise RuntimeError("PLY missing 'vertex' element")
    v = ply["vertex"]
    xs = np.asarray(v["x"], dtype=np.float64)
    ys = np.asarray(v["y"], dtype=np.float64)
    pts_xy = np.stack([xs, ys], axis=1)
    if poly_target_xy is not None:
        inside = _points_in_polygon(pts_xy, poly_target_xy)
    else:
        inside = np.ones(xs.shape[0], dtype=bool)
    data = v.data[inside]
    el = PlyElement.describe(data, "vertex")
    out_ply = PlyData([el], text=ply.text)
    try:
        out_ply.comments = list(getattr(ply, "comments", []))
    except Exception:
        pass
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_ply.write(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate LiDAR→target transform and write outputs")
    parser.add_argument("--lidar-file", required=True)
    parser.add_argument("--origin-file", required=True)
    parser.add_argument("--target-ply", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nn-radius", type=float, default=2.0)
    parser.add_argument("--max-pairs", type=int, default=3000000)
    parser.add_argument("--xy-iters", type=int, default=12)
    parser.add_argument("--xy-radius", type=float, default=0.8)
    parser.add_argument("--z-gate", type=float, default=2.5)
    parser.add_argument("--max-pairs-xy", type=int, default=3000000)
    parser.add_argument("--kml-boundary", default=None, help="KML polygon to crop to")
    parser.add_argument("--kml-crs", default="epsg:4326", help="CRS of KML coords or 'target' if already in target XY")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    x_local, y_local, z_lidar, meta = _to_local_from_las(args.lidar_file, args.origin_file)
    lidar_epsg = meta.get("epsg")
    origin_e = meta["origin_e"]
    origin_n = meta["origin_n"]
    lidar_xy_local = np.stack([x_local, y_local], axis=1)
    target_xyz = _read_ply_xyz(args.target_ply)
    # 1) Coarse Z alignment
    z_offset, z_stats = compute_z_alignment(lidar_xy_local, z_lidar, target_xyz, args.nn_radius, args.max_pairs)
    # 2) Build coarse XY transform to enable cropping in target space
    coarse_s, coarse_R, coarse_t = _compute_coarse_xy(lidar_xy_local, target_xyz[:, :2], args.nn_radius, min(args.max_pairs, 1000000))
    coarse_Mxy = coarse_s * coarse_R
    # 3) Prepare polygon in target XY
    poly_target_xy = None
    if args.kml_boundary is not None:
        poly_target_xy = _maybe_make_poly_in_target_xy(args.kml_boundary, args.kml_crs, lidar_epsg, origin_e, origin_n,
                                                       coarse_Mxy, coarse_t)
    # 4) Crop both target and LiDAR (for optimization) prior to refinement
    if poly_target_xy is not None:
        # Crop target in its frame
        mask_tgt = _points_in_polygon(target_xyz[:, :2], poly_target_xy)
        target_xyz = target_xyz[mask_tgt]
        # Map LiDAR to target frame via coarse transform then crop
        lidar_xy_coarse = (lidar_xy_local @ coarse_Mxy.T) + coarse_t
        mask_lidar = _points_in_polygon(lidar_xy_coarse, poly_target_xy)
        lidar_xy_local = lidar_xy_local[mask_lidar]
        z_lidar = z_lidar[mask_lidar]
        print(f"Cropping with KML polygon → lidar_in: {lidar_xy_local.shape[0]}, target_in: {target_xyz.shape[0]}")
        if lidar_xy_local.shape[0] == 0 or target_xyz.shape[0] == 0:
            raise RuntimeError("Cropping removed all points; check KML alignment and CRS settings")
    # 5) Refine XY + Z on cropped data
    s, theta_deg, R, t, z_off_final, stats_xy = refine_xy_and_z(
        lidar_xy_local, z_lidar, target_xyz, z_offset, args.xy_radius, args.z_gate, args.max_pairs_xy, args.xy_iters
    )
    Mxy = s * R
    xy_aligned = (lidar_xy_local @ Mxy.T) + t
    z_aligned = z_lidar + z_off_final
    # 6) Write outputs (cropped)
    base = os.path.splitext(os.path.basename(args.lidar_file))[0]
    out_laz = os.path.join(args.out_dir, f"{base}_aligned.las")
    out_json = os.path.join(args.out_dir, f"{base}_align.json")
    out_align_txt = os.path.join(args.out_dir, f"{base}_target_align.txt")
    # Write aligned (cropped) LiDAR
    write_aligned_laz(args.lidar_file, out_laz, origin_e, origin_n, Mxy, t, z_off_final, poly_target_xy)
    # Save transform
    save_transform_json(out_json, origin_e, origin_n, z_off_final, lidar_epsg, s, R, t)
    with open(out_json, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing["stats"] = {"coarse_z": z_stats, "refine_xy": stats_xy}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    with open(out_align_txt, "w", encoding="utf-8") as f:
        f.write(f"{origin_e} {origin_n} {z_off_final}\n")
    # Also write cropped target PLY for preview convenience if polygon provided
    if poly_target_xy is not None:
        tgt_base = os.path.splitext(os.path.basename(args.target_ply))[0]
        out_tgt_cropped = os.path.join(args.out_dir, f"{tgt_base}_cropped.ply")
        _write_cropped_target_ply_from_file(args.target_ply, poly_target_xy, out_tgt_cropped)
    print("Alignment complete")
    print(json.dumps(existing["stats"], indent=2))
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