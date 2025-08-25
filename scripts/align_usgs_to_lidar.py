#!/usr/bin/env python3
"""
Align a USGS DTM GeoTIFF vertically to match LiDAR ground elevations from a LAS file.

Approach:
- Load USGS DTM raster
- Load LiDAR LAS points (optionally filter to ground classification == 2)
- Bin LiDAR points into raster grid cells and compute per-cell mean elevation
- Compute a robust vertical offset as median( raster - lidar_mean ) over valid cells
- Apply the offset to the entire raster and write an aligned GeoTIFF

This script assumes LiDAR and raster share the same CRS.
"""

import argparse
import sys
from typing import Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.transform import Affine
except Exception as exc:  # pragma: no cover
    print("ERROR: rasterio is required to run this script.", file=sys.stderr)
    raise

try:
    import laspy
except Exception as exc:  # pragma: no cover
    print("ERROR: laspy is required to run this script.", file=sys.stderr)
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align a USGS DTM vertically to LiDAR ground heights")
    parser.add_argument("--input-raster", required=True, help="Path to input USGS DTM GeoTIFF")
    parser.add_argument("--input-lidar", required=True, help="Path to input LiDAR LAS/LAZ file")
    parser.add_argument(
        "--output-raster",
        required=True,
        help="Path to output aligned GeoTIFF",
    )
    parser.add_argument(
        "--min-points-per-cell",
        type=int,
        default=3,
        help="Minimum LiDAR points per raster cell to treat it as valid (default: 3)",
    )
    parser.add_argument(
        "--use-ground-class",
        action="store_true",
        help="If set, use only ground-classified points (classification == 2)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optionally limit the number of valid cells randomly sampled to estimate the offset (0 = no limit)",
    )
    return parser.parse_args()


def compute_col_row(inv_transform: Affine, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute raster column and row indices for arrays of x,y using an inverse affine transform.

    The result indices are not clipped to bounds and are floor()'d to integers.
    """
    # Ensure standard NumPy arrays (avoid laspy ScaledArrayView arithmetic issues)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    # Vectorized application of inverse transform
    cols = inv_transform.a * xs + inv_transform.b * ys + inv_transform.c
    rows = inv_transform.d * xs + inv_transform.e * ys + inv_transform.f
    cols = np.floor(cols).astype(np.int64)
    rows = np.floor(rows).astype(np.int64)
    return cols, rows


def aggregate_lidar_to_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    inv_transform: Affine,
    width: int,
    height: int,
    min_points_per_cell: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate LiDAR point elevations into raster grid cells.

    Returns:
        lidar_mean_grid: 2D array (height, width) of per-cell mean Z (np.nan where insufficient points)
        lidar_count_grid: 2D array (height, width) of per-cell counts
    """
    cols, rows = compute_col_row(inv_transform, xs, ys)

    # Keep only points inside raster bounds
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    if not np.any(inside):
        raise RuntimeError("No LiDAR points overlap the raster extent.")

    cols = cols[inside]
    rows = rows[inside]
    zs = zs[inside]

    # Flattened linear indices for bincount
    linear_idx = rows * width + cols

    total_cells = width * height
    sum_z = np.bincount(linear_idx, weights=zs, minlength=total_cells)
    count = np.bincount(linear_idx, minlength=total_cells)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_flat = sum_z / count
    # Where count < min_points_per_cell, mark as NaN
    valid_mask = count >= min_points_per_cell
    mean_flat[~valid_mask] = np.nan

    lidar_mean_grid = mean_flat.reshape((height, width))
    lidar_count_grid = count.reshape((height, width))
    return lidar_mean_grid, lidar_count_grid


def robust_vertical_offset(
    raster_data: np.ndarray,
    lidar_mean_grid: np.ndarray,
    max_samples: int = 0,
) -> float:
    """Compute a robust vertical offset as the median of differences.

    We compute median(raster - lidar) over cells where both are valid.
    Optionally subsample valid cells to at most max_samples for speed.
    """
    # raster_data may be a masked array; convert to ndarray and mask
    if np.ma.is_masked(raster_data):
        raster_mask = ~raster_data.mask
        raster_values = raster_data.filled(np.nan)
    else:
        raster_values = raster_data
        raster_mask = ~np.isnan(raster_values)

    lidar_mask = ~np.isnan(lidar_mean_grid)
    both_valid = raster_mask & lidar_mask
    if not np.any(both_valid):
        raise RuntimeError("No overlapping valid cells between raster and LiDAR grid to estimate offset.")

    diffs = raster_values[both_valid] - lidar_mean_grid[both_valid]

    if max_samples and diffs.size > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(diffs.size, size=max_samples, replace=False)
        diffs = diffs[idx]

    # Median is robust to outliers
    return float(np.nanmedian(diffs))


def main() -> None:
    args = parse_args()

    # Open raster
    with rasterio.open(args.input_raster) as src:
        raster_data = src.read(1, masked=True)
        meta = src.meta.copy()
        transform = src.transform
        inv_transform = ~transform
        width = src.width
        height = src.height
        nodata = src.nodata

    # Load LAS
    las = laspy.read(args.input_lidar)
    xs = np.asarray(las.x, dtype=np.float64)
    ys = np.asarray(las.y, dtype=np.float64)
    zs = np.asarray(las.z, dtype=np.float64)

    if args.use_ground_class and hasattr(las, "classification"):
        ground_mask = (las.classification == 2)
        xs = xs[ground_mask]
        ys = ys[ground_mask]
        zs = zs[ground_mask]

    # Aggregate LiDAR into raster grid
    lidar_mean_grid, lidar_count_grid = aggregate_lidar_to_grid(
        xs=xs,
        ys=ys,
        zs=zs,
        inv_transform=inv_transform,
        width=width,
        height=height,
        min_points_per_cell=args.min_points_per_cell,
    )

    # Estimate vertical offset
    offset = robust_vertical_offset(raster_data, lidar_mean_grid, max_samples=args.max_samples)
    print(f"Estimated vertical offset (raster - lidar): {offset:.4f} meters")

    # Apply offset so that output matches LiDAR heights: output = raster - offset
    if np.ma.is_masked(raster_data):
        aligned = raster_data.filled(np.nan) - offset
        aligned = np.where(raster_data.mask, np.nan if nodata is None else nodata, aligned)
    else:
        aligned = raster_data.astype(np.float64) - offset

    # Prepare output dtype (float32) and nodata handling
    out_meta = meta.copy()
    out_meta.update({
        "dtype": "float32",
    })

    # Write output
    with rasterio.open(args.output_raster, "w", **out_meta) as dst:
        data_to_write = aligned.astype(np.float32)
        # If nodata is set, replace NaNs with nodata value
        if nodata is not None and np.isnan(nodata) is False:
            data_to_write = np.where(np.isnan(data_to_write), nodata, data_to_write)
        dst.write(data_to_write, 1)

    print(f"Aligned raster written to: {args.output_raster}")


if __name__ == "__main__":
    main()


