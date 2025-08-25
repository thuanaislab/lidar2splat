#!/usr/bin/env python3
"""
Convert a DTM GeoTIFF into a LAS point cloud by sampling raster cells.

For each selected raster cell (optionally strided), a single point is created at
the pixel center with Z taken from the raster. Nodata cells are skipped by default.

Notes:
- This produces gridded points, not a triangulated surface.
- By default, classification is set to 2 (Ground).
- CRS WKT is embedded if available (LAS 1.4 WKT VLR), best-effort.

Example:
  python3 scripts/tiff_to_las.py \
    --input-raster /path/to/input.tif \
    --output-las /path/to/output.las \
    --stride 2 --classification 2
"""

import argparse
import sys
from typing import Tuple

import numpy as np

try:
    import rasterio
except Exception as exc:  # pragma: no cover
    print("ERROR: rasterio is required to run this script.", file=sys.stderr)
    raise

try:
    import laspy
    from laspy import LasData, LasHeader
except Exception as exc:  # pragma: no cover
    print("ERROR: laspy is required to run this script.", file=sys.stderr)
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a DTM GeoTIFF into a LAS point cloud")
    parser.add_argument("--input-raster", required=True, help="Path to input GeoTIFF (DTM)")
    parser.add_argument("--output-las", required=True, help="Path to output LAS file (.las)")
    parser.add_argument("--stride", type=int, default=1, help="Sample every Nth pixel in row/col (default: 1)")
    parser.add_argument("--classification", type=int, default=2, help="LAS classification code to set (default: 2 ground)")
    parser.add_argument(
        "--nodata-action",
        choices=["skip", "zero"],
        default="skip",
        help="How to handle nodata pixels: skip them or write Z=0 (default: skip)",
    )
    parser.add_argument(
        "--scale-xyz",
        type=float,
        default=0.01,
        help="LAS XYZ scale (meters per integer step). Default 0.01 (1 cm).",
    )
    return parser.parse_args()


def pixel_centers_to_coords(
    transform: rasterio.Affine, rows: np.ndarray, cols: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute XY coordinates for pixel centers for arrays of row/col indices.

    Handles rotated/sheared transforms using affine parameters directly.
    """
    rows = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)

    # Broadcast to grid
    cols_grid, rows_grid = np.meshgrid(cols, rows)

    # Center offset of +0.5 in pixel space
    x = transform.a * (cols_grid + 0.5) + transform.b * (rows_grid + 0.5) + transform.c
    y = transform.d * (cols_grid + 0.5) + transform.e * (rows_grid + 0.5) + transform.f
    return x, y


def main() -> None:
    args = parse_args()

    with rasterio.open(args.input_raster) as src:
        band = src.read(1, masked=True)
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs
        nodata_value = src.nodata

    stride = max(1, int(args.stride))
    selected_rows = np.arange(0, height, stride)
    selected_cols = np.arange(0, width, stride)

    # Z values for the sampled grid
    z_grid = band[selected_rows][:, selected_cols]
    z_values = np.asarray(z_grid).astype(np.float64)

    # Build mask: skip nodata by default, or fill zeros
    mask = np.ma.getmaskarray(z_grid)
    if args.nodata_action == "skip":
        valid_mask = ~mask
    else:
        # zero-fill nodata
        z_values = np.where(mask, 0.0, z_values)
        valid_mask = np.ones_like(z_values, dtype=bool)

    # Generate XY for sampled grid
    x_grid, y_grid = pixel_centers_to_coords(transform, selected_rows, selected_cols)

    # Flatten and filter
    x_flat = x_grid[valid_mask]
    y_flat = y_grid[valid_mask]
    z_flat = z_values[valid_mask]

    if x_flat.size == 0:
        raise RuntimeError("No valid pixels to export. Check nodata handling and stride.")

    # Prepare LAS header
    header = LasHeader(point_format=0, version="1.2")
    header.offsets = np.array([float(np.min(x_flat)), float(np.min(y_flat)), float(np.min(z_flat))])
    header.scales = np.array([args.scale_xyz, args.scale_xyz, args.scale_xyz])

    las = LasData(header)
    las.x = x_flat
    las.y = y_flat
    las.z = z_flat
    try:
        las.classification = np.full(x_flat.shape, int(args.classification), dtype=np.uint8)
    except Exception:
        pass

    # Embed CRS WKT if available (LAS 1.4 WKT VLR). This is best-effort and backward-compatible.
    try:
        if crs is not None:
            wkt = crs.to_wkt()
            from laspy.vlrs.known import WktCoordinateSystemVlr

            vlr = WktCoordinateSystemVlr(wkt=wkt)
            las.header.add_vlr(vlr)
            # Mark WKT usage bit if available
            try:
                ge = las.header.global_encoding
                ge.wkt = True  # type: ignore[attr-defined]
                las.header.global_encoding = ge
            except Exception:
                pass
    except Exception:
        # If WKT VLR isn't supported, continue without CRS embedding
        pass

    # Write LAS
    las.write(args.output_las)
    print(f"Wrote LAS: {args.output_las} with {x_flat.size} points (stride={stride})")


if __name__ == "__main__":
    main()


