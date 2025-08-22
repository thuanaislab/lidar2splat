#!/usr/bin/env python3

import argparse
import os
import sys
from typing import Optional

import numpy as np


def downsample_las_random(
    in_path: str,
    out_path: Optional[str],
    fraction: float,
    chunk_size: int = 2_000_000,
    seed: int = 123,
) -> str:
    import laspy  # type: ignore

    in_path = os.path.abspath(in_path)
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    if out_path is None:
        base, _ext = os.path.splitext(in_path)
        # Always write LAZ for smaller download size
        out_path = f"{base}_ds.laz"
    out_path = os.path.abspath(out_path)

    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be within (0, 1]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    rng = np.random.default_rng(seed)
    total_in = 0
    total_out = 0

    with laspy.open(in_path) as reader:
        # Prepare output header based on input header
        hdr_in = reader.header
        # Create a compatible header for writing
        hdr_out = laspy.LasHeader(point_format=hdr_in.point_format, version=hdr_in.version)
        # Preserve scales and offsets to avoid precision loss and keep coordinates consistent
        try:
            hdr_out.scales = hdr_in.scales
        except Exception:
            hdr_out.scales = [0.001, 0.001, 0.001]
        try:
            hdr_out.offsets = hdr_in.offsets
        except Exception:
            hdr_out.offsets = [0.0, 0.0, 0.0]
        # Try to preserve VLRs (including CRS) if possible
        try:
            hdr_out.vlrs = list(hdr_in.vlrs)
        except Exception:
            pass

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with laspy.open(out_path, mode="w", header=hdr_out) as writer:
            for chunk in reader.chunk_iterator(chunk_size):
                # chunk is a PointRecord; sample uniformly at random within this chunk
                n = len(chunk)
                if n == 0:
                    continue
                total_in += n
                if fraction >= 1.0:
                    # Write all points
                    writer.write_points(chunk)
                    total_out += n
                    continue
                # Bernoulli mask per point for stable memory use
                mask = rng.random(n) < fraction
                if np.any(mask):
                    writer.write_points(chunk[mask])
                    total_out += int(mask.sum())

    # Ensure at least one point got written in extremely small fractions on tiny files
    if total_out == 0:
        raise RuntimeError(
            "Downsampling produced 0 points. Increase --fraction or check input file."
        )
    print(f"Downsampled {total_out} of {total_in} points (fraction≈{total_out/max(total_in,1):.6f}) → {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Downsample a LAS/LAZ by random fraction and write a LAZ preview")
    parser.add_argument("--file", required=True, help="Input LAS/LAZ path")
    parser.add_argument("--out", default=None, help="Output path (default: <input>_ds.laz)")
    parser.add_argument("--fraction", type=float, default=0.02, help="Random keep fraction in (0,1]")
    parser.add_argument("--chunk-size", type=int, default=2_000_000, help="Chunk size for streaming IO")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    args = parser.parse_args()

    try:
        out = downsample_las_random(
            in_path=args.file,
            out_path=args.out,
            fraction=args.fraction,
            chunk_size=args.chunk_size,
            seed=args.seed,
        )
        print(out)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main()) 