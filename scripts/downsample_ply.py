#!/usr/bin/env python3

import argparse
import os
import sys
from typing import Optional

import numpy as np


def downsample_ply_random(in_path: str, out_path: Optional[str], fraction: float, seed: int = 123) -> str:
    from plyfile import PlyData, PlyElement  # type: ignore

    in_path = os.path.abspath(in_path)
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    if out_path is None:
        base, _ext = os.path.splitext(in_path)
        out_path = f"{base}_preview.ply"
    out_path = os.path.abspath(out_path)

    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be within (0, 1]")

    ply = PlyData.read(in_path)
    if "vertex" not in ply:
        raise RuntimeError("PLY missing 'vertex' element")
    vertex = ply["vertex"]
    vertex_data = vertex.data
    n = len(vertex_data)
    if n == 0:
        raise RuntimeError("PLY has zero vertices")

    rng = np.random.default_rng(seed)
    if fraction >= 1.0:
        idx = np.arange(n)
    else:
        mask = rng.random(n) < fraction
        if not np.any(mask):
            # Ensure at least 1 vertex remains
            pick = int(rng.integers(0, n))
            mask[pick] = True
        idx = np.nonzero(mask)[0]

    vertex_ds = vertex_data[idx]

    # Rebuild PlyData with the downsampled vertex, preserving properties (including color if present)
    el_vertex = PlyElement.describe(vertex_ds, "vertex")
    out_ply = PlyData([el_vertex], text=ply.text)
    try:
        out_ply.comments = list(getattr(ply, "comments", []))
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_ply.write(out_path)

    print(f"Downsampled PLY vertices: {len(vertex_ds)} of {n} → {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Downsample a PLY (vertex list) by random fraction and write a preview PLY")
    parser.add_argument("--file", required=True, help="Input PLY path")
    parser.add_argument("--out", default=None, help="Output PLY path (default: <input>_preview.ply)")
    parser.add_argument("--fraction", type=float, default=0.02, help="Random keep fraction in (0,1]")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    try:
        out = downsample_ply_random(
            in_path=args.file,
            out_path=args.out,
            fraction=args.fraction,
            seed=args.seed,
        )
        print(out)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main()) 