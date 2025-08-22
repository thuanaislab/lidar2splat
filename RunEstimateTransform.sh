#!/usr/bin/env bash
set -euo pipefail

# Paths
ROOT_DIR="/pipeline/TRANSFORM"
DATA_DIR="$ROOT_DIR/data/site3"
OUT_DIR="$ROOT_DIR/output"
SCRIPT_DIR="$ROOT_DIR/scripts"

LIDAR_FILE="$DATA_DIR/lidar/MurphysSite3-NoColor.las"
ORIGIN_FILE="$DATA_DIR/splat/origin_point.txt"
TARGET_PLY="$DATA_DIR/splat/sparse_transformed.ply"

mkdir -p "$OUT_DIR"

source trans-env/bin/activate

echo "[1/3] Estimating transform and writing aligned LiDAR + JSON + TXT..."
python3 "$SCRIPT_DIR/estimate_transform.py" \
  --lidar-file "$LIDAR_FILE" \
  --origin-file "$ORIGIN_FILE" \
  --target-ply "$TARGET_PLY" \
  --out-dir "$OUT_DIR" \
  --nn-radius 2.0 --max-pairs 3000000 \
  --xy-iters 12 --xy-radius 0.8 --z-gate 2.5 --max-pairs-xy 3000000

BASE="$(basename "$LIDAR_FILE")"
BASE_NO_EXT="${BASE%.*}"
ALIGN_JSON="$OUT_DIR/${BASE_NO_EXT}_align.json"
ALIGNED_LAS="$OUT_DIR/${BASE_NO_EXT}_aligned.las"

echo "[2/3] Applying transform JSON to LiDAR (sanity check apply)..."
python3 "$SCRIPT_DIR/apply_transform_file.py" \
  --file "$LIDAR_FILE" \
  --transform-json "$ALIGN_JSON" \
  --out "$OUT_DIR/${BASE_NO_EXT}_applied.las"

echo "[3/3] Verifying alignment against target PLY..."
python3 "$SCRIPT_DIR/verify_alignment.py" \
  --file "$LIDAR_FILE" \
  --transform-json "$ALIGN_JSON" \
  --target-ply "$TARGET_PLY" \
  --nn-radius 0.8 --max-samples 3000000 | tee "$OUT_DIR/${BASE_NO_EXT}_verify_stats.json"

echo "Done. Outputs in: $OUT_DIR" 