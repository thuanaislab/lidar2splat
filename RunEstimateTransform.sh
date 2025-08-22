#!/usr/bin/env bash
set -euo pipefail

# Paths
ROOT_DIR="/pipeline/TRANSFORM"
SITE_NAME="Site3"
DATA_DIR="$ROOT_DIR/data/$SITE_NAME"
OUT_DIR="$ROOT_DIR/output/$SITE_NAME"
mkdir -p "$OUT_DIR"
SCRIPT_DIR="$ROOT_DIR/scripts"

LIDAR_FILE="$DATA_DIR/lidar/Lidar-NoColor.las"
ORIGIN_FILE="$DATA_DIR/splat/origin_point.txt"
TARGET_PLY="$DATA_DIR/splat/sparse_transformed.ply"
KML_FILE="$DATA_DIR/$SITE_NAME.kml"

mkdir -p "$OUT_DIR"

source "$ROOT_DIR/trans-env/bin/activate"

echo "[1/3] Estimating transform and writing aligned LiDAR + JSON + TXT..."
python3 "$SCRIPT_DIR/estimate_transform.py" \
  --lidar-file "$LIDAR_FILE" \
  --origin-file "$ORIGIN_FILE" \
  --target-ply "$TARGET_PLY" \
  --out-dir "$OUT_DIR" \
  --nn-radius 0.5 --max-pairs 3000000 \
  --xy-iters 24 --xy-radius 0.25 --z-gate 0.5 --max-pairs-xy 3000000 \
  --kml-boundary "$KML_FILE" \
  --kml-crs epsg:4326 \
  --lidar-sample 10000000 \
  --lidar-chunk-size 2000000

BASE="$(basename "$LIDAR_FILE")"
BASE_NO_EXT="${BASE%.*}"
ALIGN_JSON="$OUT_DIR/${BASE_NO_EXT}_align.json"
ALIGNED_LAS="$OUT_DIR/${BASE_NO_EXT}_aligned.las"

# Produce a small downsampled preview for quick visual checks (cropped aligned LiDAR)
echo "[1b] Creating downsampled preview LAZ (aligned LiDAR)..."
python3 "$SCRIPT_DIR/downsample_las.py" \
  --file "$ALIGNED_LAS" \
  --out "$OUT_DIR/${BASE_NO_EXT}_aligned_preview.laz" \
  --fraction 0.02

# Also produce a downsampled preview of the target reference PLY (cropped version if available)
echo "[1c] Creating downsampled preview PLY (target reference)..."
TARGET_BASE="$(basename "$TARGET_PLY")"
TARGET_NO_EXT="${TARGET_BASE%.*}"
CROPPED_TARGET_PLY="$OUT_DIR/${TARGET_NO_EXT}_cropped.ply"
TARGET_PREVIEW_SRC="$TARGET_PLY"
if [ -f "$CROPPED_TARGET_PLY" ]; then
  TARGET_PREVIEW_SRC="$CROPPED_TARGET_PLY"
fi
python3 "$SCRIPT_DIR/downsample_ply.py" \
  --file "$TARGET_PREVIEW_SRC" \
  --out "$OUT_DIR/${TARGET_NO_EXT}_preview.ply" \
  --fraction 0.02

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
  --nn-radius 0.5 --max-samples 3000000 | tee "$OUT_DIR/${BASE_NO_EXT}_verify_stats.json"

echo "Done. Outputs in: $OUT_DIR" 