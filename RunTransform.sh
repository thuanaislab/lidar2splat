#!/usr/bin/env bash
set -euo pipefail

# Paths
ROOT_DIR="/pipeline/TRANSFORM"
SCRIPT_DIR="$ROOT_DIR/scripts"
SITE_NAME="Site3"
OUT_DIR="$ROOT_DIR/output/$SITE_NAME"
FORMAT="ply"

# Defaults (can be overridden by args)
INPUT_FILE="/pipeline/TRANSFORM/data/$SITE_NAME/usgs/site3_usgs_dtm_shifted.$FORMAT"
ALIGN_JSON="$OUT_DIR/Lidar-NoColor_align.json"
KEEP_CRS_FLAG=""  # set to "--keep-crs" to preserve original CRS VLRs

# Arg1: input LAS/LAZ path
# Arg2: transform JSON path
if [ "${1-}" != "" ]; then
  INPUT_FILE="$1"
fi
if [ "${2-}" != "" ]; then
  ALIGN_JSON="$2"
fi

if [ ! -f "$INPUT_FILE" ]; then
  echo "ERROR: Input file not found: $INPUT_FILE" >&2
  exit 1
fi
if [ ! -f "$ALIGN_JSON" ]; then
  echo "ERROR: Transform JSON not found: $ALIGN_JSON" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

source "$ROOT_DIR/trans-env/bin/activate"

BASE="$(basename "$INPUT_FILE")"
BASE_NO_EXT="${BASE%.*}"
OUT_LAS="$OUT_DIR/${BASE_NO_EXT}_aligned.$FORMAT"

echo "Applying transform to: $INPUT_FILE"
python3 "$SCRIPT_DIR/apply_transform_file.py" \
  --file "$INPUT_FILE" \
  --transform-json "$ALIGN_JSON" \
  --out "$OUT_LAS" \
  $KEEP_CRS_FLAG

# echo "Creating small preview (LAZ) for quick visualization..."
# python3 "$SCRIPT_DIR/downsample_las.py" \
#   --file "$OUT_LAS" \
#   --out "$OUT_DIR/${BASE_NO_EXT}_aligned_preview.laz" \
#   --fraction 0.02

echo "Done. Outputs:"
echo "  Aligned LAS:   $OUT_LAS"
echo "  Preview LAZ:   $OUT_DIR/${BASE_NO_EXT}_aligned_preview.$FORMAT" 