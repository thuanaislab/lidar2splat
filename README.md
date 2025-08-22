# TRANSFORM: LiDAR ↔ Target Point Cloud Alignment

This repository aligns a LiDAR LAS/LAZ file to a target point cloud (PLY) and verifies the alignment. The pipeline:

1) Estimate a 2D similarity + Z-offset transform using nearest-neighbor statistics
2) Apply the transform to the entire LAS/LAZ
3) Verify alignment quality with robust metrics

The entrypoint is `RunEstimateTransform.sh` which orchestrates all steps against the sample dataset in `data/site3`.

## Repo structure
- `RunEstimateTransform.sh`: Orchestrates estimate → apply → verify for the sample dataset
- `setup.sh`: Creates a Python virtualenv `trans-env` and installs dependencies
- `requirements.txt`: Python dependencies
- `scripts/`
  - `estimate_transform.py`: Estimates transform and writes aligned file + JSON
  - `apply_transform_file.py`: Applies a transform JSON to a LAS/LAZ
  - `verify_alignment.py`: Computes alignment quality metrics
- `data/site3/`
  - `lidar/MurphysSite3-NoColor.las`: Example LiDAR input (projected CRS required)
  - `splat/sparse_transformed.ply`: Target point cloud
  - `splat/origin_point.txt`: Origin lat lon on the first line, e.g. `37.12345 -121.23456`
- `output/`: Results directory created by the run script

## Requirements
- Linux, Python 3.8–3.11
- Sufficient RAM and disk; sample files are large (≈5.6 GB LAS, ≈1.4 GB PLY)
- Internet access for pip install

## Setup
```bash
cd /pipeline/TRANSFORM
bash setup.sh
# activate when needed
source trans-env/bin/activate
```

If you prefer an absolute path when activating:
```bash
source /pipeline/TRANSFORM/trans-env/bin/activate
```

## Quickstart (sample dataset)
```bash
cd /pipeline/TRANSFORM
bash RunEstimateTransform.sh
```
This will:
- Estimate the transform and write:
  - `output/MurphysSite3-NoColor_aligned.las`
  - `output/MurphysSite3-NoColor_align.json`
  - `output/MurphysSite3-NoColor_target_align.txt` (origin_e, origin_n, z_offset)
- Apply the transform again as a sanity check:
  - `output/MurphysSite3-NoColor_applied.las`
- Verify and save metrics to:
  - `output/MurphysSite3-NoColor_verify_stats.json`

Note: `RunEstimateTransform.sh` expects to run from the repo root so that `source trans-env/bin/activate` resolves correctly. If running from another directory, activate the venv via its absolute path.

## Using the scripts directly
You can run the Python scripts individually. Activate the venv first.

### estimate_transform.py
Estimates the transform and writes aligned LAS and a transform JSON.
```bash
python3 scripts/estimate_transform.py \
  --lidar-file /path/to/input.las \
  --origin-file /path/to/origin_point.txt \
  --target-ply /path/to/target.ply \
  --out-dir /path/to/output \
  --nn-radius 2.0 --max-pairs 3000000 \
  --xy-iters 12 --xy-radius 0.8 --z-gate 2.5 --max-pairs-xy 3000000
```
- Input LAS/LAZ must have a projected CRS in its header; if missing or geographic (EPSG:4326), the script will error.
- `origin_point.txt` must contain latitude and longitude on the first line, separated by space or comma.
- Outputs (assuming input `foo.las`):
  - `out-dir/foo_aligned.las`: Aligned LiDAR
  - `out-dir/foo_align.json`: Transform JSON (contains 4×4 matrix and stats)
  - `out-dir/foo_target_align.txt`: `origin_e origin_n z_offset`

Key parameters:
- `--nn-radius`: XY nearest-neighbor radius (meters) for coarse Z alignment
- `--xy-iters`: Max iterations for XY similarity + Z refinement
- `--xy-radius`: XY inlier radius for matching to target
- `--z-gate`: Z residual gate for inliers
- Sampling limits: `--max-pairs`, `--max-pairs-xy`

### apply_transform_file.py
Applies a transform JSON to an entire LAS/LAZ.
```bash
python3 scripts/apply_transform_file.py \
  --file /path/to/input.las \
  --transform-json /path/to/foo_align.json \
  --out /path/to/output/foo_applied.las \
  --keep-crs  # optional: keep original CRS VLRs
```
- If `--out` is omitted, the tool writes `<input>_applied.<ext>`.
- By default, CRS VLRs are stripped to avoid stale/incorrect CRS after transform; use `--keep-crs` to retain them.

### verify_alignment.py
Computes alignment metrics by transforming samples from the LAS/LAZ and comparing to nearest neighbors in the target PLY.
```bash
python3 scripts/verify_alignment.py \
  --file /path/to/input.las \
  --transform-json /path/to/foo_align.json \
  --target-ply /path/to/target.ply \
  --nn-radius 0.8 --max-samples 3000000 --chunk-size 2000000
```
Outputs JSON with fields like:
```json
{
  "num_matches": 1234567,
  "nn_radius_m": 0.8,
  "xy_rmse_m": 0.12,
  "xy_median_m": 0.08,
  "z_rmse_m": 0.15,
  "z_median_abs_m": 0.10,
  "rmse_3d_m": 0.19,
  "p95_3d_m": 0.45
}
```

## Transform JSON
`estimate_transform.py` writes a JSON with a 4×4 matrix and metadata. The matrix applies as `P_out = P_in @ A^T + b` where `A = M[:3,:3]`, `b = M[:3,3]`. The JSON also stores 2D rotation, scale, translation, Z offset, and summary stats.

## Performance & tips
- Large files: The sample data are large; ensure enough RAM and disk. `verify_alignment.py` streams in chunks.
- Reproducibility: Random sampling uses fixed seeds for consistency.
- Speed/quality trade-offs: Increase `--max-pairs`/`--max-pairs-xy` and `--max-samples` for better statistics at higher cost.
- CRS: Input LAS must have a valid projected CRS. If missing, reproject upstream or add the correct CRS.

## Adapting to a new site
- Replace inputs in `data/<site>/` and update `RunEstimateTransform.sh` paths (`DATA_DIR`, `LIDAR_FILE`, `ORIGIN_FILE`, `TARGET_PLY`).
- Or invoke the Python scripts directly with your custom paths.

## Troubleshooting
- "Input file CRS is missing or not projected": Ensure the LAS has a projected CRS (e.g., UTM) in its header.
- "No LiDAR points have a target neighbor within X m": Increase `--nn-radius` or validate XY alignment / coordinate frames.
- Verify step finds zero matches: Increase `--nn-radius` or check that the transform and PLY are in the same local frame.

## License
No license specified. Add one if you plan to distribute. 