#!/usr/bin/env python3
"""
Convert point GeoPackages inside “*-item” directories to GeoParquet and a
clean GeoPackage, enriching them with flow and HUC codes.

Run:
    python process_flood_points.py /path/to/main/flood/folder
"""

import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd

# ── paths you can tweak ───────────────────────────────────────────────────────
CATCHMENT_PATH = Path("/efs/fim-data/hand_fim/temp/brad/subset_nwm_catchments.gpkg")
HUC12_PATH     = Path("/efs/fim-data/hand_fim/inputs/wbd/WBD_National.gpkg")
OUTPUT_BASE    = Path("/efs/fim-data/hand_fim/temp/brad/hwm2")
# ──────────────────────────────────────────────────────────────────────────────

CATCHMENT_ID_FIELD = "ID"          # in subset catchments
CSV_ID_FIELD       = "feature_id"  # in *_flowfile.csv*
HUC_LAYER_NAME     = "WBDHU12"
HUC12_FIELD        = "HUC12"

FLOW_CSV_GLOB   = "*_flowfile.csv"
POINT_GPKG_GLOB = "*.gpkg"
OUT_PQ_SUFFIX   = ".parquet"

# template for new/blank columns
NEW_COLS = {
    "fid"       : "int64",
    "flow"      : "float64",
    "magnitude" : "object",
    "submitter" : "object",
    "coll_time" : "object",
    "flow_unit" : "object",
    "layer"     : "object",
    "path"      : "object",
    "HUC12"     : "object",
    "HUC10"     : "object",
    "HUC8"      : "object",
    "HUC6"      : "object",
}

def main(root_dir: Path) -> None:
    # ── load reference layers once ────────────────────────────────────────────
    if not CATCHMENT_PATH.exists():
        sys.exit(f"Catchment file not found: {CATCHMENT_PATH}")
    if not HUC12_PATH.exists():
        sys.exit(f"HUC12 file not found: {HUC12_PATH}")

    print("Loading subset catchments …")
    catchments = (
        gpd.read_file(CATCHMENT_PATH, columns=[CATCHMENT_ID_FIELD, "geometry"])
        .set_index(CATCHMENT_ID_FIELD)
    )

    print("Loading HUC12 polygons …")
    huc12s = (
        gpd.read_file(
            HUC12_PATH,
            layer=HUC_LAYER_NAME,
            columns=[HUC12_FIELD, "geometry"],
        )
        .set_index(HUC12_FIELD)
    )

    # speed up spatial joins if pygeos/rtree is available
    gpd.options.use_pygeos = True

    # ── walk through all “*-item” dirs ────────────────────────────────────────
    for item_dir in root_dir.rglob("*-item"):
        try:
            gpkg_path = next(item_dir.glob(POINT_GPKG_GLOB))
        except StopIteration:
            continue

        csv_path = next(item_dir.glob(FLOW_CSV_GLOB), None)
        if csv_path is None:
            print(f"⚠︎  {item_dir} has no flowfile CSV; skipping")
            continue

        print(f"→ Processing {gpkg_path.relative_to(root_dir)}")

        # read point layer
        points = gpd.read_file(gpkg_path)
        if points.crs != catchments.crs:
            points = points.to_crs(catchments.crs)

        # add catchment feature_id
        points = (
            gpd.sjoin(points, catchments.reset_index(), how="left", op="within")
            .rename(columns={CATCHMENT_ID_FIELD: "feature_id"})
            .drop(columns=["index_right"])
        )

        # add HUC12
        points = (
            gpd.sjoin(points, huc12s.reset_index(), how="left", op="within")
            .rename(columns={HUC12_FIELD: "HUC12"})
            .drop(columns=["index_right"])
        )

        # derive HUC10 / HUC8 / HUC6
        points["HUC10"] = points["HUC12"].str[:10]
        points["HUC8"]  = points["HUC12"].str[:8]
        points["HUC6"]  = points["HUC12"].str[:6]

        # attach discharge
        discharge_tbl = (
            pd.read_csv(csv_path, usecols=[CSV_ID_FIELD, "discharge"])
            .rename(columns={CSV_ID_FIELD: "feature_id", "discharge": "flow"})
        )
        points = points.merge(discharge_tbl, on="feature_id", how="left")

        # ensure blank columns exist with correct dtypes
        for col, dtype in NEW_COLS.items():
            if col not in points.columns:
                points[col] = pd.Series(dtype=dtype)

        # ── write outputs ────────────────────────────────────────────────────
        out_dir = OUTPUT_BASE / item_dir.relative_to(root_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_parquet = out_dir / (gpkg_path.stem + OUT_PQ_SUFFIX)
        points.to_parquet(out_parquet, index=False)

        # GeoPackage: drop user 'fid' to avoid Fiona schema error
        gpkg_safe = points.drop(columns=["fid"], errors="ignore")
        out_gpkg = out_dir / (gpkg_path.stem + ".gpkg")
        gpkg_safe.to_file(out_gpkg, driver="GPKG")

        print(f"   ✔ saved {out_parquet.name} and {out_gpkg.name}")

    print("Done.")

# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} /path/to/main/flood/folder")
    main(Path(sys.argv[1]).expanduser().resolve())
