#!/usr/bin/env python3

import sys
from pathlib import Path
from collections import defaultdict
import geopandas as gpd
import pandas as pd

import warnings
warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

INPUT_DIR = Path("/efs/fim-data/hand_fim/temp/brad/hwm3")
OUT_GPKG_DIR = Path("/efs/fim-data/hand_fim/temp/brad/hwm_by_huc8_gpkg13")
OUT_PARQUET_DIR = Path("/efs/fim-data/hand_fim/temp/brad/hwm_by_huc8_parquet13")

# Reference file for CRS
REF_CRS_FILE = Path("/efs/fim-data/hand_fim/inputs/rating_curve/water_edge_database/calibration_points/01080203.parquet")

# Quality values to exclude
BAD_QUALITY = {'Poor: +/- 0.40 ft', 'Fair: +/- 0.20 ft', 'Unknown/Historical'}
BAD_ENVIRONMENT = {'Coastal'}

def main():
    if not INPUT_DIR.exists():
        sys.exit(f"Input directory not found: {INPUT_DIR}")
    OUT_GPKG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    # Read CRS from reference file
    try:
        ref_crs = gpd.read_parquet(REF_CRS_FILE).crs
        if ref_crs is None:
            sys.exit(f"❌ CRS could not be determined from {REF_CRS_FILE}")
        print(f"ℹ Using reference CRS: {ref_crs}")
    except Exception as e:
        sys.exit(f"❌ Failed to read reference CRS from {REF_CRS_FILE}: {e}")

    huc8_groups = defaultdict(list)
    all_data = []  # collect all points for quality summary

    for gpkg_path in INPUT_DIR.rglob("*.gpkg"):
        try:
            gdf = gpd.read_file(gpkg_path)

            # Check for required columns
            if "HUC8" not in gdf.columns or "flow" not in gdf.columns:
                print(f"⚠︎  {gpkg_path} missing HUC8 or flow column; skipping")
                continue

            # Set 'coll_time', 'layer', 'path', and 'submitter' columns
            for col in ['coll_time', 'layer', 'path']:
                if col in gdf.columns:
                    gdf[col] = "not applicable"
            gdf['submitter'] = "usgs_hwm"

            # Drop rows where flow is NaN or 0
            gdf = gdf[gdf["flow"].notna() & (gdf["flow"] > 0)]

            # Drop rows with bad hwmQualityName
            if "hwmQualityName" in gdf.columns:
                gdf = gdf[~gdf["hwmQualityName"].isin(BAD_QUALITY)]

            # Drop rows with bad hwm_environment
            if "hwm_environment" in gdf.columns:
                gdf = gdf[~gdf["hwm_environment"].isin(BAD_ENVIRONMENT)]

            all_data.append(gdf)  # collect for summary

            for huc8, group in gdf.groupby("HUC8"):
                if pd.isna(huc8) or not str(huc8).strip():
                    continue
                huc8_groups[str(huc8).zfill(8)].append(group)

        except Exception as e:
            print(f"❌ Failed to process {gpkg_path}: {e}")

    for huc8, gdfs in huc8_groups.items():
        combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

        # Reproject to reference CRS
        if combined.crs != ref_crs:
            combined = combined.to_crs(ref_crs)

        out_gpkg_path = OUT_GPKG_DIR / f"{huc8}.gpkg"
        out_parquet_path = OUT_PARQUET_DIR / f"{huc8}.parquet"

        try:
            combined.to_file(out_gpkg_path, driver="GPKG")
            combined.to_parquet(out_parquet_path, index=False)
            print(f"✔ Wrote: {out_gpkg_path.name}, {out_parquet_path.name}")
        except Exception as e:
            print(f"❌ Failed to write outputs for HUC8 {huc8}: {e}")

    # Summary stats
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        print("\nUnique hwmQualityName values:")
        print(full_df["hwmQualityName"].dropna().unique())

        print("\nUnique hwm_environment values:")
        print(full_df["hwm_environment"].dropna().unique())

    print("\nAll done.")

if __name__ == "__main__":
    main()
