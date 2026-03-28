#!/usr/bin/env python3
"""
Prepare optimized input files for depth comparison.

Pre-joins NWM catchments with flow stream orders and converts structures
from GDB to GeoParquet for fast spatial-filtered loading at runtime.

Example usage:
python prepare_depth_inputs.py \
    --catchments_path /path/to/nwm_catchments.gpkg \
    --flows_path /path/to/nwm_flows.gpkg \
    --structures_path /path/to/TX_Structures.gdb \
    --bbox -97.81,29.62,-96.40,30.56 \
    --output_dir /path/to/prepared_inputs
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import geopandas as gpd
import pandas as pd


def prepare_catchments(
    catchments_path: str,
    flows_path: str,
    output_path: str,
    bbox: tuple[float, float, float, float] | None = None,
):
    """
    Pre-join NWM catchments with flow stream orders and save as GeoParquet.

    Parameters
    ----------
    catchments_path : str
        Path to NWM catchments GeoPackage.
    flows_path : str
        Path to NWM flows GeoPackage (must have ID and order_ columns).
    output_path : str
        Path to save joined catchments GeoParquet.
    bbox : tuple, optional
        Bounding box (minx, miny, maxx, maxy) in WGS84 to spatially filter.
    """
    import pyproj
    from shapely.geometry import box as shapely_box
    from shapely.ops import transform

    t0 = time.time()
    print("Loading catchments...")

    if bbox:
        # Read a small sample to detect the file's CRS
        sample = gpd.read_file(catchments_path, rows=1)
        file_crs = sample.crs

        if file_crs and not file_crs.is_geographic:
            # Reproject WGS84 bbox into the file's native CRS
            proj = pyproj.Transformer.from_crs("EPSG:4326", file_crs, always_xy=True)
            native_box = transform(proj.transform, shapely_box(*bbox))
            native_bbox = native_box.bounds
            print(f"  Reprojected bbox to {file_crs.to_epsg() or 'custom CRS'}: {native_bbox}")
            catchments = gpd.read_file(catchments_path, bbox=native_bbox)
        else:
            catchments = gpd.read_file(catchments_path, bbox=bbox)
    else:
        catchments = gpd.read_file(catchments_path)

    print(f"  Loaded {len(catchments)} catchments in {time.time() - t0:.1f}s")

    t1 = time.time()
    print("Loading flow attributes...")
    flows_df = gpd.read_file(flows_path, columns=["ID", "order_"], ignore_geometry=True)
    print(f"  Loaded {len(flows_df)} flow records in {time.time() - t1:.1f}s")

    catchments = catchments.merge(flows_df, on="ID", how="left")
    catchments = catchments.dropna(subset=["order_"])
    catchments["order_"] = catchments["order_"].astype(int)
    print(f"  {len(catchments)} catchments with stream order")
    print(f"  Stream orders: {sorted(catchments['order_'].unique())}")

    catchments.to_parquet(output_path, engine="pyarrow", compression="snappy")
    print(f"  Saved to {output_path} in {time.time() - t0:.1f}s total")


def prepare_structures(
    structures_path: str,
    output_path: str,
    bbox: tuple[float, float, float, float] | None = None,
):
    """
    Extract structures within bbox and save as GeoParquet.

    Parameters
    ----------
    structures_path : str
        Path to structures vector file (GDB, GPKG, shapefile).
    output_path : str
        Path to save structures GeoParquet.
    bbox : tuple, optional
        Bounding box (minx, miny, maxx, maxy) in WGS84 to spatially filter.
    """
    t0 = time.time()
    print("Loading structures...")
    gdf = gpd.read_file(structures_path, bbox=bbox) if bbox else gpd.read_file(structures_path)
    print(f"  Loaded {len(gdf)} structures in {time.time() - t0:.1f}s")

    gdf.to_parquet(output_path, engine="pyarrow", compression="snappy")
    print(f"  Saved to {output_path} in {time.time() - t0:.1f}s total")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare optimized input files for depth comparison."
    )
    parser.add_argument("--catchments_path", type=str, required=False,
                        help="Path to NWM catchments GeoPackage.")
    parser.add_argument("--flows_path", type=str, required=False,
                        help="Path to NWM flows GeoPackage (must have ID and order_ columns).")
    parser.add_argument("--structures_path", type=str, required=False,
                        help="Path to structures vector file (GDB, GPKG, shapefile).")
    parser.add_argument("--bbox", type=str, required=False,
                        help="Bounding box as minx,miny,maxx,maxy in WGS84.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save prepared input files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    bbox = tuple(float(x) for x in args.bbox.split(",")) if args.bbox else None

    if args.catchments_path and args.flows_path:
        prepare_catchments(
            args.catchments_path,
            args.flows_path,
            os.path.join(args.output_dir, "catchments_with_stream_order.parquet"),
            bbox=bbox,
        )

    if args.structures_path:
        prepare_structures(
            args.structures_path,
            os.path.join(args.output_dir, "structures.parquet"),
            bbox=bbox,
        )


if __name__ == "__main__":
    main()
