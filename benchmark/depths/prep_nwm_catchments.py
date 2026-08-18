#!/usr/bin/env python3
"""
Prepare NWM catchments with stream order for depth comparison.

Joins NWM catchments with flow stream orders, keeps only feature_id,
stream_order, and geometry, then saves as FlatGeobuf.

With --dissolve, also writes a dissolved version (one row per stream order)
with pre-computed n_catchments and area_sqkm for fast runtime use.

Example usage:
    python prep_nwm_catchments.py \
        --catchments /path/to/nwm_catchments.gpkg \
        --flows /path/to/nwm_flows.gpkg \
        --output /path/to/catchments_with_stream_order.fgb \
        --bbox -97.81,29.62,-96.40,30.56 \
        --dissolve
"""
from __future__ import annotations

import argparse
import os
import time

import geopandas as gpd


def main():
    parser = argparse.ArgumentParser(
        description="Prepare NWM catchments with stream order as lightweight FlatGeobuf."
    )
    parser.add_argument(
        "--catchments", required=True,
        help="Path to NWM catchments file (GPKG, GDB, etc.).",
    )
    parser.add_argument(
        "--flows", required=True,
        help="Path to NWM flows file (must have ID and order_ columns).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to save output file (.fgb or .parquet).",
    )
    parser.add_argument(
        "--bbox", default=None,
        help="Bounding box as minx,miny,maxx,maxy in WGS84 to spatially filter.",
    )
    parser.add_argument(
        "--target-crs", default=None,
        help="Reproject to this CRS (e.g. EPSG:5070). If omitted, keeps original CRS.",
    )
    parser.add_argument(
        "--dissolve", action="store_true",
        help="Also write a dissolved-by-stream-order file (one row per SO).",
    )
    args = parser.parse_args()

    t0 = time.time()
    bbox = tuple(float(x) for x in args.bbox.split(",")) if args.bbox else None

    # Load catchments with optional bbox filter
    print(f"Loading catchments from {args.catchments} ...")
    if bbox:
        import pyproj
        from shapely.geometry import box as shapely_box
        from shapely.ops import transform

        sample = gpd.read_file(args.catchments, rows=1)
        file_crs = sample.crs
        if file_crs and not file_crs.is_geographic:
            proj = pyproj.Transformer.from_crs("EPSG:4326", file_crs, always_xy=True)
            native_box = transform(proj.transform, shapely_box(*bbox))
            catchments = gpd.read_file(args.catchments, bbox=native_box.bounds)
        else:
            catchments = gpd.read_file(args.catchments, bbox=bbox)
    else:
        catchments = gpd.read_file(args.catchments)
    print(f"  {len(catchments):,} catchments in {time.time() - t0:.1f}s")

    # Load flow attributes (ID + order_ only, no geometry)
    t1 = time.time()
    print(f"Loading flows from {args.flows} ...")
    flows_df = gpd.read_file(args.flows, columns=["ID", "order_"], ignore_geometry=True)
    print(f"  {len(flows_df):,} flow records in {time.time() - t1:.1f}s")

    # Join and filter
    catchments = catchments.merge(flows_df, on="ID", how="left")
    catchments = catchments.dropna(subset=["order_"])
    catchments["order_"] = catchments["order_"].astype(int)

    # Rename and keep only the fields we need
    catchments = catchments.rename(columns={"ID": "feature_id", "order_": "stream_order"})
    catchments = catchments[["feature_id", "stream_order", "geometry"]]
    print(f"  {len(catchments):,} catchments with stream order")
    print(f"  Stream orders: {sorted(catchments['stream_order'].unique())}")

    # Reproject if requested
    if args.target_crs and str(catchments.crs) != args.target_crs:
        print(f"Reprojecting to {args.target_crs} ...")
        catchments = catchments.to_crs(args.target_crs)

    # Write output
    t2 = time.time()
    if args.output.endswith(".fgb"):
        catchments.to_file(args.output, driver="FlatGeobuf")
    elif args.output.endswith(".parquet"):
        catchments.to_parquet(args.output, engine="pyarrow", compression="snappy")
    else:
        catchments.to_file(args.output)
    print(f"  Saved to {args.output} in {time.time() - t2:.1f}s")

    # Dissolved version: one row per stream order
    if args.dissolve:
        t3 = time.time()
        # Compute per-catchment area before dissolving
        catchments["area_sqkm"] = catchments.geometry.area / 1e6
        n_catchments = catchments.groupby("stream_order").size().rename("n_catchments")
        area_by_so = catchments.groupby("stream_order")["area_sqkm"].sum()

        dissolved = catchments.dissolve(by="stream_order", as_index=False)[
            ["stream_order", "geometry"]
        ]
        dissolved = dissolved.merge(n_catchments, on="stream_order")
        dissolved = dissolved.merge(area_by_so, on="stream_order")
        dissolved = gpd.GeoDataFrame(dissolved, geometry="geometry", crs=catchments.crs)

        # Build dissolved output path: foo.fgb → foo_dissolved.fgb
        base, ext = os.path.splitext(args.output)
        dissolved_path = f"{base}_dissolved{ext}"
        if dissolved_path.endswith(".fgb"):
            dissolved.to_file(dissolved_path, driver="FlatGeobuf")
        elif dissolved_path.endswith(".parquet"):
            dissolved.to_parquet(dissolved_path, engine="pyarrow", compression="snappy")
        else:
            dissolved.to_file(dissolved_path)
        print(f"  Dissolved: {len(dissolved)} stream orders → {dissolved_path} in {time.time() - t3:.1f}s")
        print(f"  Columns: {list(dissolved.columns)}")

    # Verify
    if args.output.endswith(".parquet"):
        verify = gpd.read_parquet(args.output)
    else:
        verify = gpd.read_file(args.output)
    assert len(verify) == len(catchments), f"MISMATCH: {len(verify)} != {len(catchments)}"
    print(f"  Verified: {len(verify):,} catchments, columns: {list(verify.columns)}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
