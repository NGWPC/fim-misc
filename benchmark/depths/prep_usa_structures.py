#!/usr/bin/env python3
"""
Prepare USA Structures for depth comparison.

Reads the USA Structures GDB in batches using pyogrio to avoid loading
everything into memory at once. Saves as FlatGeobuf with geometry only
(no attribute columns needed for depth evaluation).

Optionally reprojects to a target CRS at the end.

Example usage:
    python prep_usa_structures.py \
        --input /path/to/TX_Structures.gdb \
        --output /path/to/TX_Structures_5070.fgb \
        --target-crs EPSG:5070
"""
from __future__ import annotations

import argparse
import os
import time

import geopandas as gpd
import pandas as pd
import pyogrio

BATCH_SIZE = 500_000


def main():
    parser = argparse.ArgumentParser(
        description="Prepare USA Structures as lightweight FlatGeobuf (geometry only)."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to USA Structures file (GDB, GPKG, etc.).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to save output file (.fgb or .parquet).",
    )
    parser.add_argument(
        "--layer", default=None,
        help="Layer name if input has multiple layers.",
    )
    parser.add_argument(
        "--target-crs", default=None,
        help="Reproject structures to this CRS (e.g. EPSG:5070). If omitted, keeps original CRS.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Rows per batch (default: {BATCH_SIZE}).",
    )
    args = parser.parse_args()

    t0 = time.time()
    tmp_dir = args.output + "_parts"
    os.makedirs(tmp_dir, exist_ok=True)

    kwargs = {}
    if args.layer:
        kwargs["layer"] = args.layer

    info = pyogrio.read_info(args.input, **kwargs)
    total_features = info["features"]
    print(f"Source: {args.input}")
    print(f"  Total features: {total_features:,}")
    print(f"  CRS: {info['crs']}")
    print(f"  Reading geometry only in batches of {args.batch_size:,}")

    total_rows = 0
    part_files = []
    offset = 0

    while offset < total_features:
        gdf = gpd.read_file(
            args.input,
            columns=["geometry"],
            skip_features=offset,
            max_features=args.batch_size,
            **kwargs,
        )
        if len(gdf) == 0:
            break

        part_path = os.path.join(tmp_dir, f"part_{len(part_files):04d}.parquet")
        gdf.to_parquet(part_path, engine="pyarrow", compression="snappy")
        total_rows += len(gdf)
        part_files.append(part_path)
        offset += len(gdf)

        elapsed = time.time() - t0
        pct = offset / total_features * 100
        print(f"  {total_rows:,} / {total_features:,} ({pct:.0f}%) in {elapsed:.0f}s")
        del gdf

    t1 = time.time()
    print(f"Read complete: {total_rows:,} rows in {t1 - t0:.0f}s ({len(part_files)} parts)")

    # Concatenate parts
    print("Merging parts ...")
    parts = [gpd.read_parquet(p) for p in part_files]
    merged = pd.concat(parts, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry")

    if args.target_crs and str(merged.crs) != args.target_crs:
        print(f"Reprojecting {len(merged):,} structures to {args.target_crs} ...")
        merged = merged.to_crs(args.target_crs)

    # Write output
    if args.output.endswith(".fgb"):
        merged.to_file(args.output, driver="FlatGeobuf")
    elif args.output.endswith(".parquet"):
        merged.to_parquet(args.output, engine="pyarrow", compression="snappy")
    else:
        merged.to_file(args.output)
    del parts, merged

    t2 = time.time()
    print(f"  Saved to {args.output} in {t2 - t1:.0f}s")

    # Cleanup parts
    for p in part_files:
        os.remove(p)
    os.rmdir(tmp_dir)

    # Verify
    if args.output.endswith(".parquet"):
        verify = gpd.read_parquet(args.output)
    else:
        verify = gpd.read_file(args.output)
    assert len(verify) == total_rows, f"MISMATCH: {len(verify)} != {total_rows}"
    print(f"  Verified: {len(verify):,} structures, columns: {list(verify.columns)}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
