"""
Depth comparisons

Example usage:
python depth_compare.py \
    --candidate-map data/candidate_dummy_huc_12090301_depth_500yr.tif \
    --benchmark-map data/benchmark_ble_huc_12090301_depth_500yr.tif \
    --agreement-map data/agreement_dummy_ble_huc_12090301_depth_500yr.tif \
    --metrics-parquet data/depth_metrics_huc_12090301_depth_500yr.parquet \
    --structures data/structures.gdb \
    --structures-metrics-parquet data/structures_metrics.parquet \
    --num-workers 6
    --epsilon 0.1
"""
from __future__ import annotations
from typing import Iterable, Literal

import argparse
import os
import sys
import gc

import pyproj

# Set PROJ and GDAL data paths for odc-geo (only if they exist)
_proj_lib = os.path.join(sys.prefix, "share", "proj")
_gdal_data = os.path.join(sys.prefix, "share", "gdal")
if os.path.isdir(_proj_lib):
    os.environ["PROJ_LIB"] = _proj_lib
    pyproj.datadir.set_data_dir(_proj_lib)
if os.path.isdir(_gdal_data):
    os.environ["GDAL_DATA"] = _gdal_data

import numpy as np
import pandas as pd
import rioxarray as rxr
import dask
import gval
import geopandas as gpd
from exactextract import exact_extract
from gval.comparison.pairing_functions import difference
from gval.comparison.compute_continuous_metrics import _compute_continuous_metrics


def compare_depth_maps(
    candidate_map_path: str | os.PathLike,
    benchmark_map_path: str | os.PathLike,
    agreement_map_path: str | os.PathLike | None = None,
    metrics_parquet_path: str | os.PathLike | None = None,
    chunk_size: int | None = None,
    target_map: Literal["benchmark", "candidate"] = "benchmark",
    resampling: str | None = "bilinear",
    subsampling_df: gpd.GeoDataFrame | None = None,
    subsampling_average: str = "none",
    metrics: str | Iterable[str] = "all",
    nodata: float | int | None = -9999,
    encode_nodata: bool = True,
    epsilon: float = 0.1,
    clear_memory: bool = True,
):
    """
    Compare candidate depth map against benchmark depth map and save report.

    Parameters
    ----------
    candidate_map_path : str or os.PathLike
        Path to candidate depth map raster file.
    benchmark_map_path : str or os.PathLike
        Path to benchmark depth map raster file.
    agreement_map_path : str or os.PathLike, default = None
        Path to save agreement map raster file. If None, agreement map is not saved.
    metrics_parquet_path : str or os.PathLike, default = None
        Path to save metrics parquet file. If None, metrics table is not saved.
    chunk_size : int, default = None
        Chunk size for dask arrays. If None, use 'auto' to use block size
    epsilon : float, default = 0.1
        Small value to avoid division by zero in some metrics.
    target_map : Literal["benchmark", "candidate"], default = "benchmark"
        Target map for homogenization.
    resampling : str, default = "bilinear"
        Resampling method for homogenization.
    subsampling_df : gpd.GeoDataFrame, default = None
        DataFrame with subsampling points. If None, no subsampling is performed.
    subsampling_average : str, default = "none"
        Subsampling average method: 'none', 'mean', or 'median'.
    metrics : str or Iterable[str], default = "all"
        Metrics to compute, or 'all' for all metrics.
    nodata : float or int, default = -9999
        NoData value for the rasters.
    encode_nodata : bool, default = True
        Whether to encode NoData in the agreement map.
    epsilon : float, default = 0.1
        Small value to avoid division by zero in some metrics.
    clear_memory : bool, default = True
        Whether to force garbage collection after computation.
    
    Returns
    -------
    Tuple[xr.DataArray, DataFrame[Metrics_df]]
        Agreement map xarray and metrics table dataframe.
    """
    # Open candidate and benchmark maps
    with (
        rxr.open_rasterio(
            candidate_map_path, masked=True, chunks="auto" if chunk_size is None else {"x": chunk_size, "y": chunk_size}
        ).squeeze() as da_candidate,
        rxr.open_rasterio(
            benchmark_map_path, masked=True, chunks="auto" if chunk_size is None else {"x": chunk_size, "y": chunk_size}
        ).squeeze() as da_benchmark,
    ):
        
        # debugging: subset first 8 chunks
        #chunk_size = da_candidate.chunks[0][0] if chunk_size is None else chunk_size
        #da_candidate = da_candidate.isel(x=slice(0, chunk_size), y=slice(0, chunk_size))
        #da_benchmark = da_benchmark.isel(x=slice(0, chunk_size), y=slice(0, chunk_size))

        # Homogenize candidate and benchmark maps
        da_candidate, da_benchmark = da_candidate.gval.homogenize(
            da_benchmark,
            target_map='benchmark',
            resampling=resampling,
        )

        # Preserve CRS from benchmark for later use (as WKT to avoid reference invalidation)
        crs = da_benchmark.rio.crs.to_wkt()

        # Define "wet" as depth > 0 (not just notnull), so that candidate cells
        # with value 0 (dry within its extent) don't create false domain overlap
        # with benchmark nodata areas.
        #
        # The comparison domain is where at least one map has depth > 0.
        # Within that domain, dry cells are filled with 0 so both over- and
        # under-prediction are captured. Areas where both maps are dry or
        # nodata are excluded entirely.
        benchmark_wet = (da_benchmark > 0) & da_benchmark.notnull()
        candidate_wet = (da_candidate > 0) & da_candidate.notnull()
        either_wet = benchmark_wet | candidate_wet
        da_candidate = da_candidate.where(candidate_wet, other=0.0).where(either_wet)
        da_benchmark = da_benchmark.where(benchmark_wet, other=0.0).where(either_wet)

        # Compute agreement map
        results = da_candidate.gval.compute_agreement_map(
            benchmark_map=da_benchmark,
            comparison_function=difference,
            nodata=nodata,
            encode_nodata=encode_nodata,
            subsampling_df=subsampling_df,
            continuous=True,
        )

        # If sampling_df return type gives three values assign all vars results, otherwise only agreement map results
        agreement_map, da_candidate, da_benchmark = (
            results if subsampling_df is not None else (results, da_candidate, da_benchmark)
        )

        # Compute metrics table
        metrics_table = _compute_continuous_metrics(
            agreement_map=agreement_map,
            candidate_map=da_candidate,
            benchmark_map=da_benchmark,
            metrics=metrics,
            subsampling_df=subsampling_df,
            subsampling_average=subsampling_average,
            epsilon=epsilon,
        )

    if clear_memory:
        del da_candidate, da_benchmark
        gc.collect()

    # Save agreement map if path provided
    if agreement_map_path is not None:
        agreement_map = agreement_map.rio.write_crs(crs)
        agreement_map.rio.to_raster(
            agreement_map_path,
            driver="COG",
            tiled=True,
            compress="LZW",
            dtype="float32",
            BIGTIFF="IF_SAFER",
        )

    # Save metrics table if path provided
    if metrics_parquet_path is not None:
        metrics_table.to_parquet(metrics_parquet_path, engine="pyarrow", index=False, compression="snappy")

    return agreement_map, metrics_table

def compare_structures(
    agreement_map,
    structures_path: str | os.PathLike,
    structures_metrics_parquet_path: str | os.PathLike | None = None,
    structures_gpkg_path: str | os.PathLike | None = None,
    units: str = "feet",
):
    """
    Summarize the agreement map at building footprint locations.

    Reprojects the agreement map to WGS84, polygonizes and simplifies the
    valid-data domain, then uses the simplified polygon as a mask to load
    only structures intersecting the flood domain. Runs zonal statistics of
    the agreement map on those structures and computes summary metrics
    including depth agreement buckets and bias direction.

    Parameters
    ----------
    agreement_map : xr.DataArray
        Agreement map (candidate - benchmark depth difference) as returned
        by compare_depth_maps.
    structures_path : str or os.PathLike
        Path to structures vector file (e.g. GDB, GPKG, shapefile).
    structures_metrics_parquet_path : str or os.PathLike, default = None
        Path to save summary metrics parquet. If None, not saved.
    structures_gpkg_path : str or os.PathLike, default = None
        Path to save per-structure results as GeoPackage with geometry
        and categorical columns for symbolization. If None, not saved.

    Returns
    -------
    Tuple[gpd.GeoDataFrame, pd.DataFrame]
        Per-structure results GeoDataFrame and summary metrics DataFrame.
    """
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    # Compute into memory if needed
    if hasattr(agreement_map, "compute"):
        agreement_map = agreement_map.compute()

    # --- Reproject agreement map to WGS84 (to match structures natively) ---
    print("Reprojecting agreement map to EPSG:4326...")
    agreement_map_wgs84 = agreement_map.rio.reproject("EPSG:4326")

    # --- Polygonize, simplify, and buffer the agreement domain ---
    print("Polygonizing agreement domain...")
    valid_mask = agreement_map_wgs84.notnull().values.astype(np.uint8)
    transform = agreement_map_wgs84.rio.transform()
    domain_polys = [
        shape(geom) for geom, val in shapes(valid_mask, mask=valid_mask == 1, transform=transform)
        if val == 1
    ]
    if not domain_polys:
        print("No valid cells in agreement map — no structures to compare.")
        return gpd.GeoDataFrame(), pd.DataFrame()
    domain_polygon = unary_union(domain_polys)
    res = abs(transform.a)
    domain_simple = domain_polygon.simplify(res * 2).buffer(res * 3)
    print(f"Agreement domain polygonized and simplified: {len(domain_polys)} polygon(s) merged.")

    # --- Load only structures intersecting the buffered domain ---
    print("Loading structures within buffered domain...")
    domain_mask = gpd.GeoDataFrame(geometry=[domain_simple], crs="EPSG:4326")
    domain_structures = gpd.read_file(structures_path, mask=domain_mask)
    if len(domain_structures) == 0:
        print("No structures found within agreement domain.")
        return gpd.GeoDataFrame(), pd.DataFrame()
    print(f"Loaded {len(domain_structures)} structures within agreement domain.")

    # --- Zonal stats: summarize agreement map at domain structures ---
    print("Running zonal stats...")
    stats = exact_extract(
        agreement_map_wgs84, domain_structures, ["mean", "min", "max", "count"], include_cols=[], output="pandas",
    )

    domain_structures = domain_structures.copy()
    domain_structures["mean_depth_diff"] = stats["mean"].fillna(0.0)
    domain_structures["min_depth_diff"] = stats["min"].fillna(0.0)
    domain_structures["max_depth_diff"] = stats["max"].fillna(0.0)
    domain_structures["pixel_count"] = stats["count"]

    # Keep structures with raster coverage
    domain_structures = domain_structures[domain_structures["pixel_count"] > 0].copy()
    print(f"{len(domain_structures)} structures have agreement map coverage.")

    diff = domain_structures["mean_depth_diff"].values
    abs_diff = np.abs(diff)
    n_domain_valid = len(domain_structures)

    # --- Per-structure categorical columns (for GPKG symbolization) ---
    # Thresholds in feet; convert if input units are meters
    ft_scale = 0.3048 if units == "meters" else 1.0
    buckets = pd.cut(
        abs_diff,
        bins=[0, 1 * ft_scale, 3 * ft_scale, 5 * ft_scale, np.inf],
        labels=["< 1ft", "1-3ft", "3-5ft", "> 5ft"],
        include_lowest=True,
    )
    domain_structures["agreement_bucket"] = buckets.astype(str)
    domain_structures["bias_direction"] = np.where(
        diff > ft_scale * 0.1, "over",
        np.where(diff < -ft_scale * 0.1, "under", "match"),
    )

    # --- Summary metrics ---
    mae_d = float(np.mean(abs_diff))
    mse_d = float(np.mean(diff ** 2))
    rmse_d = float(np.sqrt(mse_d))
    mean_signed_d = float(np.mean(diff))
    median_ae = float(np.median(abs_diff))
    p90_ae = float(np.percentile(abs_diff, 90))
    max_ae = float(np.max(abs_diff))

    # Bucket counts and percentages
    n_within_1ft = int(np.sum(abs_diff < 1 * ft_scale))
    n_within_3ft = int(np.sum(abs_diff < 3 * ft_scale))
    n_within_5ft = int(np.sum(abs_diff < 5 * ft_scale))
    n_gt_5ft = int(np.sum(abs_diff >= 5 * ft_scale))

    # Bias direction counts
    n_over = int(np.sum(diff > ft_scale * 0.1))
    n_under = int(np.sum(diff < -ft_scale * 0.1))
    n_match = n_domain_valid - n_over - n_under

    summary = pd.DataFrame({
        "structures_in_domain": [n_domain_valid],
        "mean_absolute_error": [mae_d],
        "root_mean_squared_error": [rmse_d],
        "mean_squared_error": [mse_d],
        "mean_signed_error": [mean_signed_d],
        "median_absolute_error": [median_ae],
        "p90_absolute_error": [p90_ae],
        "max_absolute_error": [max_ae],
        "n_within_1ft": [n_within_1ft],
        "pct_within_1ft": [n_within_1ft / n_domain_valid * 100],
        "n_within_3ft": [n_within_3ft],
        "pct_within_3ft": [n_within_3ft / n_domain_valid * 100],
        "n_within_5ft": [n_within_5ft],
        "pct_within_5ft": [n_within_5ft / n_domain_valid * 100],
        "n_gt_5ft": [n_gt_5ft],
        "pct_gt_5ft": [n_gt_5ft / n_domain_valid * 100],
        "n_over_predict": [n_over],
        "pct_over_predict": [n_over / n_domain_valid * 100],
        "n_under_predict": [n_under],
        "pct_under_predict": [n_under / n_domain_valid * 100],
        "n_match": [n_match],
        "pct_match": [n_match / n_domain_valid * 100],
    })

    print(f"\nDomain structures with coverage: {n_domain_valid}")

    if structures_metrics_parquet_path is not None:
        summary.to_parquet(structures_metrics_parquet_path, engine="pyarrow", index=False, compression="snappy")

    if structures_gpkg_path is not None:
        domain_structures.to_file(structures_gpkg_path, driver="GPKG", layer="structures")
        print(f"Saved {len(domain_structures)} domain structures to {structures_gpkg_path}")

    return domain_structures, summary


def compare_stream_orders(
    candidate_map,
    benchmark_map,
    agreement_map,
    catchments_path: str | os.PathLike,
    flows_path: str | os.PathLike,
    output_dir: str | os.PathLike | None = None,
    metrics: str | Iterable[str] = "all",
    epsilon: float = 0.1,
    units: str = "feet",
):
    """
    Compute GVAL depth metrics per stream order by clipping the candidate
    and benchmark maps to catchments of each stream order.

    Parameters
    ----------
    candidate_map : xr.DataArray
        Candidate depth map (homogenized, domain-clipped).
    benchmark_map : xr.DataArray
        Benchmark depth map (homogenized, domain-clipped).
    agreement_map : xr.DataArray
        Agreement map (candidate - benchmark).
    catchments_path : str or os.PathLike
        Path to NWM catchments GeoPackage.
    flows_path : str or os.PathLike
        Path to NWM flows GeoPackage (must have ID and order_ columns).
    output_dir : str or os.PathLike, optional
        Directory to save per-SO metrics parquets and a combined parquet.
    metrics : str or Iterable[str], default = "all"
        Metrics to compute via GVAL.
    epsilon : float, default = 0.1
        Epsilon for GVAL metrics.
    units : str, default = "feet"
        Units of the input rasters.

    Returns
    -------
    dict
        {stream_order: pd.DataFrame} of GVAL metrics per stream order.
    """
    from shapely.geometry import box

    # Compute into memory if needed
    if hasattr(agreement_map, "compute"):
        agreement_map = agreement_map.compute()
    if hasattr(candidate_map, "compute"):
        candidate_map = candidate_map.compute()
    if hasattr(benchmark_map, "compute"):
        benchmark_map = benchmark_map.compute()

    # Build spatial mask from agreement raster bounds
    bounds = agreement_map.rio.bounds()
    domain_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
    domain_mask = gpd.GeoDataFrame(geometry=[domain_box], crs=agreement_map.rio.crs)

    # Load catchments intersecting domain
    print("Loading catchments within agreement domain...")
    catchments = gpd.read_file(catchments_path, mask=domain_mask)
    print(f"  Found {len(catchments)} catchments in domain")

    if len(catchments) == 0:
        return {}

    # Join stream order from flows
    print("  Reading flow attributes for stream order...")
    flows_df = gpd.read_file(flows_path, columns=["ID", "order_"], ignore_geometry=True)

    catchments = catchments.merge(flows_df, on="ID", how="left")
    catchments = catchments.dropna(subset=["order_"])
    catchments["order_"] = catchments["order_"].astype(int)
    print(f"  Catchments with stream order: {len(catchments)}")
    print(f"  Stream orders found: {sorted(catchments['order_'].unique())}")

    # Ensure CRS matches the raster
    if catchments.crs != agreement_map.rio.crs:
        catchments = catchments.to_crs(agreement_map.rio.crs)

    so_metrics = {}
    ft_scale = 0.3048 if units == "meters" else 1.0

    for so, grp in catchments.groupby("order_"):
        so = int(so)
        print(f"\n  Processing SO{so} ({len(grp)} catchments)...")
        geoms = grp.geometry.values

        # Clip all three rasters to this SO's catchments
        try:
            clipped_agreement = agreement_map.rio.clip(geoms, agreement_map.rio.crs,
                                                        drop=True, all_touched=True)
            clipped_candidate = candidate_map.rio.clip(geoms, candidate_map.rio.crs,
                                                        drop=True, all_touched=True)
            clipped_benchmark = benchmark_map.rio.clip(geoms, benchmark_map.rio.crs,
                                                        drop=True, all_touched=True)
        except Exception as e:
            print(f"    Skipping SO{so}: clip failed ({e})")
            continue

        # Check for valid data
        valid = clipped_agreement.values[~np.isnan(clipped_agreement.values)]
        if len(valid) < 2:
            print(f"    Skipping SO{so}: insufficient valid pixels ({len(valid)})")
            continue

        # Compute GVAL metrics on the clipped rasters
        try:
            metrics_table = _compute_continuous_metrics(
                agreement_map=clipped_agreement,
                candidate_map=clipped_candidate,
                benchmark_map=clipped_benchmark,
                metrics=metrics,
                subsampling_df=None,
                subsampling_average="none",
                epsilon=epsilon,
            )
        except Exception as e:
            print(f"    Skipping SO{so}: metrics computation failed ({e})")
            continue

        # Add stream order and catchment metadata
        metrics_table["stream_order"] = so
        metrics_table["n_catchments"] = len(grp)
        metrics_table["area_sqkm"] = float(grp["AreaSqKM"].sum())
        metrics_table["n_valid_pixels"] = len(valid)

        # Add bias counts (same logic as compare_structures)
        diff = valid
        abs_diff = np.abs(diff)
        n_over = int(np.sum(diff > ft_scale * 0.1))
        n_under = int(np.sum(diff < -ft_scale * 0.1))
        n_match = len(valid) - n_over - n_under
        metrics_table["n_over_predict"] = n_over
        metrics_table["n_under_predict"] = n_under
        metrics_table["n_match"] = n_match

        so_metrics[so] = metrics_table
        print(f"    SO{so}: {len(valid):,} pixels, MAE={float(metrics_table['mean_absolute_error'].iloc[0]):.3f}, "
              f"bias: under={n_under}, match={n_match}, over={n_over}")

        del clipped_agreement, clipped_candidate, clipped_benchmark
        gc.collect()

    # Save outputs
    if output_dir is not None and so_metrics:
        os.makedirs(output_dir, exist_ok=True)
        all_rows = []
        for so in sorted(so_metrics.keys()):
            df = so_metrics[so]
            df.to_parquet(
                os.path.join(output_dir, f"depth_metrics_so{so}.parquet"),
                engine="pyarrow", index=False, compression="snappy",
            )
            all_rows.append(df)
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_parquet(
            os.path.join(output_dir, "depth_metrics_by_stream_order.parquet"),
            engine="pyarrow", index=False, compression="snappy",
        )
        print(f"\n  Saved per-SO metrics to {output_dir}")

    return so_metrics


def main():
    parser = argparse.ArgumentParser(description="Compare candidate depth map against benchmark depth map.")
    parser.add_argument("--candidate-map", type=str, required=True, help="Path to candidate depth map raster file.")
    parser.add_argument("--benchmark-map", type=str, required=True, help="Path to benchmark depth map raster file.")
    parser.add_argument("--agreement-map", type=str, default=None, help="Path to save agreement map raster file.")
    parser.add_argument("--metrics-parquet", type=str, default=None, help="Path to save metrics parquet file.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size for dask arrays. Default is 'auto' to use block size in raster.")
    parser.add_argument("--resampling", type=str, default="bilinear", help="Resampling method for homogenization.")
    parser.add_argument("--target-map", type=str, default="benchmark", choices=["benchmark", "candidate"], help="Target map for homogenization.")
    parser.add_argument("--subsampling-file", type=str, default=None, help="Path to CSV file with subsampling points.")
    parser.add_argument("--subsampling-average", type=str, default="none", help="Subsampling average method: 'none', 'mean', or 'median'.")
    parser.add_argument("--metrics", type=str, default="all", help="Metrics to compute, or 'all' for all metrics.")
    parser.add_argument("--nodata", type=float, default=-9999, help="NoData value for the rasters.")
    parser.add_argument("--no-encode-nodata", action="store_true", help="Whether to encode NoData in the agreement map.")
    parser.add_argument("--keep-memory", action="store_false", help="Whether to keep not force garbage collection after computation.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Small value to avoid division by zero in some metrics.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of Dask workers.")
    parser.add_argument("--structures", type=str, default=None, help="Path to structures vector file (GDB, GPKG, shapefile).")
    parser.add_argument("--structures-metrics-parquet", type=str, default=None, help="Path to save structures metrics parquet file.")
    parser.add_argument("--structures-gpkg", type=str, default=None, help="Path to save per-structure results as GeoPackage (includes geometry + metrics).")
    parser.add_argument("--units", type=str, default="feet", choices=["meters", "feet"],
                        help="Units of the input rasters. Affects bucket thresholds (default: feet).")
    parser.add_argument("--catchments", type=str, default=None,
                        help="Path to NWM catchments GeoPackage for per-stream-order analysis.")
    parser.add_argument("--flows", type=str, default=None,
                        help="Path to NWM flows GeoPackage (must have ID and order_ columns).")
    parser.add_argument("--so-output-dir", type=str, default=None,
                        help="Directory to save per-stream-order metrics parquets.")

    args = parser.parse_args()

    dask.config.set(scheduler="threads", num_workers=args.num_workers)

    # Load subsampling points if provided
    subsampling_df = gpd.read_file(args.subsampling_file) if args.subsampling_file else None

    # If SO analysis requested, delay clearing candidate/benchmark from memory
    needs_so = args.catchments is not None and args.flows is not None
    clear = (not args.keep_memory) and (not needs_so)

    # Compare depth maps
    agreement_map, metric_table = compare_depth_maps(
        args.candidate_map,
        args.benchmark_map,
        agreement_map_path=args.agreement_map,
        metrics_parquet_path=args.metrics_parquet,
        chunk_size=args.chunk_size,
        subsampling_df=subsampling_df,
        subsampling_average=args.subsampling_average,
        metrics=args.metrics,
        nodata=args.nodata,
        encode_nodata=not args.no_encode_nodata,
        clear_memory=clear,
        epsilon=args.epsilon,
    )

    print("Agreement map and metrics table computed.")

    print("Agreement Map:")
    print(agreement_map)

    print("Metric Table:")
    print(metric_table.T)

    # Compare structures if provided
    if args.structures is not None:
        structures_gdf, structures_summary = compare_structures(
            agreement_map,
            args.structures,
            structures_metrics_parquet_path=args.structures_metrics_parquet,
            structures_gpkg_path=args.structures_gpkg,
            units=args.units,
        )

        print("\nStructures Metrics:")
        print(structures_summary.T)

    # Compare by stream order if catchments + flows provided
    if needs_so:
        # Re-open candidate/benchmark since compare_depth_maps may have closed them
        da_candidate = rxr.open_rasterio(args.candidate_map, masked=True).squeeze().compute()
        da_benchmark = rxr.open_rasterio(args.benchmark_map, masked=True).squeeze().compute()

        # Homogenize to match the agreement map grid
        da_candidate, da_benchmark = da_candidate.gval.homogenize(
            da_benchmark, target_map="benchmark", resampling=args.resampling,
        )

        # Apply same wet-domain masking as compare_depth_maps
        benchmark_wet = (da_benchmark > 0) & da_benchmark.notnull()
        candidate_wet = (da_candidate > 0) & da_candidate.notnull()
        either_wet = benchmark_wet | candidate_wet
        da_candidate = da_candidate.where(candidate_wet, other=0.0).where(either_wet)
        da_benchmark = da_benchmark.where(benchmark_wet, other=0.0).where(either_wet)

        so_output = args.so_output_dir
        if so_output is None and args.agreement_map is not None:
            so_output = os.path.join(os.path.dirname(args.agreement_map), "stream_order_metrics")

        so_metrics = compare_stream_orders(
            candidate_map=da_candidate,
            benchmark_map=da_benchmark,
            agreement_map=agreement_map,
            catchments_path=args.catchments,
            flows_path=args.flows,
            output_dir=so_output,
            metrics=args.metrics,
            epsilon=args.epsilon,
            units=args.units,
        )

        print("\nStream Order Metrics:")
        for so in sorted(so_metrics.keys()):
            print(f"\n  SO{so}:")
            print(so_metrics[so].T)

        del da_candidate, da_benchmark
        gc.collect()

if __name__ == "__main__":
    main()
