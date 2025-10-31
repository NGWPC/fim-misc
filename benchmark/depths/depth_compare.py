"""
Depth comparisons

Example usage:
python depth_compare.py \
    --candidate-map data/candidate_dummy_huc_12090301_depth_500yr.tif \
    --benchmark-map data/benchmark_ble_huc_12090301_depth_500yr.tif \
    --agreement-map data/agreement_dummy_ble_huc_12090301_depth_500yr.tif \
    --metrics-parquet data/depth_metrics_huc_12090301_depth_500yr.parquet \
    --num-workers 4
    --epsilon 0.1
"""
from __future__ import annotations
from typing import Iterable, Literal

import argparse
import os
import sys
import gc

import pyproj

# Set PROJ and GDAL data paths for odc-geo
os.environ["PROJ_LIB"] = os.path.join(sys.prefix, "share", "proj")
os.environ["GDAL_DATA"] = os.path.join(sys.prefix, "share", "gdal")

# Ensure pyproj uses the correct PROJ data directory
pyproj.datadir.set_data_dir(os.environ["PROJ_LIB"])

import rioxarray as rxr
import dask.array as da
import dask
from pyproj import CRS
import gval
import geopandas as gpd
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
        agreement_map.rio.to_raster(
            agreement_map_path,
            driver="COG",
            tiled=True,
            compress="LZW",
            blockxsize=da_candidate.chunks[1][0],
            blockysize=da_candidate.chunks[0][0],
            dtype="float32",
            BIGTIFF="IF_SAFER",
        )

    # Save metrics table if path provided
    if metrics_parquet_path is not None:
        metrics_table.to_parquet(metrics_parquet_path, engine="pyarrow", index=False, compression="snappy")

    return agreement_map, metrics_table

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

    args = parser.parse_args()

    dask.config.set(scheduler="threads", num_workers=args.num_workers)

    # Load subsampling points if provided
    subsampling_df = gpd.read_file(args.subsampling_file) if args.subsampling_file else None

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
        clear_memory=not args.keep_memory,
        epsilon=args.epsilon,
    )

    print("Agreement map and metrics table computed.")

    print("Agreement Map:")
    print(agreement_map)

    print("Metric Table:")
    print(metric_table.T)

if __name__ == "__main__":
    main()
