#!/usr/bin/env python3
"""
Compare candidate depth map against benchmark depth map.

Computes an agreement map (candidate - benchmark), continuous error
metrics via GVAL, per-structure zonal statistics, and per-stream-order
breakdowns.  Designed for fast execution on pre-processed inputs
(GeoParquet structures and pre-joined catchments).

Example usage:
python depth_compare.py \
    --candidate_path data/candidate_10m.tif \
    --benchmark_path data/benchmark_10m.tif \
    --agreement_map_path data/agreement.tif \
    --metrics_path data/depth_metrics.parquet \
    --structures_path data/structures.parquet \
    --structures_metrics_path data/structures_metrics.parquet \
    --structures_gpkg_path data/structures.gpkg \
    --catchments_path data/catchments_with_stream_order.parquet \
    --so_output_dir data/stream_order_metrics \
    --epsilon 0.1
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import logging
import json
from typing import Iterable, Literal

import pyproj

# Set PROJ and GDAL data paths (only if they exist)
_proj_lib = os.path.join(sys.prefix, "share", "proj")
_gdal_data = os.path.join(sys.prefix, "share", "gdal")
if os.path.isdir(_proj_lib):
    os.environ["PROJ_LIB"] = _proj_lib
    pyproj.datadir.set_data_dir(_proj_lib)
if os.path.isdir(_gdal_data):
    os.environ["GDAL_DATA"] = _gdal_data

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from gval.comparison.pairing_functions import difference
from exactextract import exact_extract

from gval_optimizations import compute_continuous_metrics_fast

JOB_ID = "depth_compare"


# ---------------------------------------------------------------------------
# Fast raster I/O
# ---------------------------------------------------------------------------

def load_raster_fast(path: str) -> xr.DataArray:
    """
    Load a raster with rasterio and wrap in an xarray DataArray with
    rioxarray metadata.  ~10x faster than rxr.open_rasterio(masked=True)
    for large files because it avoids xarray's coordinate inference overhead.
    """
    with rasterio.open(path) as src:
        data = src.read(1, masked=False).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        h, w = data.shape

    if nodata is not None:
        data[data == nodata] = np.nan

    xs = np.arange(w, dtype=np.float64) * transform.a + transform.c + transform.a / 2
    ys = np.arange(h, dtype=np.float64) * transform.e + transform.f + transform.e / 2

    da = xr.DataArray(data, dims=["y", "x"], coords={"y": ys, "x": xs})
    da.rio.write_crs(crs, inplace=True)
    da.rio.write_nodata(np.nan, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    return da


# ---------------------------------------------------------------------------
# Logging (JSON to stderr, following autoeval-jobs pattern)
# ---------------------------------------------------------------------------

def setup_logger(job_id: str) -> logging.Logger:
    """Configure structured JSON logger to stderr."""
    log = logging.getLogger(job_id)
    log.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    handler = logging.StreamHandler(sys.stderr)

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps({
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "job_id": job_id,
                "message": record.getMessage(),
            })

    handler.setFormatter(JsonFormatter())
    log.handlers = [handler]
    return log


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_depth_maps(
    candidate_map_path: str,
    benchmark_map_path: str,
    agreement_map_path: str | None = None,
    metrics_path: str | None = None,
    target_map: Literal["benchmark", "candidate"] = "benchmark",
    resampling: str = "bilinear",
    metrics: str | Iterable[str] = "all",
    nodata: float | int = -9999,
    encode_nodata: bool = True,
    epsilon: float = 0.1,
    log: logging.Logger | None = None,
):
    """
    Compare candidate vs benchmark depth maps using GVAL.

    Parameters
    ----------
    candidate_map_path : str
        Path to candidate depth map raster.
    benchmark_map_path : str
        Path to benchmark depth map raster.
    agreement_map_path : str, optional
        Path to save agreement map COG.
    metrics_path : str, optional
        Path to save metrics parquet.
    target_map : str, default "benchmark"
        Target map for GVAL homogenization.
    resampling : str, default "bilinear"
        Resampling method for homogenization.
    metrics : str or list, default "all"
        Metrics to compute.
    nodata : float, default -9999
        NoData value for the agreement map.
    encode_nodata : bool, default True
        Whether to encode NoData.
    epsilon : float, default 0.1
        Guard value for division-by-zero metrics.
    log : logging.Logger, optional
        Logger instance.

    Returns
    -------
    tuple
        (agreement_map, candidate_map, benchmark_map, metrics_table)
    """
    if log:
        log.info("Loading rasters")
    t0 = time.time()

    # Load rasters with rasterio directly — keep original nodata (-9999)
    # instead of converting to NaN (saves time on large rasters).
    with rasterio.open(candidate_map_path) as src:
        c_data = src.read(1, masked=False).astype(np.float32)
        c_nodata = src.nodata
        c_tf = src.transform
        c_crs = src.crs

    with rasterio.open(benchmark_map_path) as src:
        b_data = src.read(1, masked=False).astype(np.float32)
        b_nodata = src.nodata
        b_tf = src.transform
        b_crs = src.crs

    if log:
        log.info(f"Rasters loaded in {time.time() - t0:.2f}s")

    # Check if grids already match
    t1 = time.time()
    _grids_match = (
        c_crs == b_crs
        and c_data.shape == b_data.shape
        and c_tf == b_tf
    )

    if not _grids_match:
        # Grids differ — wrap in xarray and use GVAL homogenize
        c_h, c_w = c_data.shape
        c_xs = np.arange(c_w, dtype=np.float64) * c_tf.a + c_tf.c + c_tf.a / 2
        c_ys = np.arange(c_h, dtype=np.float64) * c_tf.e + c_tf.f + c_tf.e / 2

        b_h, b_w = b_data.shape
        b_xs = np.arange(b_w, dtype=np.float64) * b_tf.a + b_tf.c + b_tf.a / 2
        b_ys = np.arange(b_h, dtype=np.float64) * b_tf.e + b_tf.f + b_tf.e / 2

        da_c = xr.DataArray(c_data, dims=["y", "x"],
                            coords={"y": c_ys, "x": c_xs})
        da_c.rio.write_crs(c_crs, inplace=True)
        da_c.rio.write_nodata(c_nodata or np.nan, inplace=True)

        da_b = xr.DataArray(b_data, dims=["y", "x"],
                            coords={"y": b_ys, "x": b_xs})
        da_b.rio.write_crs(b_crs, inplace=True)
        da_b.rio.write_nodata(b_nodata or np.nan, inplace=True)

        da_c, da_b = da_c.gval.homogenize(da_b, target_map=target_map,
                                           resampling=resampling)
        c_data = da_c.values
        b_data = da_b.values
        b_tf = da_b.rio.transform()
        b_crs = da_b.rio.crs
        del da_c, da_b

    if log:
        log.info(f"Homogenized in {time.time() - t1:.2f}s")

    # Preserve CRS
    crs = b_crs.to_wkt()

    # Wet-domain masking using original nodata values (NOT np.isfinite
    # which treats -9999 as valid — that was a previous bug).
    t2 = time.time()
    c_nd = c_nodata if c_nodata is not None else np.nan
    b_nd = b_nodata if b_nodata is not None else np.nan

    if np.isnan(c_nd):
        c_valid = np.isfinite(c_data) & (c_data > 0)
    else:
        c_valid = (c_data != c_nd) & (c_data > 0)

    if np.isnan(b_nd):
        b_valid = np.isfinite(b_data) & (b_data > 0)
    else:
        b_valid = (b_data != b_nd) & (b_data > 0)

    either_wet = c_valid | b_valid

    # Fill dry-in-domain cells with 0; out-of-domain with nodata
    nd_val = nodata if nodata is not None else -9999.0
    c_masked = np.where(either_wet, np.where(c_valid, c_data, 0.0), nd_val).astype(np.float32)
    b_masked = np.where(either_wet, np.where(b_valid, b_data, 0.0), nd_val).astype(np.float32)

    # Compute agreement map as numpy: candidate - benchmark
    agr_data = np.where(either_wet, c_masked - b_masked, nd_val).astype(np.float32)

    # Wrap in xarray for downstream compatibility (structures, stream order)
    h, w = b_data.shape
    xs = np.arange(w, dtype=np.float64) * b_tf.a + b_tf.c + b_tf.a / 2
    ys = np.arange(h, dtype=np.float64) * b_tf.e + b_tf.f + b_tf.e / 2

    agreement_map = xr.DataArray(agr_data, dims=["y", "x"],
                                 coords={"y": ys, "x": xs})
    agreement_map.rio.write_crs(b_crs, inplace=True)
    agreement_map.rio.write_nodata(nd_val, inplace=True)
    agreement_map.rio.write_transform(b_tf, inplace=True)

    da_candidate = xr.DataArray(c_masked, dims=["y", "x"],
                                coords={"y": ys, "x": xs})
    da_candidate.rio.write_crs(b_crs, inplace=True)
    da_candidate.rio.write_nodata(nd_val, inplace=True)

    da_benchmark = xr.DataArray(b_masked, dims=["y", "x"],
                                coords={"y": ys, "x": xs})
    da_benchmark.rio.write_crs(b_crs, inplace=True)
    da_benchmark.rio.write_nodata(nd_val, inplace=True)

    if log:
        log.info(f"Agreement map computed in {time.time() - t2:.2f}s")

    # Compute metrics using optimized single-pass implementation
    t3 = time.time()
    metrics_table = compute_continuous_metrics_fast(
        agreement_map=agreement_map,
        candidate_map=da_candidate,
        benchmark_map=da_benchmark,
        metrics=metrics,
        epsilon=epsilon,
    )
    if log:
        log.info(f"Metrics computed in {time.time() - t3:.2f}s")

    # Save agreement map using rasterio directly for speed
    if agreement_map_path is not None:
        t4 = time.time()
        from rasterio.crs import CRS as RioCRS

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": w,
            "height": h,
            "count": 1,
            "crs": RioCRS.from_wkt(crs),
            "transform": b_tf,
            "nodata": nd_val,
            "compress": "zstd",
            "zstd_level": 1,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

        # agr_data already has nodata encoded as nd_val (-9999)
        with rasterio.open(agreement_map_path, "w", **profile) as dst:
            dst.write(agr_data, 1)

        if log:
            log.info(f"Agreement map saved in {time.time() - t4:.2f}s")

    # Save metrics
    if metrics_path is not None:
        metrics_table.to_parquet(
            metrics_path, engine="pyarrow", index=False, compression="snappy"
        )

    return agreement_map, da_candidate, da_benchmark, metrics_table


# ---------------------------------------------------------------------------
# Structures comparison
# ---------------------------------------------------------------------------

def compare_structures(
    agreement_map: xr.DataArray,
    structures_path: str,
    structures_metrics_path: str | None = None,
    structures_gpkg_path: str | None = None,
    units: str = "feet",
    agreement_map_path: str | None = None,
    log: logging.Logger | None = None,
):
    """
    Summarize agreement map at building footprint locations.

    Parameters
    ----------
    agreement_map : xr.DataArray
        Agreement map (candidate - benchmark).
    structures_path : str
        Path to structures file (GeoParquet, GDB, GPKG, shapefile).
    structures_metrics_path : str, optional
        Path to save summary metrics parquet.
    structures_gpkg_path : str, optional
        Path to save per-structure results as GeoPackage.
    units : str, default "feet"
        Input units. Affects bucket thresholds.
    log : logging.Logger, optional
        Logger instance.

    Returns
    -------
    tuple
        (per_structure_gdf, summary_df)
    """
    t0 = time.time()
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    if hasattr(agreement_map, "compute"):
        agreement_map = agreement_map.compute()

    agr_nodata = agreement_map.rio.nodata
    if agr_nodata is None:
        agr_nodata = -9999.0
    raster_crs = agreement_map.rio.crs

    # --- Polygonize valid domain, simplify, buffer ---
    t_domain = time.time()
    agr_data = agreement_map.values
    agr_tf = agreement_map.rio.transform()
    px = abs(agr_tf.a)

    if np.isnan(agr_nodata):
        valid_mask = np.isfinite(agr_data).astype(np.uint8)
    else:
        valid_mask = ((agr_data != agr_nodata) & np.isfinite(agr_data)).astype(np.uint8)

    domain_polys = [
        shape(geom) for geom, val in shapes(valid_mask, mask=valid_mask == 1, transform=agr_tf)
        if val == 1
    ]
    if not domain_polys:
        if log:
            log.info("No valid cells in agreement map")
        return gpd.GeoDataFrame(), pd.DataFrame()

    # Simplify and buffer each domain polygon individually
    domain_buffered = [p.simplify(px * 2).buffer(px * 3) for p in domain_polys]
    if log:
        log.info(f"Flood domain polygon built in {time.time() - t_domain:.2f}s "
                 f"({len(domain_polys)} polygons)")

    # --- Load only structures intersecting the buffered domain ---
    t1 = time.time()
    if structures_path.endswith(".parquet"):
        all_structures = gpd.read_parquet(structures_path)
        if all_structures.crs != raster_crs:
            all_structures = all_structures.to_crs(raster_crs)
        # Query each domain polygon individually (fast with sindex)
        hit_set = set()
        for poly in domain_buffered:
            hits = all_structures.sindex.query(poly, predicate="intersects")
            hit_set.update(hits)
        domain_structures = all_structures.iloc[sorted(hit_set)].copy()
        del all_structures
    else:
        from shapely.ops import unary_union as _unary_union
        domain_simple = _unary_union(domain_buffered)
        domain_mask = gpd.GeoDataFrame(geometry=[domain_simple], crs=raster_crs)
        domain_structures = gpd.read_file(structures_path, mask=domain_mask)
        if domain_structures.crs != raster_crs:
            domain_structures = domain_structures.to_crs(raster_crs)
    if len(domain_structures) == 0:
        if log:
            log.info("No structures in flood domain")
        return gpd.GeoDataFrame(), pd.DataFrame()
    if log:
        log.info(f"Loaded {len(domain_structures)} structures in {time.time() - t1:.2f}s")

    # --- Zonal stats (use file-backed raster for proper nodata handling) ---
    t2 = time.time()
    _raster_ds = None
    if agreement_map_path and os.path.exists(agreement_map_path):
        _raster_ds = rasterio.open(agreement_map_path)
        raster_source = _raster_ds
    else:
        raster_source = agreement_map

    stats = exact_extract(
        raster_source, domain_structures, ["mean", "min", "max", "count"],
        include_cols=[], output="pandas",
    )
    if _raster_ds:
        _raster_ds.close()
    domain_structures = domain_structures.copy()
    domain_structures["mean_depth_diff"] = stats["mean"].fillna(0.0)
    domain_structures["min_depth_diff"] = stats["min"].fillna(0.0)
    domain_structures["max_depth_diff"] = stats["max"].fillna(0.0)
    domain_structures["pixel_count"] = stats["count"]
    domain_structures = domain_structures[domain_structures["pixel_count"] > 0].copy()

    if log:
        log.info(f"Zonal stats in {time.time() - t2:.2f}s, {len(domain_structures)} with coverage")

    diff = domain_structures["mean_depth_diff"].values
    abs_diff = np.abs(diff)
    n = len(domain_structures)

    # Per-structure categoricals
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

    # Summary metrics
    mae_d = float(np.mean(abs_diff))
    mse_d = float(np.mean(diff ** 2))
    rmse_d = float(np.sqrt(mse_d))
    mean_signed_d = float(np.mean(diff))
    median_ae = float(np.median(abs_diff))
    p90_ae = float(np.percentile(abs_diff, 90))
    max_ae = float(np.max(abs_diff))

    n_within_1ft = int(np.sum(abs_diff < 1 * ft_scale))
    n_within_3ft = int(np.sum(abs_diff < 3 * ft_scale))
    n_within_5ft = int(np.sum(abs_diff < 5 * ft_scale))
    n_gt_5ft = int(np.sum(abs_diff >= 5 * ft_scale))
    n_over = int(np.sum(diff > ft_scale * 0.1))
    n_under = int(np.sum(diff < -ft_scale * 0.1))
    n_match = n - n_over - n_under

    summary = pd.DataFrame({
        "structures_in_domain": [n],
        "mean_absolute_error": [mae_d],
        "root_mean_squared_error": [rmse_d],
        "mean_squared_error": [mse_d],
        "mean_signed_error": [mean_signed_d],
        "median_absolute_error": [median_ae],
        "p90_absolute_error": [p90_ae],
        "max_absolute_error": [max_ae],
        "n_within_1ft": [n_within_1ft],
        "pct_within_1ft": [n_within_1ft / n * 100],
        "n_within_3ft": [n_within_3ft],
        "pct_within_3ft": [n_within_3ft / n * 100],
        "n_within_5ft": [n_within_5ft],
        "pct_within_5ft": [n_within_5ft / n * 100],
        "n_gt_5ft": [n_gt_5ft],
        "pct_gt_5ft": [n_gt_5ft / n * 100],
        "n_over_predict": [n_over],
        "pct_over_predict": [n_over / n * 100],
        "n_under_predict": [n_under],
        "pct_under_predict": [n_under / n * 100],
        "n_match": [n_match],
        "pct_match": [n_match / n * 100],
    })

    if structures_metrics_path is not None:
        summary.to_parquet(
            structures_metrics_path, engine="pyarrow", index=False, compression="snappy"
        )

    if structures_gpkg_path is not None:
        # Save in WGS84 for reporting compatibility
        out_gdf = domain_structures.to_crs("EPSG:4326") if domain_structures.crs != "EPSG:4326" else domain_structures
        out_gdf.to_file(structures_gpkg_path, driver="GPKG", layer="structures")

    if log:
        log.info(f"Structures comparison done in {time.time() - t0:.2f}s")

    return domain_structures, summary


# ---------------------------------------------------------------------------
# Stream-order comparison
# ---------------------------------------------------------------------------

def compare_stream_orders(
    candidate_map: xr.DataArray,
    benchmark_map: xr.DataArray,
    agreement_map: xr.DataArray,
    catchments_path: str,
    output_dir: str | None = None,
    metrics: str | Iterable[str] = "all",
    epsilon: float = 0.1,
    units: str = "feet",
    log: logging.Logger | None = None,
):
    """
    Compute depth metrics per stream order.

    Parameters
    ----------
    candidate_map : xr.DataArray
        Candidate depth map (homogenized, domain-clipped).
    benchmark_map : xr.DataArray
        Benchmark depth map (homogenized, domain-clipped).
    agreement_map : xr.DataArray
        Agreement map (candidate - benchmark).
    catchments_path : str
        Path to pre-joined catchments with stream order (GeoParquet or GPKG).
    output_dir : str, optional
        Directory to save per-SO metrics parquets.
    metrics : str or list, default "all"
        Metrics to compute.
    epsilon : float, default 0.1
        Epsilon for metrics.
    units : str, default "feet"
        Input units.
    log : logging.Logger, optional
        Logger instance.

    Returns
    -------
    dict
        {stream_order: pd.DataFrame} of metrics per stream order.
    """
    from shapely.geometry import box
    from rasterio.features import rasterize

    t0 = time.time()

    if hasattr(agreement_map, "compute"):
        agreement_map = agreement_map.compute()
    if hasattr(candidate_map, "compute"):
        candidate_map = candidate_map.compute()
    if hasattr(benchmark_map, "compute"):
        benchmark_map = benchmark_map.compute()

    raster_crs = agreement_map.rio.crs
    bounds = agreement_map.rio.bounds()
    domain_box = box(bounds[0], bounds[1], bounds[2], bounds[3])

    t1 = time.time()
    if catchments_path.endswith(".parquet"):
        catchments = gpd.read_parquet(catchments_path)
    else:
        domain_mask = gpd.GeoDataFrame(geometry=[domain_box], crs=raster_crs)
        catchments = gpd.read_file(catchments_path, mask=domain_mask)

    if log:
        log.info(f"Loaded {len(catchments)} catchments in {time.time() - t1:.2f}s")

    if len(catchments) == 0:
        return {}

    if "order_" not in catchments.columns:
        if log:
            log.info("No order_ column; skipping stream order analysis")
        return {}

    catchments = catchments.dropna(subset=["order_"])
    catchments["order_"] = catchments["order_"].astype(int)

    if catchments.crs != raster_crs:
        catchments = catchments.to_crs(raster_crs)

    catchments = catchments[catchments.geometry.intersects(domain_box)].copy()

    if log:
        log.info(f"Stream orders: {sorted(catchments['order_'].unique())}")

    # Rasterize stream-order labels onto the agreement grid in one pass.
    # This replaces N per-SO clip operations (3 rasters * N SOs) with a
    # single rasterize + numpy groupby.
    t_rast = time.time()
    agr_tf = agreement_map.rio.transform()
    agr_shape = agreement_map.values.shape

    so_raster = rasterize(
        [(geom, so) for geom, so in zip(catchments.geometry, catchments["order_"])],
        out_shape=agr_shape,
        transform=agr_tf,
        fill=0,
        dtype=np.int16,
        all_touched=True,
    )
    if log:
        log.info(f"Rasterized stream orders in {time.time() - t_rast:.2f}s")

    # Get numpy arrays
    e_arr = agreement_map.values.ravel()
    c_arr = candidate_map.values.ravel()
    b_arr = benchmark_map.values.ravel()
    so_arr = so_raster.ravel()

    # Build area lookup
    area_by_so = catchments.groupby("order_")["AreaSqKM"].sum().to_dict()
    count_by_so = catchments.groupby("order_").size().to_dict()

    so_metrics = {}
    ft_scale = 0.3048 if units == "meters" else 1.0

    for so in sorted(catchments["order_"].unique()):
        so = int(so)
        # Mask out nodata — handles both NaN and sentinel values like -9999
        agr_nd = agreement_map.rio.nodata if hasattr(agreement_map, "rio") else None
        _finite = np.isfinite(e_arr) & np.isfinite(c_arr) & np.isfinite(b_arr)
        if agr_nd is not None and not np.isnan(agr_nd):
            _finite &= (e_arr != agr_nd) & (c_arr != agr_nd) & (b_arr != agr_nd)
        mask = (so_arr == so) & _finite
        e_so = e_arr[mask]
        c_so = c_arr[mask]
        b_so = b_arr[mask]

        if len(e_so) < 2:
            continue

        metrics_table = compute_continuous_metrics_fast(
            agreement_map=e_so,
            candidate_map=c_so,
            benchmark_map=b_so,
            metrics=metrics,
            epsilon=epsilon,
        )

        metrics_table["stream_order"] = so
        metrics_table["n_catchments"] = count_by_so.get(so, 0)
        metrics_table["area_sqkm"] = area_by_so.get(so, 0.0)
        metrics_table["n_valid_pixels"] = len(e_so)

        abs_diff = np.abs(e_so)
        n_over = int(np.sum(e_so > ft_scale * 0.1))
        n_under = int(np.sum(e_so < -ft_scale * 0.1))
        n_match = len(e_so) - n_over - n_under
        metrics_table["n_over_predict"] = n_over
        metrics_table["n_under_predict"] = n_under
        metrics_table["n_match"] = n_match

        so_metrics[so] = metrics_table

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

    if log:
        log.info(f"Stream order analysis done in {time.time() - t0:.2f}s")

    return so_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare candidate depth map against benchmark.")
    parser.add_argument("--candidate_path", type=str, required=True,
                        help="Path to candidate depth map raster.")
    parser.add_argument("--benchmark_path", type=str, required=True,
                        help="Path to benchmark depth map raster.")
    parser.add_argument("--agreement_map_path", type=str, required=False, default=None,
                        help="Path to save agreement map COG.")
    parser.add_argument("--metrics_path", type=str, required=False, default=None,
                        help="Path to save depth metrics parquet.")
    parser.add_argument("--resampling", type=str, default="bilinear",
                        help="Resampling method for homogenization.")
    parser.add_argument("--target_map", type=str, default="benchmark",
                        choices=["benchmark", "candidate"],
                        help="Target map for homogenization.")
    parser.add_argument("--metrics", type=str, default="all",
                        help="Metrics to compute, or 'all'.")
    parser.add_argument("--nodata", type=float, default=-9999,
                        help="NoData value for agreement map.")
    parser.add_argument("--no_encode_nodata", action="store_true",
                        help="Disable NoData encoding in agreement map.")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Guard value for division-by-zero metrics.")
    parser.add_argument("--structures_path", type=str, required=False, default=None,
                        help="Path to structures (GeoParquet, GDB, GPKG, shapefile).")
    parser.add_argument("--structures_metrics_path", type=str, required=False, default=None,
                        help="Path to save structures summary metrics parquet.")
    parser.add_argument("--structures_gpkg_path", type=str, required=False, default=None,
                        help="Path to save per-structure results GeoPackage.")
    parser.add_argument("--catchments_path", type=str, required=False, default=None,
                        help="Path to catchments with stream order (GeoParquet or GPKG).")
    parser.add_argument("--so_output_dir", type=str, required=False, default=None,
                        help="Directory to save per-stream-order metrics.")
    parser.add_argument("--units", type=str, default="feet", choices=["meters", "feet"],
                        help="Units of input rasters.")

    args = parser.parse_args()

    log = setup_logger(JOB_ID)
    total_t0 = time.time()

    try:
        # Compare depth maps
        agreement_map, da_candidate, da_benchmark, metrics_table = compare_depth_maps(
            candidate_map_path=args.candidate_path,
            benchmark_map_path=args.benchmark_path,
            agreement_map_path=args.agreement_map_path,
            metrics_path=args.metrics_path,
            target_map=args.target_map,
            resampling=args.resampling,
            metrics=args.metrics,
            nodata=args.nodata,
            encode_nodata=not args.no_encode_nodata,
            epsilon=args.epsilon,
            log=log,
        )

        log.info("Depth comparison complete")
        print("Metric Table:")
        print(metrics_table.T)

        # Structures
        if args.structures_path is not None:
            structures_gdf, structures_summary = compare_structures(
                agreement_map=agreement_map,
                structures_path=args.structures_path,
                structures_metrics_path=args.structures_metrics_path,
                structures_gpkg_path=args.structures_gpkg_path,
                units=args.units,
                agreement_map_path=args.agreement_map_path,
                log=log,
            )
            print("\nStructures Metrics:")
            print(structures_summary.T)

        # Stream orders
        if args.catchments_path is not None:
            so_output = args.so_output_dir
            if so_output is None and args.agreement_map_path is not None:
                so_output = os.path.join(
                    os.path.dirname(args.agreement_map_path), "stream_order_metrics"
                )

            so_metrics = compare_stream_orders(
                candidate_map=da_candidate,
                benchmark_map=da_benchmark,
                agreement_map=agreement_map,
                catchments_path=args.catchments_path,
                output_dir=so_output,
                metrics=args.metrics,
                epsilon=args.epsilon,
                units=args.units,
                log=log,
            )

            print("\nStream Order Metrics:")
            for so in sorted(so_metrics.keys()):
                print(f"\n  SO{so}:")
                print(so_metrics[so].T)

        total_time = time.time() - total_t0
        log.info(json.dumps({
            "output_path": args.agreement_map_path,
            "metrics_path": args.metrics_path,
            "total_time": f"{total_time:.2f}s",
        }))
        print(f"\nTotal time: {total_time:.2f}s")

    except Exception as e:
        log.error(f"{JOB_ID} run failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
