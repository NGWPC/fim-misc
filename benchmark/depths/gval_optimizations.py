#!/usr/bin/env python3
"""
Optimized GVAL continuous metrics computation.

Replaces GVAL's default per-metric array iteration with a single-pass
numpy implementation that computes all 12 continuous metrics from shared
intermediate values. Produces the same output DataFrame schema as
gval.comparison.compute_continuous_metrics._compute_continuous_metrics.
"""
from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import pandas as pd
import xarray as xr


def compute_continuous_metrics_fast(
    agreement_map: Union[xr.DataArray, xr.Dataset],
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
    metrics: Union[str, Iterable[str]] = "all",
    epsilon: float = 1e-10,
) -> pd.DataFrame:
    """
    Single-pass continuous metrics matching GVAL's output schema.

    Instead of calling each metric function independently (12+ full-array
    passes), this extracts numpy arrays once and derives all metrics from
    shared intermediates in 2-3 passes.

    Parameters
    ----------
    agreement_map : xr.DataArray or xr.Dataset
        Error map (candidate - benchmark).
    candidate_map : xr.DataArray or xr.Dataset
        Candidate depth map.
    benchmark_map : xr.DataArray or xr.Dataset
        Benchmark depth map.
    metrics : str or Iterable[str], default = "all"
        Which metrics to include. "all" returns all 12 GVAL metrics.
    epsilon : float, default = 1e-10
        Guard against division by zero.

    Returns
    -------
    pd.DataFrame
        Metrics table with "band" column and one column per metric,
        identical schema to GVAL's _compute_continuous_metrics output.
    """
    # Materialize to numpy once (supports xr.DataArray, xr.Dataset, or raw numpy)
    def _to_flat(arr):
        if hasattr(arr, "compute"):
            arr = arr.compute()
        if hasattr(arr, "values"):
            return arr.values.ravel()
        return np.asarray(arr).ravel()

    e = _to_flat(agreement_map)
    c = _to_flat(candidate_map)
    b = _to_flat(benchmark_map)

    # Mask nodata from any source. Handles both NaN and sentinel values
    # like -9999. Check for nodata attribute on xarray inputs.
    nd_val = None
    for arr in (agreement_map, candidate_map, benchmark_map):
        if hasattr(arr, "rio") and arr.rio.nodata is not None:
            nd_val = arr.rio.nodata
            break

    valid = np.isfinite(e) & np.isfinite(c) & np.isfinite(b)
    if nd_val is not None and not np.isnan(nd_val):
        valid &= (e != nd_val) & (c != nd_val) & (b != nd_val)
    e = e[valid]
    c = c[valid]
    b = b[valid]

    n = len(e)
    if n == 0:
        return pd.DataFrame({"band": ["1"]})

    # --- Single-pass intermediates ---
    abs_e = np.abs(e)
    e_sq = e * e

    sum_abs_e = abs_e.sum()
    sum_e_sq = e_sq.sum()
    sum_e = e.sum()

    b_mean = b.mean()
    b_min = b.min()
    b_max = b.max()
    b_range = b_max - b_min
    b_ss = ((b - b_mean) ** 2).sum()  # total sum of squares

    # For MAPE: guard against zero benchmark values
    b_safe = np.where(np.abs(b) < epsilon, epsilon, b)

    # --- Compute all 12 metrics ---
    mae = float(sum_abs_e / n)
    mse = float(sum_e_sq / n)
    rmse = float(np.sqrt(mse))
    mean_signed = float(sum_e / n)
    mpe = float((e / b_mean).mean())
    mape = float(np.abs(e / b_safe).mean())
    mnrmse = float(rmse / b_mean) if b_mean != 0 else float("nan")
    rnrmse = float(rmse / b_range) if b_range != 0 else float("nan")
    mnmae = float(mae / b_mean) if b_mean != 0 else float("nan")
    rnmae = float(mae / b_range) if b_range != 0 else float("nan")
    r2 = float(1.0 - sum_e_sq / b_ss) if b_ss != 0 else float("nan")
    smape = float(2.0 * sum_abs_e / (np.abs(c) + np.abs(b)).sum())

    all_metrics = {
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "mean_signed_error": mean_signed,
        "mean_percentage_error": mpe,
        "mean_absolute_percentage_error": mape,
        "mean_normalized_root_mean_squared_error": mnrmse,
        "range_normalized_root_mean_squared_error": rnrmse,
        "mean_normalized_mean_absolute_error": mnmae,
        "range_normalized_mean_absolute_error": rnmae,
        "coefficient_of_determination": r2,
        "symmetric_mean_absolute_percentage_error": smape,
    }

    # Filter to requested metrics
    if metrics != "all":
        if isinstance(metrics, str):
            metrics = [metrics]
        all_metrics = {k: v for k, v in all_metrics.items() if k in metrics}

    row = {"band": "1", **all_metrics}
    return pd.DataFrame([row])
