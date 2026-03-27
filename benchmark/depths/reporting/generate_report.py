"""
Generate a self-contained HTML dashboard from depth comparison outputs.

Example usage:
python generate_report.py \
    --depth-metrics data/depth_metrics.parquet \
    --structures-metrics data/structures_metrics.parquet \
    --structures-gpkg data/structures.gpkg \
    --agreement-map data/agreement.tif \
    --output report.html \
    --title "HUC 12090301 - 500yr Depth Comparison"
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import contextily as cx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rioxarray as rxr

M_TO_FT = 3.28084

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1a1d24",
    font=dict(family="Proxima Nova, Inter, sans-serif", color="#e0e0e0"),
    title_font=dict(family="Proxima Nova, Inter, sans-serif", color="#f0f0f0"),
    xaxis=dict(gridcolor="#2a2d35", zerolinecolor="#3a3d45"),
    yaxis=dict(gridcolor="#2a2d35", zerolinecolor="#3a3d45"),
)

# Raster metric columns that represent depth values (meters -> feet)
_DEPTH_METRIC_COLS = {
    "mean_absolute_error",
    "root_mean_squared_error",
    "mean_squared_error",
    "mean_signed_error",
    "median_absolute_error",
    "p90_absolute_error",
    "max_absolute_error",
}


def load_data(depth_metrics_path, structures_metrics_path=None,
              structures_gpkg_path=None, agreement_map_path=None,
              catchments_path=None, flows_path=None,
              so_metrics_dir=None,
              units="meters"):
    """Load all input data files, optionally converting depth values from meters to feet."""
    data = {}
    convert = units == "meters"
    scale = M_TO_FT if convert else 1.0
    scale_sq = M_TO_FT ** 2 if convert else 1.0

    dm = pd.read_parquet(depth_metrics_path)
    if convert:
        for col in dm.columns:
            if col in _DEPTH_METRIC_COLS:
                if "squared" in col:
                    dm[col] = dm[col] * scale_sq
                else:
                    dm[col] = dm[col] * scale
    dm = dm.drop(columns=["band"], errors="ignore")
    data["depth_metrics"] = dm

    if structures_metrics_path and os.path.exists(structures_metrics_path):
        sm = pd.read_parquet(structures_metrics_path)
        if convert:
            for col in sm.columns:
                if col in _DEPTH_METRIC_COLS:
                    if "squared" in col:
                        sm[col] = sm[col] * scale_sq
                    else:
                        sm[col] = sm[col] * scale
        data["structures_metrics"] = sm

    if structures_gpkg_path and os.path.exists(structures_gpkg_path):
        import geopandas as gpd
        gdf = gpd.read_file(structures_gpkg_path)
        if convert:
            for col in ["mean_depth_diff", "min_depth_diff", "max_depth_diff"]:
                if col in gdf.columns:
                    gdf[col] = gdf[col] * scale
        data["structures_gdf"] = gdf
        # Compute total structure footprint area in sq km
        if gdf.crs and gdf.crs.is_geographic:
            data["structures_area_sqkm"] = gdf.to_crs("EPSG:5070").geometry.area.sum() / 1e6
        else:
            data["structures_area_sqkm"] = gdf.geometry.area.sum() / 1e6

    if agreement_map_path and os.path.exists(agreement_map_path):
        da = rxr.open_rasterio(agreement_map_path, masked=True).squeeze().compute()
        if convert:
            da = da * scale
        data["agreement_da"] = da
        data["agreement_values"] = da.values[~np.isnan(da.values)].flatten()
        # Compute total valid raster area in sq km
        res_x = abs(float(da.x[1] - da.x[0]))
        res_y = abs(float(da.y[1] - da.y[0]))
        cell_area_m2 = res_x * res_y
        n_valid = int(np.sum(~np.isnan(da.values)))
        data["raster_area_sqkm"] = n_valid * cell_area_m2 / 1e6

    # Stream-order catchment analysis
    if (catchments_path and flows_path and agreement_map_path
            and os.path.exists(catchments_path) and os.path.exists(flows_path)
            and "agreement_da" in data):
        print("Loading catchments and computing stream order metrics...")
        import geopandas as gpd
        from exactextract import exact_extract
        from shapely.geometry import box
        import rasterio

        da = data["agreement_da"]
        # Build spatial mask from agreement raster bounds
        bounds = da.rio.bounds()  # (left, bottom, right, top)
        domain_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
        domain_mask = gpd.GeoDataFrame(geometry=[domain_box], crs=da.rio.crs)

        # Load only catchments intersecting the domain
        print("  Reading catchments within domain...")
        catchments = gpd.read_file(catchments_path, mask=domain_mask)
        print(f"  Found {len(catchments)} catchments in domain")

        if len(catchments) > 0:
            # Load flows (just ID + order_) and join
            print("  Reading flow attributes for stream order...")
            flows = gpd.read_file(flows_path, columns=["ID", "order_"])
            # Drop geometry from flows for a table join
            flows_df = pd.DataFrame({"ID": flows["ID"], "order_": flows["order_"]})
            catchments = catchments.merge(flows_df, on="ID", how="left")
            catchments = catchments.dropna(subset=["order_"])
            catchments["order_"] = catchments["order_"].astype(int)
            print(f"  Catchments with stream order: {len(catchments)}")
            print(f"  Stream orders found: {sorted(catchments['order_'].unique())}")

            if len(catchments) > 0:
                # Write agreement raster to a temp file for exactextract
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                    tmp_path = tmp.name
                da.rio.to_raster(tmp_path)

                # Compute mean depth diff per catchment using exactextract
                print("  Running zonal statistics per catchment...")
                results = exact_extract(
                    rasterio.open(tmp_path),
                    catchments,
                    ["mean", "count"],
                    output="pandas",
                )
                catchments["mean_diff"] = results["mean"]
                catchments["pixel_count"] = results["count"]

                os.unlink(tmp_path)

                # Remove catchments with no valid pixels
                catchments = catchments[catchments["pixel_count"] > 0].copy()

                # Classify each catchment as under/match/over
                # Using the same structure-level threshold: 0.1 ft
                threshold = 0.1
                catchments["bias"] = "match"
                catchments.loc[catchments["mean_diff"] < -threshold, "bias"] = "under"
                catchments.loc[catchments["mean_diff"] > threshold, "bias"] = "over"

                # Aggregate by stream order
                so_metrics = {}
                for so, grp in catchments.groupby("order_"):
                    n_total = len(grp)
                    n_under = int((grp["bias"] == "under").sum())
                    n_match = int((grp["bias"] == "match").sum())
                    n_over = int((grp["bias"] == "over").sum())
                    area_sqkm = float(grp["AreaSqKM"].sum())
                    so_metrics[int(so)] = {
                        "n_under": n_under,
                        "n_match": n_match,
                        "n_over": n_over,
                        "n_total": n_total,
                        "area_sqkm": area_sqkm,
                    }
                    print(f"  SO{so}: {n_total} catchments (under={n_under}, match={n_match}, over={n_over})")

                data["stream_order_metrics"] = so_metrics

                # Save per-catchment distributions and geometries keyed by stream order
                so_distributions = {}
                so_geometries = {}
                for so, grp in catchments.groupby("order_"):
                    so_distributions[int(so)] = grp["mean_diff"].values
                    so_geometries[int(so)] = grp.geometry
                data["stream_order_distributions"] = so_distributions
                data["stream_order_geometries"] = so_geometries

    # Load per-SO GVAL metrics from parquet files (produced by depth_compare.py)
    if so_metrics_dir and os.path.isdir(so_metrics_dir):
        print(f"Loading per-SO GVAL metrics from {so_metrics_dir}...")
        combined_path = os.path.join(so_metrics_dir, "depth_metrics_by_stream_order.parquet")
        if os.path.exists(combined_path):
            so_df = pd.read_parquet(combined_path)
            if convert:
                for col in so_df.columns:
                    if col in _DEPTH_METRIC_COLS:
                        if "squared" in col:
                            so_df[col] = so_df[col] * scale_sq
                        else:
                            so_df[col] = so_df[col] * scale
            so_raster_metrics = {}
            for _, row in so_df.iterrows():
                so = int(row["stream_order"])
                so_raster_metrics[so] = row.drop(["band", "stream_order"], errors="ignore").to_dict()
            data["stream_order_raster_metrics"] = so_raster_metrics
            print(f"  Loaded GVAL metrics for stream orders: {sorted(so_raster_metrics.keys())}")

            # Also populate bias metrics and distributions if not already loaded from catchments
            if "stream_order_metrics" not in data:
                so_bias = {}
                for so, metrics in so_raster_metrics.items():
                    so_bias[so] = {
                        "n_under": int(metrics.get("n_under_predict", 0)),
                        "n_match": int(metrics.get("n_match", 0)),
                        "n_over": int(metrics.get("n_over_predict", 0)),
                        "n_total": int(metrics.get("n_under_predict", 0) + metrics.get("n_match", 0) + metrics.get("n_over_predict", 0)),
                        "area_sqkm": metrics.get("area_sqkm"),
                    }
                data["stream_order_metrics"] = so_bias

    return data


def render_agreement_map(da):
    """Render the agreement map over a dark basemap with a locator inset."""
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import pyproj

    FONT = "Proxima Nova"
    plt.rcParams["font.family"] = ["Proxima Nova", "Helvetica Neue", "Helvetica", "Arial", "sans-serif"]

    vals = da.values
    valid = vals[~np.isnan(vals)]
    vmax = np.percentile(np.abs(valid), 95)

    # Reproject to Web Mercator for basemap tiles
    da_3857 = da.rio.reproject("EPSG:3857")
    vals_3857 = da_3857.values

    fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=150,
                           facecolor="none")

    extent_3857 = [
        float(da_3857.x.min()), float(da_3857.x.max()),
        float(da_3857.y.min()), float(da_3857.y.max()),
    ]

    # Set axis extent first so basemap tiles load for the right area
    ax.set_xlim(extent_3857[0], extent_3857[1])
    ax.set_ylim(extent_3857[2], extent_3857[3])

    # Add dark basemap FIRST (underneath)
    cx.add_basemap(ax, source=cx.providers.CartoDB.DarkMatter, crs="EPSG:3857", attribution="")

    # Then overlay the agreement map on top
    # RdBu: red = negative (under-predict), blue = positive (over-predict)
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(alpha=0)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(
        np.ma.masked_invalid(vals_3857), cmap=cmap, norm=norm,
        extent=extent_3857, origin="upper", interpolation="nearest", alpha=0.7,
        zorder=2,
    )

    ax.set_facecolor("none")

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- Locator inset map ---
    # Convert extent to lon/lat for the inset
    transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(extent_3857[0], extent_3857[2])
    lon_max, lat_max = transformer.transform(extent_3857[1], extent_3857[3])

    # Compute center and a wide view around it
    center_lon = (lon_min + lon_max) / 2
    center_lat = (lat_min + lat_max) / 2
    span_lon = (lon_max - lon_min)
    span_lat = (lat_max - lat_min)
    pad = max(span_lon, span_lat) * 8  # zoom out for regional context

    inset_lon_min = center_lon - pad
    inset_lon_max = center_lon + pad
    inset_lat_min = center_lat - pad
    inset_lat_max = center_lat + pad

    # Convert inset bounds back to 3857
    ix_min, iy_min = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(inset_lon_min, inset_lat_min)
    ix_max, iy_max = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(inset_lon_max, inset_lat_max)

    ax_inset = inset_axes(ax, width="22%", height="22%", loc="lower left",
                          borderpad=1.5)
    ax_inset.set_xlim(ix_min, ix_max)
    ax_inset.set_ylim(iy_min, iy_max)
    ax_inset.set_facecolor("#111318")

    google_hybrid = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"
    cx.add_basemap(ax_inset, source=google_hybrid, crs="EPSG:3857", attribution="")

    # Draw red rectangle showing the main map extent
    rect = Rectangle(
        (extent_3857[0], extent_3857[2]),
        extent_3857[1] - extent_3857[0],
        extent_3857[3] - extent_3857[2],
        linewidth=2, edgecolor="#e74c3c", facecolor="none", zorder=5,
    )
    ax_inset.add_patch(rect)

    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    for spine in ax_inset.spines.values():
        spine.set_edgecolor("#3a3d45")
        spine.set_linewidth(1.5)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_stream_order_maps(da, so_geometries):
    """Render a row of small agreement maps, each clipped to one stream order's catchments.

    Returns a dict {stream_order: base64_png_string} for orders that have valid data.
    """
    from rasterio.features import geometry_mask
    import pyproj

    FONT = "Proxima Nova"
    plt.rcParams["font.family"] = ["Proxima Nova", "Helvetica Neue", "Helvetica", "Arial", "sans-serif"]

    vals = da.values
    valid = vals[~np.isnan(vals)]
    vmax = float(np.percentile(np.abs(valid), 95))

    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(alpha=0)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Reproject to Web Mercator for basemap
    da_3857 = da.rio.reproject("EPSG:3857")
    transformer = pyproj.Transformer.from_crs(da.rio.crs, "EPSG:3857", always_xy=True)

    # Shared extent across all SO maps (full agreement raster extent in 3857)
    extent_3857 = [
        float(da_3857.x.min()), float(da_3857.x.max()),
        float(da_3857.y.min()), float(da_3857.y.max()),
    ]

    results = {}
    orders = sorted(so_geometries.keys())

    for so in orders:
        geoms = so_geometries[so]

        # Clip the original raster by these catchment geometries
        try:
            clipped = da.rio.clip(geoms.values, da.rio.crs, drop=False, all_touched=True)
        except Exception:
            continue

        clipped_vals = clipped.values
        if np.all(np.isnan(clipped_vals)):
            continue

        # Reproject clipped raster to 3857
        clipped_3857 = clipped.rio.reproject("EPSG:3857")

        fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=120, facecolor="none")

        ax.set_xlim(extent_3857[0], extent_3857[1])
        ax.set_ylim(extent_3857[2], extent_3857[3])

        cx.add_basemap(ax, source=cx.providers.CartoDB.DarkMatter,
                       crs="EPSG:3857", attribution="")

        clipped_extent = [
            float(clipped_3857.x.min()), float(clipped_3857.x.max()),
            float(clipped_3857.y.min()), float(clipped_3857.y.max()),
        ]

        ax.imshow(
            np.ma.masked_invalid(clipped_3857.values),
            cmap=cmap, norm=norm,
            extent=clipped_extent, origin="upper",
            interpolation="nearest", alpha=0.7, zorder=2,
        )

        ax.set_facecolor("none")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="none")
        plt.close(fig)
        buf.seek(0)
        results[so] = base64.b64encode(buf.read()).decode("utf-8")

    return results


def _fmt_metric(idx, val):
    """Format a single metric value for table display."""
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    elif isinstance(val, (float, np.floating)):
        if "pct" in str(idx).lower():
            return f"{val:.1f}%"
        else:
            return f"{val:.4f}"
    return str(val)


def make_metrics_table(metrics_df, title, extra_columns=None):
    """Create a formatted Plotly table from a metrics DataFrame.

    Parameters
    ----------
    extra_columns : dict, optional
        {column_label: {metric_name: value, ...}} to add as additional columns.
        Metric names should match the index of the transposed metrics_df.
    """
    # Transpose so metrics are rows
    display = metrics_df.T.copy()
    display.columns = ["Value"]
    display.index.name = "Metric"

    # Clean up metric names for display
    labels = [idx.replace("_", " ").title() for idx in display.index]
    raw_indices = list(display.index)

    # Format main "All" column
    formatted_all = [_fmt_metric(idx, val) for idx, val in display["Value"].items()]

    # Build header and cell value lists
    header_vals = ["<b>Metric</b>", "<b>All</b>"]
    cell_vals = [labels, formatted_all]
    aligns = ["left", "right"]

    if extra_columns:
        for col_label in sorted(extra_columns.keys(), key=lambda x: str(x)):
            col_data = extra_columns[col_label]
            col_formatted = []
            for idx in raw_indices:
                if idx in col_data:
                    col_formatted.append(_fmt_metric(idx, col_data[idx]))
                else:
                    col_formatted.append("—")
            header_vals.append(f"<b>{col_label}</b>")
            cell_vals.append(col_formatted)
            aligns.append("right")

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_vals,
            fill_color="#2a3444",
            font=dict(color="#f0f0f0", size=13, family="Inter, sans-serif"),
            align=aligns,
            height=32,
            line=dict(color="#3a3d45"),
        ),
        cells=dict(
            values=cell_vals,
            fill_color=[["#1a1d24", "#1e2128"] * (len(labels) // 2 + 1)],
            font=dict(size=12, color="#d0d0d0", family="Inter, sans-serif"),
            align=aligns,
            height=28,
            line=dict(color="#2a2d35"),
        ),
    )])
    fig.update_layout(title=dict(text=title, font=dict(size=16)), margin=dict(l=20, r=20, t=50, b=20), **DARK_LAYOUT)
    return fig


def make_agreement_histogram(values):
    """Create a histogram of agreement map values centered on zero."""
    fig = go.Figure()

    # Symmetric x-axis centered on zero
    x_max = float(np.percentile(np.abs(values), 99))
    x_max = max(x_max, 1.0)

    n_bins = 100
    bin_edges = np.linspace(-x_max, x_max, n_bins + 1)
    counts, _ = np.histogram(values, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig.add_trace(go.Bar(
        x=bin_centers.tolist(),
        y=counts.tolist(),
        marker_color="#60a5fa",
        opacity=0.85,
        name="Depth Difference",
        width=(bin_edges[1] - bin_edges[0]) * 0.95,
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)

    median_val = float(np.median(values))
    fig.add_vline(x=median_val, line_dash="dot", line_color="#e67e22", line_width=2,
                  annotation_text=f"Median: {median_val:.2f} ft", annotation_position="top right")

    fig.update_layout(
        title=dict(text="Agreement Map: Distribution of Depth Differences (Candidate - Benchmark)", font=dict(size=16)),
        xaxis_title="Depth Difference (ft)",
        yaxis_title="Cell Count",
        xaxis_range=[-x_max, x_max],
        bargap=0.02,
        margin=dict(l=60, r=20, t=50, b=50),
        **DARK_LAYOUT,
    )
    return fig


def make_bucket_chart(structures_metrics):
    """Create a bar chart for agreement buckets."""
    row = structures_metrics.iloc[0]

    categories = ["< 1ft", "1-3ft", "3-5ft", "> 5ft"]
    counts = [
        int(row["n_within_1ft"]),
        int(row["n_within_3ft"] - row["n_within_1ft"]),
        int(row["n_within_5ft"] - row["n_within_3ft"]),
        int(row["n_gt_5ft"]),
    ]
    pcts = [c / sum(counts) * 100 for c in counts]
    colors = ["#22c55e", "#6ee7a0", "#bbf7d0", "#e8f5e9"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=counts,
        marker_color=colors,
        text=[f"{c:,}<br>({p:.1f}%)" for c, p in zip(counts, pcts)],
        textposition="outside",
        textfont=dict(size=13),
    ))

    fig.update_layout(
        title=dict(text="Structure Depth Agreement Buckets", font=dict(size=16)),
        xaxis_title="Absolute Depth Difference",
        yaxis_title="Number of Structures",
        margin=dict(l=60, r=20, t=50, b=50),
        showlegend=False,
        **DARK_LAYOUT,
    )
    return fig


def _fmt_area(sqkm):
    """Format area in sq km or sq mi for display."""
    sq_mi = sqkm * 0.386102
    if sqkm >= 1:
        return f"{sqkm:,.1f} km\u00b2"
    else:
        return f"{sqkm:,.2f} km\u00b2"


def _bias_bar_html(prefix, icon_svg, label, n_under, n_match, n_over,
                    total, area_sqkm=None):
    """Unified bias spectrum bar with left label, centered bar, right stats.

    All bars share the same fixed-width columns so they align perfectly.
    Hover tooltips on each segment show percentage, count, and area breakdown.
    """
    pct_under = n_under / total * 100 if total > 0 else 0
    pct_match = n_match / total * 100 if total > 0 else 0
    pct_over = n_over / total * 100 if total > 0 else 0

    min_w = 2.0
    widths = [max(pct_under, min_w), max(pct_match, min_w), max(pct_over, min_w)]
    sc = 100.0 / sum(widths)
    w_under = widths[0] * sc
    w_match = widths[1] * sc
    w_over = widths[2] * sc

    # Area breakdown per segment (proportional to count)
    area_str_right = _fmt_area(area_sqkm) if area_sqkm is not None else ""
    if area_sqkm is not None and total > 0:
        a_under = area_sqkm * n_under / total
        a_match = area_sqkm * n_match / total
        a_over = area_sqkm * n_over / total
        tip_under = f"Under-prediction: {pct_under:.1f}% &#10;Count: {n_under:,} &#10;Area: {_fmt_area(a_under)}"
        tip_match = f"Match: {pct_match:.1f}% &#10;Count: {n_match:,} &#10;Area: {_fmt_area(a_match)}"
        tip_over = f"Over-prediction: {pct_over:.1f}% &#10;Count: {n_over:,} &#10;Area: {_fmt_area(a_over)}"
    else:
        tip_under = f"Under-prediction: {pct_under:.1f}% &#10;Count: {n_under:,}"
        tip_match = f"Match: {pct_match:.1f}% &#10;Count: {n_match:,}"
        tip_over = f"Over-prediction: {pct_over:.1f}% &#10;Count: {n_over:,}"

    css = """
    <style>
        .__PFX__-wrap {
            background: transparent;
            padding: 6px 0;
            font-family: 'Proxima Nova', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .__PFX__-bar-track {
            position: relative;
            height: 11px;
            border-radius: 6px;
            overflow: hidden;
            display: flex;
            border: none;
        }
        .__PFX__-seg { height: 100%; cursor: default; }
        .__PFX__-seg-under {
            background: #7A3B50;
            width: __W_UNDER__;
            border-radius: 6px 0 0 6px;
        }
        .__PFX__-seg-match {
            background: #E8D8D0;
            width: __W_MATCH__;
        }
        .__PFX__-seg-over {
            background: #2E5280;
            width: __W_OVER__;
            border-radius: 0 6px 6px 0;
        }
        .__PFX__-bar-track::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 50%;
            border-radius: 6px 6px 0 0;
            background: linear-gradient(180deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.03) 100%);
            pointer-events: none;
        }
    </style>
    """

    css = css.replace("__PFX__", prefix)
    css = css.replace("__W_UNDER__", f"{w_under:.2f}%")
    css = css.replace("__W_MATCH__", f"{w_match:.2f}%")
    css = css.replace("__W_OVER__", f"{w_over:.2f}%")

    html_body = f"""
    <div class="{prefix}-wrap">
        <div style="display:flex; align-items:center; gap:0;">
            <div style="width:110px; flex-shrink:0; display:flex; align-items:center; gap:6px;">
                {icon_svg}
                <span style="color:#e0e0e0; font-size:12px; font-weight:500; white-space:nowrap;">{label}</span>
            </div>
            <div style="flex:1; min-width:0;">
                <div class="{prefix}-bar-track">
                    <div class="{prefix}-seg {prefix}-seg-under" title="{tip_under}"></div>
                    <div class="{prefix}-seg {prefix}-seg-match" title="{tip_match}"></div>
                    <div class="{prefix}-seg {prefix}-seg-over" title="{tip_over}"></div>
                </div>
            </div>
            <div style="width:160px; flex-shrink:0; text-align:right; white-space:nowrap;">
                <span style="color:#6a6d75; font-size:10px; font-weight:400;">n={total:,}</span>
                {"" if not area_str_right else f'<span style="color:#4a4d55; font-size:10px; margin:0 4px;">|</span><span style="color:#6a6d75; font-size:10px; font-weight:400;">{area_str_right}</span>'}
            </div>
        </div>
    </div>
    """

    return css + html_body


# Shared SVG icons
_HOUSE_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
    'fill="none" stroke="#e0e0e0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>'
    '<polyline points="9 22 9 12 15 12 15 22"></polyline></svg>'
)

_GRID_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
    'fill="none" stroke="#e0e0e0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
    'style="flex-shrink:0;">'
    '<rect x="3" y="3" width="7" height="7"></rect>'
    '<rect x="14" y="3" width="7" height="7"></rect>'
    '<rect x="3" y="14" width="7" height="7"></rect>'
    '<rect x="14" y="14" width="7" height="7"></rect>'
    '</svg>'
)

_WAVE_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
    'fill="none" stroke="#e0e0e0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M2 6c.6.5 1.2 1 2.5 1C7 7 7 5 9.5 5c2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"></path>'
    '<path d="M2 12c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"></path>'
    '<path d="M2 18c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"></path>'
    '</svg>'
)


def make_bias_chart(structures_metrics, area_sqkm=None):
    """Create a structures bias spectrum bar."""
    row = structures_metrics.iloc[0]
    n_over = int(row["n_over_predict"])
    n_under = int(row["n_under_predict"])
    n_match = int(row["n_match"])
    total = n_over + n_under + n_match
    return _bias_bar_html("fi", _HOUSE_ICON, "Structures", n_under, n_match, n_over, total, area_sqkm)


def make_raster_bias_chart(agreement_values, epsilon=1.0, area_sqkm=None):
    """Create a raster-level bias spectrum bar."""
    import numpy as np
    vals = np.asarray(agreement_values)
    n_under = int(np.sum(vals < -epsilon))
    n_match = int(np.sum(np.abs(vals) <= epsilon))
    n_over = int(np.sum(vals > epsilon))
    total = n_under + n_match + n_over
    return _bias_bar_html("ri", _GRID_ICON, "All Pixels", n_under, n_match, n_over, total, area_sqkm)


def make_stream_order_histograms(so_distributions):
    """Create a row of small histograms showing per-catchment mean depth diff by stream order.

    Returns (fig, subplot_domains) where subplot_domains is a list of (left, right)
    fractions for each column, so the map grid above can align to them.
    """
    orders = sorted(so_distributions.keys())
    n = len(orders)
    if n == 0:
        return go.Figure(), []

    h_spacing = 0.04
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[f"SO{so}" for so in orders],
        horizontal_spacing=h_spacing,
    )

    # Extract subplot x-domains for alignment
    subplot_domains = []
    for i in range(1, n + 1):
        xaxis = f'xaxis{i}' if i > 1 else 'xaxis'
        domain = fig.layout[xaxis].domain
        subplot_domains.append((domain[0], domain[1]))

    # Compute shared x-axis range across all stream orders
    all_vals = np.concatenate([so_distributions[so] for so in orders])
    x_max = float(np.percentile(np.abs(all_vals), 99))
    x_max = max(x_max, 1.0)

    n_bins = 40
    bin_edges = np.linspace(-x_max, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = (bin_edges[1] - bin_edges[0]) * 0.92

    # Pre-compute all histograms to find shared y-axis max
    hist_data = []
    y_max = 0
    for so in orders:
        vals = so_distributions[so]
        counts, _ = np.histogram(vals, bins=bin_edges)
        median_val = float(np.median(vals))
        hist_data.append((counts, median_val))
        y_max = max(y_max, int(counts.max()))
    y_max = int(y_max * 1.1)  # 10% padding

    for i, (so, (counts, median_val)) in enumerate(zip(orders, hist_data), 1):
        fig.add_trace(go.Bar(
            x=bin_centers.tolist(),
            y=counts.tolist(),
            marker_color="#60a5fa",
            opacity=0.85,
            width=bar_width,
            name=f"SO{so}",
            showlegend=False,
        ), row=1, col=i)

        fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5,
                       row=1, col=i)
        fig.add_vline(x=median_val, line_dash="dot", line_color="#e67e22", line_width=1.5,
                       row=1, col=i)

        fig.update_xaxes(range=[-x_max, x_max], row=1, col=i,
                          gridcolor="#2a2d35", zerolinecolor="#3a3d45",
                          title_text="Depth Diff (ft)" if i == 1 else "")
        fig.update_yaxes(range=[0, y_max], gridcolor="#2a2d35", zerolinecolor="#3a3d45",
                          title_text="Catchments" if i == 1 else "",
                          row=1, col=i)

    fig.update_layout(
        height=300,
        margin=dict(l=50, r=20, t=40, b=50),
        bargap=0.02,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1a1d24",
        font=dict(family="Proxima Nova, Inter, sans-serif", color="#e0e0e0"),
    )

    # Style subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=13, color="#e0e0e0",
                                   family="Proxima Nova, Inter, sans-serif")

    return fig, subplot_domains


def make_stream_order_bias_charts(so_metrics):
    """Create spectrum bias bars for each stream order."""
    all_html = []
    for so in sorted(so_metrics.keys()):
        m = so_metrics[so]
        n_under = m["n_under"]
        n_match = m["n_match"]
        n_over = m["n_over"]
        total = n_under + n_match + n_over
        if total == 0:
            continue
        prefix = f"so{so}"
        label = f"SO{so}"
        area_sqkm = m.get("area_sqkm")
        all_html.append(_bias_bar_html(prefix, _WAVE_ICON, label, n_under, n_match, n_over, total, area_sqkm))
    return "\n".join(all_html)


def make_structure_boxplot(structures_gdf):
    """Create a box plot of per-structure mean depth differences."""
    diff = structures_gdf["mean_depth_diff"].values

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=diff,
        name="Mean Depth Diff",
        marker_color="#3498db",
        boxmean="sd",
    ))

    fig.update_layout(
        title=dict(text="Per-Structure Mean Depth Difference Distribution", font=dict(size=16)),
        yaxis_title="Depth Difference (ft)",
        margin=dict(l=60, r=20, t=50, b=30),
        showlegend=False,
        **DARK_LAYOUT,
    )
    return fig


def make_structure_histogram(structures_gdf):
    """Create a histogram of per-structure mean depth differences."""
    diff = structures_gdf["mean_depth_diff"].values

    # Symmetric x-axis centered on zero
    x_max = float(np.percentile(np.abs(diff), 99))
    x_max = max(x_max, 1.0)

    counts, bin_edges = np.histogram(diff, bins=80, range=(-x_max, x_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers.tolist(),
        y=counts.tolist(),
        marker_color="#a78bfa",
        opacity=0.85,
        width=(bin_edges[1] - bin_edges[0]) * 0.95,
        name="Per-Structure Diff",
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)

    fig.update_layout(
        title=dict(text="Per-Structure Mean Depth Difference Distribution", font=dict(size=16)),
        xaxis_title="Mean Depth Difference (ft)",
        yaxis_title="Structure Count",
        xaxis_range=[-x_max, x_max],
        bargap=0.02,
        margin=dict(l=60, r=20, t=50, b=50),
        **DARK_LAYOUT,
    )
    return fig


def _caption(text):
    """Wrap text in a styled caption paragraph."""
    return f'<p class="caption">{text}</p>'


def build_executive_summary(data, paths):
    """Generate an executive summary HTML block from the loaded data and file paths."""
    bullets = []

    # Data sources (show filename only)
    if paths.get("candidate_map"):
        bullets.append(f"<b>Candidate map:</b> <code>{os.path.basename(paths['candidate_map'])}</code>")
    if paths.get("benchmark_map"):
        bullets.append(f"<b>Benchmark map:</b> <code>{os.path.basename(paths['benchmark_map'])}</code>")
    if paths.get("structures"):
        source = paths.get("structures_source", "")
        if source:
            bullets.append(f'<b>Structures:</b> <a href="{source}" style="color:#60a5fa;">{source}</a>')
        else:
            bullets.append(f"<b>Structures:</b> <code>{os.path.basename(paths['structures'])}</code>")

    # Area evaluated
    if "agreement_da" in data:
        da = data["agreement_da"]
        crs = da.rio.crs
        bounds = da.rio.bounds()  # (left, bottom, right, top)
        n_valid = int(np.sum(da.notnull().values))
        n_total = int(da.size)
        bullets.append(
            f"<b>Analysis area:</b> {n_valid:,} of {n_total:,} raster cells have valid data "
            f"(CRS: {crs})"
        )

    # Raster-level takeaway
    if "depth_metrics" in data:
        dm = data["depth_metrics"].iloc[0]
        r2 = dm.get("coefficient_of_determination", None)
        mae = dm.get("mean_absolute_error", None)
        rmse = dm.get("root_mean_squared_error", None)
        parts = []
        if r2 is not None:
            parts.append(f"R² = {r2:.3f}")
        if mae is not None:
            parts.append(f"MAE = {mae:.2f} ft")
        if rmse is not None:
            parts.append(f"RMSE = {rmse:.2f} ft")
        if parts:
            bullets.append(f"<b>Raster depth agreement:</b> {', '.join(parts)}")

    # Structure-level takeaway
    if "structures_metrics" in data:
        sm = data["structures_metrics"].iloc[0]
        n_struct = int(sm.get("structures_in_domain", 0))
        pct_1ft = sm.get("pct_within_1ft", 0)
        pct_3ft = sm.get("pct_within_3ft", 0)
        pct_5ft = sm.get("pct_within_5ft", 0)
        mae_s = sm.get("mean_absolute_error", 0)
        pct_over = sm.get("pct_over_predict", 0)
        pct_under = sm.get("pct_under_predict", 0)

        bullets.append(f"<b>Structures evaluated:</b> {n_struct:,} buildings within the flood domain")
        bullets.append(
            f"<b>Structure depth agreement:</b> "
            f"{pct_1ft:.1f}% within 1 ft, "
            f"{pct_3ft:.1f}% within 3 ft, "
            f"{pct_5ft:.1f}% within 5 ft "
            f"(MAE = {mae_s:.2f} ft)"
        )

        # Bias summary
        if pct_under > 80:
            bias_desc = f"The candidate strongly under-predicts depth at structures ({pct_under:.0f}% of buildings)."
        elif pct_over > 80:
            bias_desc = f"The candidate strongly over-predicts depth at structures ({pct_over:.0f}% of buildings)."
        elif pct_under > pct_over:
            bias_desc = f"The candidate tends to under-predict depth ({pct_under:.0f}% vs {pct_over:.0f}% over-predict)."
        elif pct_over > pct_under:
            bias_desc = f"The candidate tends to over-predict depth ({pct_over:.0f}% vs {pct_under:.0f}% under-predict)."
        else:
            bias_desc = "The candidate shows no strong directional bias."
        bullets.append(f"<b>Bias:</b> {bias_desc}")

    # Build headline sentences
    sentences = []

    if "depth_metrics" in data:
        dm = data["depth_metrics"].iloc[0]
        mae = dm.get("mean_absolute_error", None)
        r2 = dm.get("coefficient_of_determination", None)
        mse_val = dm.get("mean_signed_error", 0)
        if mae is not None:
            r2_part = f" and an R² of {r2:.2f}" if r2 is not None else ""
            if mae < 1:
                quality_html = '<b style="color:#4ade80;">good agreement</b>'
                sentences.append(
                    f"The candidate map shows {quality_html} "
                    f"with the benchmark, with a mean absolute error of {mae:.1f} ft{r2_part}."
                )
            elif mse_val < -1:
                quality_html = '<b style="color:#f87171;">under-prediction</b>'
                sentences.append(
                    f"The candidate map shows {quality_html} "
                    f"relative to the benchmark, with a mean absolute error of {mae:.1f} ft{r2_part}."
                )
            elif mse_val > 1:
                quality_html = '<b style="color:#60a5fa;">over-prediction</b>'
                sentences.append(
                    f"The candidate map shows {quality_html} "
                    f"relative to the benchmark, with a mean absolute error of {mae:.1f} ft{r2_part}."
                )
            else:
                quality_html = '<b style="color:#fbbf24;">moderate agreement</b>'
                sentences.append(
                    f"The candidate map shows {quality_html} "
                    f"with the benchmark, with a mean absolute error of {mae:.1f} ft{r2_part}."
                )

    if "structures_metrics" in data:
        sm = data["structures_metrics"].iloc[0]
        pct_3ft = sm.get("pct_within_3ft", 0)
        n_struct = int(sm.get("structures_in_domain", 0))
        pct_over = sm.get("pct_over_predict", 0)
        pct_under = sm.get("pct_under_predict", 0)


        # Pick the most meaningful bucket to highlight, colored by quality
        pct_1ft = sm.get("pct_within_1ft", 0)
        pct_5ft = sm.get("pct_within_5ft", 0)
        pct_gt5 = sm.get("pct_gt_5ft", 0)

        if pct_1ft >= 80:
            pct_color = "#4ade80"
        elif pct_1ft >= 60:
            pct_color = "#fbbf24"
        elif pct_1ft >= 40:
            pct_color = "#fb923c"
        else:
            pct_color = "#f87171"

        # Determine bias direction and intensity
        bias_dir = "under" if pct_under > pct_over else "over"
        bias_pct = max(pct_under, pct_over)
        if bias_pct >= 80:
            intensity = "greatly"
        elif bias_pct >= 60:
            intensity = "moderately"
        elif bias_pct >= 45:
            intensity = "slightly"
        else:
            intensity = None  # no clear bias

        if pct_1ft >= 80:
            quality_word = '<b style="color:#4ade80;">predicted well</b>'
        elif pct_1ft >= 60:
            quality_word = '<b style="color:#fbbf24;">predicted with moderate accuracy</b>'
        elif intensity and bias_dir == "under":
            quality_word = f'<b style="color:#f87171;">{intensity} under-predicted</b>'
        elif intensity and bias_dir == "over":
            quality_word = f'<b style="color:#60a5fa;">{intensity} over-predicted</b>'
        elif pct_1ft >= 40:
            quality_word = '<b style="color:#fb923c;">predicted with weak accuracy</b>'
        else:
            quality_word = '<b style="color:#f87171;">predicted poorly</b>'

        sentences.append(
            f"Flood depths at structures were {quality_word}, with "
            f'<b>{pct_1ft:.0f}%</b> of structures '
            f"having an absolute mean depth difference within 1 ft of the benchmark "
            f"and <b>{pct_3ft:.0f}%</b> within 3 ft."
        )

    headline = "<br><br>".join(sentences)

    bullet_html = "\n".join(f"<li>{b}</li>" for b in bullets)
    headline_html = f'<p class="headline">{headline}</p>' if headline else ""

    takeaway_html = f"""
    <div class="summary-box">
        <h2>Key Takeaways</h2>
        {headline_html}
        __BIAS_CHART__
    </div>
    """

    details_html = f"""
    <div class="summary-box">
        <h2>Run Details</h2>
        <ul>{bullet_html}</ul>
    </div>
    """

    return takeaway_html, details_html


def build_html(data, title, paths=None, offline=True):
    """Assemble all figures into a single HTML string."""
    if paths is None:
        paths = {}

    sections = []

    # Header
    sections.append(f"""
    <div style="color:#f0f0f0; padding:28px 0px; margin-bottom:24px;">
        <h1 style="margin:0 0 8px 0; font-size:30px; font-weight:600; letter-spacing:-0.5px; font-family:inherit;">{title}</h1>
        <p style="margin:0; opacity:0.5; font-size:13px; font-weight:300; font-family:inherit;">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """)

    # Key takeaways + spectrum chart at the top; run details go to the bottom
    takeaway_html, details_html = build_executive_summary(data, paths)
    bias_charts = ""
    if "agreement_values" in data:
        bias_charts += make_raster_bias_chart(data["agreement_values"], area_sqkm=data.get("raster_area_sqkm"))
    if "structures_metrics" in data:
        bias_charts += make_bias_chart(data["structures_metrics"], area_sqkm=data.get("structures_area_sqkm"))
    if "stream_order_metrics" in data:
        bias_charts += make_stream_order_bias_charts(data["stream_order_metrics"])
    sections.append(takeaway_html.replace("__BIAS_CHART__", bias_charts))

    # Agreement map image
    if "agreement_da" in data:
        b64 = render_agreement_map(data["agreement_da"])
        sections.append(f"""
        <div class="chart-container" style="text-align:center;">
            <img src="data:image/png;base64,{b64}" style="max-width:100%; height:auto; border-radius:4px;" />
            {_caption('Map of depth differences between the candidate and benchmark maps.'
                      '<span style="color:#e74c3c;">Red areas</span> indicate the candidate under-predicts depth; '
                      '<span style="color:#60a5fa;">blue areas</span> indicate over-prediction. '
                      'Transparent regions had no flooding in either map.')}
        </div>
        """)

    # Raster metrics table (with per-SO columns if available)
    if "depth_metrics" in data:
        so_extra = None
        if "stream_order_raster_metrics" in data:
            so_extra = {f"SO{so}": metrics
                        for so, metrics in data["stream_order_raster_metrics"].items()}
        fig = make_metrics_table(data["depth_metrics"], "Raster Depth Metrics",
                                  extra_columns=so_extra)
        h = max(400, len(data["depth_metrics"].columns) * 30 + 80)
        fig.update_layout(height=h)
        sections.append(f"""<div class="chart-container">
            {fig.to_html(full_html=False, include_plotlyjs=False)}
            {_caption("Cell-level error metrics computed across all raster cells where either map shows flooding. "
                      "Lower MAE and RMSE indicate better agreement; R² closer to 1.0 indicates stronger correlation.")}
        </div>""")

    # Agreement map histogram
    if "agreement_values" in data:
        fig = make_agreement_histogram(data["agreement_values"])
        fig.update_layout(height=400)
        sections.append(f"""<div class="chart-container">
            {fig.to_html(full_html=False, include_plotlyjs=False)}
            {_caption('Distribution of per-cell depth differences. Values near zero indicate agreement. '
                      'A left-skewed distribution means the candidate generally under-predicts; right-skewed means over-prediction. '
                      'The <span style="color:#e74c3c;">dashed red line</span> marks zero difference; '
                      'the <span style="color:#e67e22;">dotted orange line</span> marks the median.')}
        </div>""")

    # Stream order maps + histograms (aligned via Plotly subplot domains)
    has_so_maps = "stream_order_geometries" in data and "agreement_da" in data
    has_so_dists = "stream_order_distributions" in data
    if has_so_maps or has_so_dists:
        # Build histograms first to get subplot domains for alignment
        so_fig = None
        subplot_domains = []
        if has_so_dists:
            so_fig, subplot_domains = make_stream_order_histograms(data["stream_order_distributions"])

        so_maps = {}
        if has_so_maps:
            so_maps = render_stream_order_maps(data["agreement_da"], data["stream_order_geometries"])

        so_orders = sorted(
            so_maps.keys() if so_maps
            else data.get("stream_order_distributions", {}).keys()
        )
        n_cols = len(so_orders)

        if so_maps and n_cols > 0 and subplot_domains:
            # Down-arrow SVG (thin, subtle)
            arrow_svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="16" viewBox="0 0 12 16" '
                'fill="none" stroke="#6a6d75" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
                '<line x1="6" y1="1" x2="6" y2="13"></line>'
                '<polyline points="2 10 6 14 10 10"></polyline>'
                '</svg>'
            )

            # Build grid-template-columns from Plotly subplot domains
            # Each domain is (left_frac, right_frac) of the plot area.
            # We need to account for gaps between subplots as well.
            col_parts = []
            prev_right = 0.0
            for idx, (dl, dr) in enumerate(subplot_domains):
                gap = dl - prev_right
                if gap > 0.001:
                    col_parts.append(f"{gap * 100:.2f}%")  # spacer
                col_parts.append(f"{(dr - dl) * 100:.2f}%")  # column
                prev_right = dr

            grid_cols = " ".join(col_parts)

            # Build cells (with empty spacers for gaps)
            map_cells = ""
            arrow_cells = ""
            cell_idx = 0
            prev_right = 0.0
            for idx, so in enumerate(so_orders):
                if idx >= len(subplot_domains):
                    break
                dl, dr = subplot_domains[idx]
                gap = dl - prev_right
                if gap > 0.001:
                    map_cells += '<div></div>'
                    arrow_cells += '<div></div>'

                if so in so_maps:
                    map_cells += (
                        f'<div style="text-align:center;">'
                        f'<img src="data:image/png;base64,{so_maps[so]}" '
                        f'style="width:100%; height:auto; border-radius:4px;" />'
                        f'</div>'
                    )
                else:
                    map_cells += '<div></div>'
                arrow_cells += f'<div style="text-align:center;">{arrow_svg}</div>'
                prev_right = dr

            # Plotly margins: l=50px, r=20px — apply as padding on the grid container
            sections.append(f"""<div class="chart-container" style="padding-bottom:0; margin-bottom:0;">
                <div style="display:grid; grid-template-columns:{grid_cols}; padding:0 20px 0 50px;">
                    {map_cells}
                </div>
                <div style="display:grid; grid-template-columns:{grid_cols}; padding:4px 20px 0 50px;">
                    {arrow_cells}
                </div>
            </div>""")

        if so_fig is not None:
            sections.append(f"""<div class="chart-container" style="padding-top:0;">
                {so_fig.to_html(full_html=False, include_plotlyjs=False)}
                {_caption('Per-catchment mean depth difference distributions by stream order. '
                          'The <span style="color:#e74c3c;">dashed red line</span> marks zero; '
                          'the <span style="color:#e67e22;">dotted orange line</span> marks the median. '
                          'Distributions shifted right indicate over-prediction; left indicates under-prediction.')}
            </div>""")

    # Structure metrics table
    if "structures_metrics" in data:
        fig = make_metrics_table(data["structures_metrics"], "Structure Depth Metrics")
        h = max(400, len(data["structures_metrics"].columns) * 30 + 80)
        fig.update_layout(height=h)
        sections.append(f"""<div class="chart-container">
            {fig.to_html(full_html=False, include_plotlyjs=False)}
            {_caption("Summary metrics for building footprints within the flood domain. "
                      "Agreement buckets show what fraction of structures have depth predictions within 1, 3, or 5 feet of the benchmark.")}
        </div>""")

    # Bucket chart
    if "structures_metrics" in data:
        bucket_fig = make_bucket_chart(data["structures_metrics"])
        bucket_fig.update_layout(height=420)
        sections.append(f"""<div class="chart-container">
            {bucket_fig.to_html(full_html=False, include_plotlyjs=False)}
            {_caption('Number of structures in each depth agreement bucket. '
                      '<span style="color:#27ae60;">Green bars</span> indicate close agreement; '
                      '<span style="color:#e74c3c;">red bars</span> indicate large discrepancies. '
                      'A good candidate model will concentrate structures in the leftmost bars.')}
        </div>""")

    # Per-structure histogram
    if "structures_gdf" in data:
        hist_fig = make_structure_histogram(data["structures_gdf"])
        hist_fig.update_layout(height=420)
        sections.append(f"""<div class="chart-container">
            {hist_fig.to_html(full_html=False, include_plotlyjs=False)}
            {_caption('Histogram of mean depth difference per structure. Negative values mean the candidate predicts less flooding '
                      'than the benchmark at that building; positive values mean more. '
                      'A tight cluster around the <span style="color:#e74c3c;">dashed red zero line</span> is ideal.')}
        </div>""")

    # (Bias direction donut is shown at the top alongside the summary)

    # Per-structure box plot (disabled)
    # if "structures_gdf" in data:
    #     box_fig = make_structure_boxplot(data["structures_gdf"])
    #     box_fig.update_layout(height=420)
    #     sections.append(f"""<div class="chart-container">
    #         {box_fig.to_html(full_html=False, include_plotlyjs=False)}
    #     </div>""")

    # Run details bullet list at the very bottom
    sections.append(details_html)

    # Assemble full HTML
    body = "\n".join(sections)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Proxima Nova', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #111318;
            color: #e0e0e0;
            padding: 24px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .chart-container {{
            background: transparent;
            
            padding: 16px;
            margin-bottom: 16px;
            
        }}
        .caption {{
            color: #8a8d94;
            font-size: 12.5px;
            font-weight: 300;
            line-height: 1.6;
            margin: 10px 4px 0 4px;
            padding-top: 8px;
            border-top: 1px solid #2a2d35;
        }}
        .summary-box {{
            background: transparent;
            padding: 0px 0px 16px 0px;
            margin-bottom: 16px;
        }}
        .summary-box h2 {{
            color: #f0f0f0;
            font-size: 20px;
            font-weight: 600;
            margin: 0 0 12px 0;
        }}
        .summary-box .headline {{
            color: #e8e8e8;
            font-size: 15px;
            font-weight: 400;
            line-height: 1.6;
            margin: 0 0 16px 0;
            padding-bottom: 14px;
            border-bottom: 1px solid #2a2d35;
        }}
        .summary-box ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .summary-box li {{
            color: #c8cad0;
            font-size: 13.5px;
            font-weight: 300;
            line-height: 1.7;
            padding: 6px 0;
            border-bottom: 1px solid #22252c;
        }}
        .summary-box li:last-child {{
            border-bottom: none;
        }}
        .summary-box li b {{
            color: #e0e0e0;
            font-weight: 500;
        }}
        .summary-box code {{
            background: #22252c;
            color: #9ba0aa;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
            word-break: break-all;
        }}
    </style>
</head>
<body>
{body}
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate HTML dashboard from depth comparison outputs.")
    parser.add_argument("--depth-metrics", type=str, required=True, help="Path to raster depth metrics parquet.")
    parser.add_argument("--structures-metrics", type=str, default=None, help="Path to structures metrics parquet.")
    parser.add_argument("--structures-gpkg", type=str, default=None, help="Path to structures GeoPackage.")
    parser.add_argument("--agreement-map", type=str, default=None, help="Path to agreement map raster (for histogram).")
    parser.add_argument("--candidate-map", type=str, default=None, help="Path to candidate depth map (for summary display).")
    parser.add_argument("--benchmark-map", type=str, default=None, help="Path to benchmark depth map (for summary display).")
    parser.add_argument("--structures", type=str, default=None, help="Path to structures file (for summary display).")
    parser.add_argument("--structures-source", type=str, default=None, help="URL source for structures data.")
    parser.add_argument("--catchments", type=str, default=None, help="Path to NWM catchments GeoPackage.")
    parser.add_argument("--flows", type=str, default=None, help="Path to NWM flows GeoPackage with stream order (order_ column).")
    parser.add_argument("--so-metrics-dir", type=str, default=None,
                        help="Directory with per-SO GVAL metrics parquets (from depth_compare.py --catchments).")
    parser.add_argument("--output", type=str, default="report.html", help="Output HTML file path.")
    parser.add_argument("--title", type=str, default="Depth Comparison Report", help="Report title.")
    parser.add_argument("--units", type=str, default="meters", choices=["meters", "feet"],
                        help="Units of the input rasters. If 'meters', values are converted to feet for display. If 'feet', no conversion is applied.")

    args = parser.parse_args()

    print("Loading data...")
    data = load_data(
        args.depth_metrics,
        structures_metrics_path=args.structures_metrics,
        structures_gpkg_path=args.structures_gpkg,
        agreement_map_path=args.agreement_map,
        catchments_path=args.catchments,
        flows_path=args.flows,
        so_metrics_dir=args.so_metrics_dir,
        units=args.units,
    )

    paths = {
        "candidate_map": args.candidate_map,
        "benchmark_map": args.benchmark_map,
        "structures": args.structures,
        "structures_source": args.structures_source,
    }

    print("Building report...")
    html = build_html(data, args.title, paths=paths)

    with open(args.output, "w") as f:
        f.write(html)

    print(f"Report saved to {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
