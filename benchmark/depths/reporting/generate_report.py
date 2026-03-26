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

    if agreement_map_path and os.path.exists(agreement_map_path):
        da = rxr.open_rasterio(agreement_map_path, masked=True).squeeze().compute()
        if convert:
            da = da * scale
        data["agreement_da"] = da
        data["agreement_values"] = da.values[~np.isnan(da.values)].flatten()

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
                           facecolor="#1a1d24")

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

    ax.set_facecolor("#1a1d24")

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Depth Difference (ft) — Red: under-predict | Blue: over-predict", fontsize=11, color="#e0e0e0", fontfamily=FONT)
    cbar.ax.yaxis.set_tick_params(color="#e0e0e0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#e0e0e0", fontfamily=FONT, fontsize=9)

    ax.set_title("Agreement Map", fontsize=14, fontweight="bold", color="#f0f0f0", fontfamily=FONT)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

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
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#1a1d24")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_metrics_table(metrics_df, title):
    """Create a formatted Plotly table from a metrics DataFrame."""
    # Transpose so metrics are rows
    display = metrics_df.T.copy()
    display.columns = ["Value"]
    display.index.name = "Metric"

    # Format values
    formatted = []
    for idx, val in display["Value"].items():
        if isinstance(val, (int, np.integer)):
            formatted.append(f"{val:,}")
        elif isinstance(val, (float, np.floating)):
            if "pct" in str(idx).lower():
                formatted.append(f"{val:.1f}%")
            else:
                formatted.append(f"{val:.4f}")
        else:
            formatted.append(str(val))

    # Clean up metric names for display
    labels = [idx.replace("_", " ").title() for idx in display.index]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Value</b>"],
            fill_color="#2a3444",
            font=dict(color="#f0f0f0", size=13, family="Inter, sans-serif"),
            align=["left", "right"],
            height=32,
            line=dict(color="#3a3d45"),
        ),
        cells=dict(
            values=[labels, formatted],
            fill_color=[["#1a1d24", "#1e2128"] * (len(labels) // 2 + 1)],
            font=dict(size=12, color="#d0d0d0", family="Inter, sans-serif"),
            align=["left", "right"],
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


def make_bias_chart(structures_metrics):
    """Create a pie chart for bias direction."""
    row = structures_metrics.iloc[0]

    labels = ["Over-predict", "Under-predict", "Match"]
    values = [int(row["n_over_predict"]), int(row["n_under_predict"]), int(row["n_match"])]
    colors = ["#e74c3c", "#3498db", "#27ae60"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont=dict(size=11),
        hole=0.35,
    )])

    fig.update_layout(
        title=dict(text="Structure Bias Direction", font=dict(size=16)),
        margin=dict(l=40, r=40, t=50, b=40),
        **DARK_LAYOUT,
    )
    return fig


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

    return f"""
    <div class="summary-box">
        <h2>Key Takeaways</h2>
        {headline_html}
        <ul>{bullet_html}</ul>
    </div>
    """


def build_html(data, title, paths=None, offline=True):
    """Assemble all figures into a single HTML string."""
    if paths is None:
        paths = {}

    sections = []

    # Header
    sections.append(f"""
    <div style="background:linear-gradient(135deg, #1e2530, #2a3444); color:#f0f0f0; padding:28px 36px; margin-bottom:24px; border-radius:10px; border:1px solid #3a3d45;">
        <h1 style="margin:0 0 8px 0; font-size:30px; font-weight:600; letter-spacing:-0.5px;">{title}</h1>
        <p style="margin:0; opacity:0.5; font-size:13px; font-weight:300;">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """)

    # Executive summary + bias donut side by side
    summary_html = build_executive_summary(data, paths)
    if "structures_metrics" in data:
        bias_fig = make_bias_chart(data["structures_metrics"])
        bias_fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
        donut_html = bias_fig.to_html(full_html=False, include_plotlyjs=False)
        sections.append(f"""
        <div style="display:flex; gap:24px; align-items:stretch;">
            <div style="flex:1; min-width:0;">{summary_html}</div>
            <div style="flex:1; min-width:0;" class="chart-container">{donut_html}</div>
        </div>
        """)
    else:
        sections.append(summary_html)

    # Agreement map image
    if "agreement_da" in data:
        b64 = render_agreement_map(data["agreement_da"])
        sections.append(f"""
        <div class="chart-container" style="text-align:center;">
            <img src="data:image/png;base64,{b64}" style="max-width:100%; height:auto; border-radius:4px;" />
            {_caption('Spatial view of depth differences between the candidate and benchmark maps. '
                      '<span style="color:#e74c3c;">Red areas</span> indicate the candidate under-predicts depth; '
                      '<span style="color:#60a5fa;">blue areas</span> indicate over-prediction. '
                      'Transparent regions had no flooding in either map.')}
        </div>
        """)

    # Raster metrics table
    if "depth_metrics" in data:
        fig = make_metrics_table(data["depth_metrics"], "Raster Depth Metrics")
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

    # Per-structure box plot
    if "structures_gdf" in data:
        box_fig = make_structure_boxplot(data["structures_gdf"])
        box_fig.update_layout(height=420)
        sections.append(f"""<div class="chart-container">
            {box_fig.to_html(full_html=False, include_plotlyjs=False)}
            {_caption('Box plot showing the spread of per-structure depth differences. '
                      'The <span style="color:#3498db;">box</span> spans the interquartile range (25th–75th percentile); '
                      'the line is the median; the diamond marks the mean. '
                      'Whiskers extend to 1.5x IQR; dots beyond are outliers.')}
        </div>""")

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
            background: #1a1d24;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid #2a2d35;
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
            background: #1a1d24;
            border-radius: 10px;
            padding: 24px 32px;
            margin-bottom: 16px;
            border: 1px solid #2a2d35;
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
