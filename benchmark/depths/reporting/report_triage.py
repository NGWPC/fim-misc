"""
Generate a spatial discrepancies HTML report showing the worst-performing
structures and catchments, with filterable maps at a fixed 1:11,200 scale.

Example usage:
    python report_spatial_discrepancies.py \
        --structures-gpkg outputs/structures.gpkg \
        --catchments inputs/catchments.fgb \
        --agreement-map outputs/agreement_map.tif \
        --top-pct 5 \
        --units feet \
        --output discrepancies.html
"""
from __future__ import annotations

import argparse
import base64
import os
import tempfile
import time
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import box as shapely_box

M_TO_FT = 3.28084

# Map rendering constants
IMG_SIZE = 300
DPI = 96

# Structures: 1:11,200
SCALE_STRUCT = 11_200
METERS_PER_PX_S = (1 / DPI) * 0.0254 * SCALE_STRUCT   # ~2.966 m/px
HALF_M_S = IMG_SIZE * METERS_PER_PX_S / 2               # ~445 m

# Catchments: 1:205,000
SCALE_CATCH = 205_000
METERS_PER_PX_C = (1 / DPI) * 0.0254 * SCALE_CATCH    # ~54.2 m/px
HALF_M_C = IMG_SIZE * METERS_PER_PX_C / 2              # ~8,125 m

BG_COLOR = (17, 19, 24)       # #111318
TICK_COLOR = (80, 83, 95)
LABEL_COLOR = (130, 133, 145)
HIGHLIGHT_COLOR = (240, 240, 255)
HIGHLIGHT_YELLOW = (255, 215, 0)
INSET_SIZE = 85          # px — locator inset thumbnail
INSET_MARGIN = 3         # px gap from card edge
INSET_AOI_COLOR = (255, 140, 0)   # orange viewport outline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_b64(img, quality=82):
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _val_to_rgb(val, vmax, alpha=0.82):
    """Map a numeric depth-diff value to an RGB tuple via RdBu colormap."""
    normed = (np.clip(val / vmax, -1.0, 1.0) + 1.0) / 2.0
    r_, g_, b_, _ = plt.cm.RdBu(normed)
    r = int(r_ * 255 * alpha + BG_COLOR[0] * (1 - alpha))
    g = int(g_ * 255 * alpha + BG_COLOR[1] * (1 - alpha))
    b = int(b_ * 255 * alpha + BG_COLOR[2] * (1 - alpha))
    return (r, g, b)


def _geom_rings(geom):
    """Yield coordinate lists for each exterior ring in a Polygon/MultiPolygon."""
    if geom.geom_type == "Polygon":
        yield list(geom.exterior.coords)
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            yield list(poly.exterior.coords)


def _to_px(x, y, cx, cy, half_m):
    """Convert EPSG:3857 coordinates to pixel coords for this map."""
    px = (x - (cx - half_m)) / (2 * half_m) * IMG_SIZE
    py = (1.0 - (y - (cy - half_m)) / (2 * half_m)) * IMG_SIZE
    return px, py


def _nice_interval(span):
    """Return a 'round' tick interval (in degrees) that gives 2–5 ticks."""
    for iv in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        if 2 <= span / iv <= 6:
            return iv
    return 0.005


def _draw_ticks(draw, cx, cy, half_m, to_wgs84, from_wgs84):
    """Draw lat/lon tick marks and decimal-degree labels on map edges."""
    # Viewport corners in WGS84
    lon_min, lat_min = to_wgs84.transform(cx - half_m, cy - half_m)
    lon_max, lat_max = to_wgs84.transform(cx + half_m, cy + half_m)

    lat_iv = _nice_interval(lat_max - lat_min)
    lon_iv = _nice_interval(lon_max - lon_min)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
    except Exception:
        font = ImageFont.load_default()

    # Latitude ticks — left edge
    lat = np.ceil(lat_min / lat_iv) * lat_iv
    while lat < lat_max:
        _, y3857 = from_wgs84.transform(lon_min, lat)
        _, py = _to_px(cx - half_m, y3857, cx, cy, half_m)
        py = int(round(py))
        if 4 <= py <= IMG_SIZE - 4:
            draw.line([(0, py), (5, py)], fill=TICK_COLOR, width=1)
            draw.text((7, py - 5), f"{lat:.3f}\u00b0", fill=LABEL_COLOR, font=font)
        lat = round(lat + lat_iv, 8)

    # Longitude ticks — bottom edge
    lon = np.ceil(lon_min / lon_iv) * lon_iv
    while lon < lon_max:
        x3857, _ = from_wgs84.transform(lon, lat_min)
        px, _ = _to_px(x3857, cy - half_m, cx, cy, half_m)
        px = int(round(px))
        if 4 <= px <= IMG_SIZE - 4:
            draw.line([(px, IMG_SIZE - 5), (px, IMG_SIZE)], fill=TICK_COLOR, width=1)
            draw.text((px - 12, IMG_SIZE - 16), f"{lon:.3f}\u00b0", fill=LABEL_COLOR, font=font)
        lon = round(lon + lon_iv, 8)


def _render_map_card(cx, cy, target_indices, all_3857, diff_col, vmax, to_wgs84, from_wgs84, da_3857=None):
    """Return a 300×300 PIL Image centered on (cx, cy), structures colored by error.

    target_indices: set of indices to highlight in yellow.
    If da_3857 is provided the agreement raster is rendered at 50% alpha underneath.
    """
    half_m = HALF_M_S
    target_set = set(target_indices)

    # --- Raster background at 50% alpha ---
    if da_3857 is not None:
        try:
            cropped = da_3857.rio.clip_box(cx - half_m, cy - half_m, cx + half_m, cy + half_m)
            vals = cropped.values.astype(float)
            if vals.ndim == 3:
                vals = vals[0]
            h, w = vals.shape
            if h > 0 and w > 0:
                row_idx = np.linspace(0, h - 1, IMG_SIZE).astype(int)
                col_idx = np.linspace(0, w - 1, IMG_SIZE).astype(int)
                vals = vals[np.ix_(row_idx, col_idx)]
                normed = (np.clip(vals / vmax, -1.0, 1.0) + 1.0) / 2.0
                rgba = plt.cm.RdBu(normed)
                af = np.where(np.isnan(vals), 0.0, 0.50).astype(np.float32)
                bg = np.array(BG_COLOR, dtype=np.float32)
                out = np.empty((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                for c in range(3):
                    out[:, :, c] = np.clip(rgba[:, :, c] * 255 * af + bg[c] * (1 - af), 0, 255)
                img = Image.fromarray(out, mode="RGB")
            else:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BG_COLOR)
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BG_COLOR)
    else:
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BG_COLOR)

    draw = ImageDraw.Draw(img)

    viewport = shapely_box(cx - half_m, cy - half_m, cx + half_m, cy + half_m)
    candidates_idx = list(all_3857.sindex.intersection(viewport.bounds))
    candidates = all_3857.iloc[candidates_idx]

    # Draw non-target features first, targets on top
    non_targets = candidates[~candidates.index.isin(target_set)]
    for _, row in non_targets.iterrows():
        val = row[diff_col]
        if pd.isna(val):
            continue
        color = _val_to_rgb(val, vmax, alpha=1.0)
        for coords in _geom_rings(row.geometry):
            px_coords = [(int(_to_px(x, y, cx, cy, half_m)[0]), int(_to_px(x, y, cx, cy, half_m)[1]))
                         for x, y in coords]
            if len(px_coords) >= 3:
                draw.polygon(px_coords, fill=color, outline=(40, 44, 56))

    # Draw all target features with yellow outline
    for tidx in target_set:
        if tidx not in all_3857.index:
            continue
        row = all_3857.loc[tidx]
        val = row[diff_col]
        fill_color = _val_to_rgb(val, vmax, alpha=1.0) if not pd.isna(val) else BG_COLOR
        for coords in _geom_rings(row.geometry):
            px_coords = [(int(_to_px(x, y, cx, cy, half_m)[0]), int(_to_px(x, y, cx, cy, half_m)[1]))
                         for x, y in coords]
            if len(px_coords) >= 3:
                draw.polygon(px_coords, fill=fill_color)
                draw.polygon(px_coords, outline=HIGHLIGHT_YELLOW)

    _draw_ticks(draw, cx, cy, half_m, to_wgs84, from_wgs84)
    return img


def _render_catchment_card(cx, cy, target_indices, all_3857, da_3857, vmax, to_wgs84, from_wgs84):
    """Return a 300×300 PIL Image: agreement raster background + catchment outlines.

    target_indices: set of indices to highlight in yellow.
    """
    half_m = HALF_M_C
    target_set = set(target_indices)

    # --- Raster background ---
    try:
        cropped = da_3857.rio.clip_box(cx - half_m, cy - half_m, cx + half_m, cy + half_m)
        vals = cropped.values.astype(float)
        if vals.ndim == 3:
            vals = vals[0]
        h, w = vals.shape
        if h > 0 and w > 0:
            row_idx = np.linspace(0, h - 1, IMG_SIZE).astype(int)
            col_idx = np.linspace(0, w - 1, IMG_SIZE).astype(int)
            vals = vals[np.ix_(row_idx, col_idx)]
            normed = (np.clip(vals / vmax, -1.0, 1.0) + 1.0) / 2.0
            rgba = plt.cm.RdBu(normed)
            af = np.where(np.isnan(vals), 0.0, 0.82).astype(np.float32)
            bg = np.array(BG_COLOR, dtype=np.float32)
            out = np.empty((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            for c in range(3):
                out[:, :, c] = np.clip(rgba[:, :, c] * 255 * af + bg[c] * (1 - af), 0, 255)
            img = Image.fromarray(out, mode="RGB")
        else:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BG_COLOR)
    except Exception:
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BG_COLOR)

    draw = ImageDraw.Draw(img)

    # --- Catchment outlines (no fill) ---
    viewport = shapely_box(cx - half_m, cy - half_m, cx + half_m, cy + half_m)
    candidates_idx = list(all_3857.sindex.intersection(viewport.bounds))
    candidates = all_3857.iloc[candidates_idx]

    for idx, row in candidates.iterrows():
        is_target = idx in target_set
        color = HIGHLIGHT_YELLOW if is_target else (140, 145, 165)
        width = 2 if is_target else 1
        for coords in _geom_rings(row.geometry):
            px_coords = [(int(_to_px(x, y, cx, cy, half_m)[0]), int(_to_px(x, y, cx, cy, half_m)[1]))
                         for x, y in coords]
            if len(px_coords) >= 3:
                draw.polygon(px_coords, outline=color, width=width)

    _draw_ticks(draw, cx, cy, half_m, to_wgs84, from_wgs84)
    return img


# ---------------------------------------------------------------------------
# Locator inset
# ---------------------------------------------------------------------------

def _make_inset_base(da_3857):
    """Build an INSET_SIZE×INSET_SIZE binary thumbnail of the full agreement raster.

    Returns (PIL.Image, (xmin, ymin, xmax, ymax)) in EPSG:3857.
    White pixels = valid data; black = nodata.
    """
    xmin, ymin, xmax, ymax = da_3857.rio.bounds()
    vals = da_3857.values
    if vals.ndim == 3:
        vals = vals[0]
    h, w = vals.shape
    ri = np.linspace(0, h - 1, INSET_SIZE).astype(int)
    ci = np.linspace(0, w - 1, INSET_SIZE).astype(int)
    thumb = vals[np.ix_(ri, ci)]
    out = np.full((INSET_SIZE, INSET_SIZE, 3), 68, dtype=np.uint8)   # charcoal bg
    out[~np.isnan(thumb)] = 255                                        # white data
    return Image.fromarray(out, mode="RGB"), (xmin, ymin, xmax, ymax)


def _paste_inset(card_img, inset_base, da_bounds, cx, cy, half_m, pin=False):
    """Paste a locator inset onto the bottom-right corner of card_img (in-place).

    pin=True: draw an orange map-pin marker at (cx, cy) — used for structures
              whose viewport is too small to show as a box.
    pin=False: draw a semi-transparent orange rectangle for the viewport.
    """
    xmin, ymin, xmax, ymax = da_bounds
    inset = inset_base.copy().convert("RGBA")
    overlay = Image.new("RGBA", inset.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def _to_ip(x, y):
        """3857 coords → inset pixel coords (y-flipped)."""
        px = (x - xmin) / (xmax - xmin) * INSET_SIZE
        py = (1.0 - (y - ymin) / (ymax - ymin)) * INSET_SIZE
        return px, py

    if pin:
        ipx, ipy = _to_ip(cx, cy)
        ipx, ipy = float(ipx), float(ipy)
        r = 4          # circle radius
        tip_drop = 7   # how far below circle center the tip falls
        # Circle (head of pin)
        draw.ellipse([ipx - r, ipy - r - tip_drop,
                      ipx + r, ipy + r - tip_drop],
                     fill=(*INSET_AOI_COLOR, 255))
        # Downward-pointing triangle (body of pin)
        draw.polygon([(ipx - r + 1, ipy - tip_drop + r - 1),
                      (ipx + r - 1, ipy - tip_drop + r - 1),
                      (ipx,         ipy + r)],
                     fill=(*INSET_AOI_COLOR, 255))
    else:
        ix0, iy1 = _to_ip(cx - half_m, cy - half_m)
        ix1, iy0 = _to_ip(cx + half_m, cy + half_m)
        ix0, ix1 = sorted([max(0, min(INSET_SIZE - 1, ix0)), max(0, min(INSET_SIZE - 1, ix1))])
        iy0, iy1 = sorted([max(0, min(INSET_SIZE - 1, iy0)), max(0, min(INSET_SIZE - 1, iy1))])
        draw.rectangle([ix0, iy0, ix1, iy1], fill=(*INSET_AOI_COLOR, 60),
                       outline=(*INSET_AOI_COLOR, 255), width=2)

    inset = Image.alpha_composite(inset, overlay).convert("RGB")

    # Thin dark border around inset
    bordered = Image.new("RGB", (INSET_SIZE + 2, INSET_SIZE + 2), (40, 44, 56))
    bordered.paste(inset, (1, 1))

    x_pos = IMG_SIZE - (INSET_SIZE + 2) - INSET_MARGIN
    y_pos = IMG_SIZE - (INSET_SIZE + 2) - INSET_MARGIN
    card_img.paste(bordered, (x_pos, y_pos))


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cluster_features(worst_3857, half_m):
    """Greedy proximity grouping — O(n × groups).

    Features are processed worst-first. Each feature is added to the first
    existing group whose center is within half_m of the feature's centroid.
    Otherwise a new group is started centered on that feature.

    Returns list of (cx, cy, [index, ...]) sorted by descending |error| of
    the anchor (first/worst) feature.
    """
    centroids = [(row.geometry.centroid.x, row.geometry.centroid.y)
                 for _, row in worst_3857.iterrows()]
    indices = list(worst_3857.index)

    groups = []  # [(cx, cy, [idx, ...])]
    for (fx, fy), idx in zip(centroids, indices):
        placed = False
        for g in groups:
            gcx, gcy = g[0], g[1]
            if (fx - gcx) ** 2 + (fy - gcy) ** 2 <= half_m ** 2:
                g[2].append(idx)
                placed = True
                break
        if not placed:
            groups.append([fx, fy, [idx]])

    return groups


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_structures(path, top_pct, units):
    """Load structures GeoPackage; return (all_gdf, worst_gdf)."""
    print("Loading structures...")
    t0 = time.time()
    gdf = gpd.read_file(path)
    if units == "feet":
        gdf["mean_depth_diff"] = gdf["mean_depth_diff"] * M_TO_FT
    threshold = float(np.percentile(gdf["mean_depth_diff"].abs().dropna(), 100 - top_pct))
    worst = gdf[gdf["mean_depth_diff"].abs() >= threshold].copy()
    worst = worst.sort_values("mean_depth_diff", key=abs, ascending=False)
    print(f"  {len(worst)}/{len(gdf)} structures in top {top_pct}% "
          f"(threshold: ±{threshold:.2f}) [{time.time()-t0:.1f}s]")
    return gdf, worst


def load_catchments(catchments_path, agreement_path, top_pct, units):
    """Load catchments, run exactextract for per-catchment mean_diff; return (all_gdf, worst_gdf)."""
    import rioxarray as rxr
    import rasterio
    from exactextract import exact_extract
    from shapely.geometry import box as shapely_box

    print("Loading catchments and computing zonal stats...")
    t0 = time.time()

    da = rxr.open_rasterio(agreement_path, masked=True).squeeze().compute()
    if units == "feet":
        da = da * M_TO_FT

    bounds = da.rio.bounds()
    domain_mask = gpd.GeoDataFrame(geometry=[shapely_box(*bounds)], crs=da.rio.crs)
    catchments = gpd.read_file(catchments_path, mask=domain_mask)
    print(f"  {len(catchments)} catchments in domain")

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        da.rio.to_raster(tmp_path)
        results = exact_extract(
            rasterio.open(tmp_path),
            catchments,
            ["mean", "count"],
            output="pandas",
        )
    finally:
        os.unlink(tmp_path)

    catchments["mean_diff"] = results["mean"]
    catchments["pixel_count"] = results["count"]
    catchments = catchments[catchments["pixel_count"] > 0].copy()

    threshold = float(np.percentile(catchments["mean_diff"].abs().dropna(), 100 - top_pct))
    worst = catchments[catchments["mean_diff"].abs() >= threshold].copy()
    worst = worst.sort_values("mean_diff", key=abs, ascending=False)
    print(f"  {len(worst)}/{len(catchments)} catchments in top {top_pct}% "
          f"(threshold: ±{threshold:.2f}) [{time.time()-t0:.1f}s]")
    return catchments, worst, da


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html(structures_all, structures_worst,
               catchments_all, catchments_worst, da,
               top_pct, units, title, scenario="100yr"):

    unit_label = "ft" if units == "feet" else "m"
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    from_wgs84 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Reproject to EPSG:3857 once
    print("Reprojecting to EPSG:3857...")
    t0 = time.time()
    s_3857 = structures_all.to_crs("EPSG:3857") if structures_all is not None else None
    c_3857 = catchments_all.to_crs("EPSG:3857") if catchments_all is not None else None
    da_3857 = da.rio.reproject("EPSG:3857") if da is not None else None
    inset_base, da_bounds = _make_inset_base(da_3857) if da_3857 is not None else (None, None)
    print(f"  Done [{time.time()-t0:.1f}s]")

    # Vmaxes (computed on full datasets, not just worst)
    s_vmax = float(np.percentile(structures_all["mean_depth_diff"].abs().dropna(), 95)) \
        if structures_all is not None else 1.0
    c_vmax = float(np.percentile(catchments_all["mean_diff"].abs().dropna(), 95)) \
        if catchments_all is not None else 1.0

    # --- Structure cards ---
    structure_cards_html = ""
    n_structures = 0
    scenario_set = set()
    if structures_worst is not None and len(structures_worst) > 0:
        s_worst_3857 = s_3857.loc[structures_worst.index]
        clusters = _cluster_features(s_worst_3857, HALF_M_S)
        n = len(clusters)
        n_raw = len(structures_worst)
        print(f"Rendering {n} structure maps ({n_raw} features → {n} clusters)...")
        t0 = time.time()
        for i, (cx, cy, idxs) in enumerate(clusters):
            img = _render_map_card(cx, cy, idxs, s_3857, "mean_depth_diff",
                                   s_vmax, to_wgs84, from_wgs84, da_3857=da_3857)
            if inset_base is not None:
                _paste_inset(img, inset_base, da_bounds, cx, cy, HALF_M_S, pin=True)
            b64 = _pil_to_b64(img)

            # Metadata is derived from the worst (anchor) feature
            anchor_idx = idxs[0]
            val = float(structures_all.loc[anchor_idx, "mean_depth_diff"])
            sign = "+" if val >= 0 else ""
            direction = "over" if val >= 0 else "under"

            lon, lat = to_wgs84.transform(cx, cy)
            gmaps = f"https://maps.google.com/?q={lat:.5f},{lon:.5f}&z=17&t=k"

            cluster_badge = (f'<span class="cluster-badge">&#xd7;{len(idxs)}</span>'
                             if len(idxs) > 1 else "")
            scenario_set.add(scenario)

            structure_cards_html += f"""
      <div class="map-card" data-layer="structure" data-direction="{direction}" data-scenario="{scenario}">
        <img src="data:image/jpeg;base64,{b64}" width="{IMG_SIZE}" height="{IMG_SIZE}">
        <div class="meta">
          {cluster_badge}
          <span class="val {direction}">{sign}{val:.1f} {unit_label}</span>
          <a href="{gmaps}" target="_blank" class="gmaps-link">&#x1F4CD; Maps</a>
        </div>
        <div class="copy-bar">
          <button class="copy-btn" onclick="copyText('12090301', this)">12090301</button>
        </div>
      </div>"""
            n_structures += 1
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{n} structure maps [{time.time()-t0:.1f}s]")
        print(f"  Structure maps done [{time.time()-t0:.1f}s]")

    # --- Catchment cards ---
    catchment_cards_html = ""
    n_catchments = 0
    so_set = set()
    if catchments_worst is not None and len(catchments_worst) > 0:
        c_worst_3857 = c_3857.loc[catchments_worst.index]
        clusters = _cluster_features(c_worst_3857, HALF_M_C)
        n = len(clusters)
        n_raw = len(catchments_worst)
        print(f"Rendering {n} catchment maps ({n_raw} features → {n} clusters)...")
        t0 = time.time()
        for i, (cx, cy, idxs) in enumerate(clusters):
            img = _render_catchment_card(cx, cy, idxs, c_3857, da_3857,
                                         c_vmax, to_wgs84, from_wgs84)
            if inset_base is not None:
                _paste_inset(img, inset_base, da_bounds, cx, cy, HALF_M_C)
            b64 = _pil_to_b64(img)

            # Metadata derived from anchor (worst) feature
            anchor_idx = idxs[0]
            val = float(catchments_all.loc[anchor_idx, "mean_diff"])
            anchor_so = int(catchments_all.loc[anchor_idx, "stream_order"])
            so_set.add(anchor_so)
            sign = "+" if val >= 0 else ""
            direction = "over" if val >= 0 else "under"

            lon, lat = to_wgs84.transform(cx, cy)
            gmaps = f"https://maps.google.com/?q={lat:.5f},{lon:.5f}&z=15&t=k"

            # Collect all unique SOs in cluster for display; anchor SO used for filtering
            cluster_sos = sorted({int(catchments_all.loc[ii, "stream_order"])
                                   for ii in idxs if ii in catchments_all.index})
            so_label = " ".join(f"SO{s}" for s in cluster_sos)
            cluster_badge = (f'<span class="cluster-badge">&#xd7;{len(idxs)}</span>'
                             if len(idxs) > 1 else "")
            anchor_fid = int(catchments_all.loc[anchor_idx, "feature_id"])
            scenario_set.add(scenario)

            catchment_cards_html += f"""
      <div class="map-card" data-layer="catchment" data-so="{anchor_so}" data-direction="{direction}" data-scenario="{scenario}">
        <img src="data:image/jpeg;base64,{b64}" width="{IMG_SIZE}" height="{IMG_SIZE}">
        <div class="meta">
          {cluster_badge}
          <span class="so-badge">{so_label}</span>
          <span class="val {direction}">{sign}{val:.1f} {unit_label}</span>
          <a href="{gmaps}" target="_blank" class="gmaps-link">&#x1F4CD; Maps</a>
        </div>
        <div class="copy-bar">
          <button class="copy-btn" onclick="copyText('12090301', this)">12090301</button>
          <button class="copy-btn" onclick="copyText('{anchor_fid}', this)">{anchor_fid}</button>
        </div>
      </div>"""
            n_catchments += 1
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{n} catchment maps [{time.time()-t0:.1f}s]")
        print(f"  Catchment maps done [{time.time()-t0:.1f}s]")

    so_buttons = "".join(
        f'<button class="so-btn" data-so="{so}">SO{so}</button>'
        for so in sorted(so_set)
    )
    scenario_buttons = "".join(
        f'<button class="scenario-btn" data-scenario="{sc}">{sc}</button>'
        for sc in sorted(scenario_set)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: #0d0f14;
  color: #e0e0e0;
  font-family: 'Inter', 'Proxima Nova', -apple-system, BlinkMacSystemFont, sans-serif;
}}
.header {{
  padding: 20px 24px 14px;
  border-bottom: 1px solid #1e2128;
}}
.header h1 {{ font-size: 20px; font-weight: 600; color: #f0f0f0; margin-bottom: 4px; }}
.header .subtitle {{ font-size: 13px; color: #6a6d75; }}

.filter-bar {{
  position: sticky; top: 0; z-index: 100;
  background: #111318;
  border-bottom: 1px solid #1e2128;
  padding: 10px 24px;
  display: flex; align-items: center; gap: 24px; flex-wrap: wrap;
}}
.filter-group {{ display: flex; align-items: center; gap: 6px; }}
.filter-label {{
  font-size: 10px; color: #555870; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.06em; white-space: nowrap;
}}
button {{
  background: #1a1d26; border: 1px solid #252830; color: #9098b0;
  padding: 4px 10px; border-radius: 4px; font-size: 12px; cursor: pointer;
  transition: background 0.12s, color 0.12s, border-color 0.12s;
}}
button:hover {{ background: #222530; color: #c0c8e0; }}
button.active {{
  background: #252a40; border-color: #4a5090; color: #b0b8e0;
}}

.section {{ padding: 20px 24px 28px; }}
.section-header {{
  display: flex; align-items: baseline; gap: 12px; margin-bottom: 14px;
}}
.section-header h2 {{ font-size: 15px; font-weight: 600; color: #d8d8e8; }}
.count {{ font-size: 12px; color: #555870; }}

.grid {{
  display: grid;
  grid-template-columns: repeat(5, {IMG_SIZE}px);
  gap: 10px;
}}
.map-card {{
  background: #111318;
  border: 1px solid #1a1d26;
  border-radius: 5px;
  overflow: hidden;
}}
.map-card img {{ display: block; }}
.meta {{
  padding: 6px 10px;
  display: flex; align-items: center; gap: 8px;
  font-size: 12px;
}}
.so-badge {{
  background: #1a1d26; border: 1px solid #252830;
  border-radius: 3px; padding: 1px 6px;
  font-size: 10px; color: #7080a0; font-weight: 500;
}}
.cluster-badge {{
  background: #1e2230; border: 1px solid #3a4060;
  border-radius: 3px; padding: 1px 6px;
  font-size: 10px; color: #8090c0; font-weight: 600;
}}
.val.over  {{ color: #5888d0; }}
.val.under {{ color: #c05060; }}
.gmaps-link {{
  margin-left: auto; color: #505870; text-decoration: none; font-size: 11px;
}}
.gmaps-link:hover {{ color: #9098b8; }}

.copy-bar {{
  display: flex; justify-content: flex-end; gap: 4px; padding: 2px 6px 5px;
}}
.copy-btn {{
  font-size: 11px; padding: 2px 8px; border-radius: 4px;
  border: 1px solid #252830; cursor: pointer;
  background: #181b24; color: #7080a0;
  transition: background 0.1s, color 0.1s;
}}
.copy-btn:hover {{ background: #222530; color: #b0b8d0; }}
.copy-btn.copied {{ background: #1a3a28; color: #60d090; border-color: #2a5a40; }}

.divider {{ border: none; border-top: 1px solid #1a1d26; margin: 0 24px; }}
</style>
</head>
<body>

<div class="header">
  <h1>{title}</h1>
  <div class="subtitle">Top {top_pct}% worst by |mean depth diff|&nbsp;&nbsp;&#x2022;&nbsp;&nbsp;{n_structures} structure maps&nbsp;&nbsp;&#x2022;&nbsp;&nbsp;{n_catchments} catchment maps</div>
</div>

<div class="filter-bar">
  <div class="filter-group">
    <span class="filter-label">Scenario</span>
    <button class="scenario-btn active" data-scenario="all">All</button>
    {scenario_buttons}
  </div>
  <div class="filter-group">
    <span class="filter-label">Direction</span>
    <button class="dir-btn active" data-dir="all">All</button>
    <button class="dir-btn" data-dir="over">Over &#x25B2;</button>
    <button class="dir-btn" data-dir="under">Under &#x25BC;</button>
  </div>
  <div class="filter-group">
    <span class="filter-label">Stream Order</span>
    <button class="so-btn active" data-so="all">All</button>
    {so_buttons}
  </div>
  <div class="filter-group">
    <span class="filter-label">Jump to</span>
    <button onclick="document.getElementById('sec-catchments').scrollIntoView({{behavior:'smooth'}})">Catchments</button>
    <button onclick="document.getElementById('sec-structures').scrollIntoView({{behavior:'smooth'}})">Structures</button>
  </div>
</div>

<section class="section" id="sec-catchments">
  <div class="section-header">
    <h2>Worst Catchments</h2>
    <span class="count" id="count-catchments">{n_catchments} shown</span>
  </div>
  <div class="grid" id="grid-catchments">
{catchment_cards_html}
  </div>
</section>

<hr class="divider">

<section class="section" id="sec-structures">
  <div class="section-header">
    <h2>Worst Structures</h2>
    <span class="count" id="count-structures">{n_structures} shown</span>
  </div>
  <div class="grid" id="grid-structures">
{structure_cards_html}
  </div>
</section>

<script>
(function () {{
  let activeDir      = 'all';
  let activeSO       = 'all';
  let activeScenario = 'all';

  function applyFilters() {{
    let cCount = 0, sCount = 0;
    document.querySelectorAll('.map-card').forEach(card => {{
      const dir      = card.dataset.direction;
      const so       = card.dataset.so;
      const layer    = card.dataset.layer;
      const scenario = card.dataset.scenario;
      const dirOk      = activeDir      === 'all' || activeDir      === dir;
      const soOk       = layer !== 'catchment' || activeSO === 'all' || activeSO === so;
      const scenarioOk = activeScenario === 'all' || activeScenario === scenario;
      const show  = dirOk && soOk && scenarioOk;
      card.style.display = show ? '' : 'none';
      if (show) {{ layer === 'catchment' ? cCount++ : sCount++; }}
    }});
    document.getElementById('count-catchments').textContent = cCount + ' shown';
    document.getElementById('count-structures').textContent = sCount + ' shown';
  }}

  document.querySelectorAll('.dir-btn').forEach(btn =>
    btn.addEventListener('click', () => {{
      activeDir = btn.dataset.dir;
      document.querySelectorAll('.dir-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      applyFilters();
    }})
  );

  document.querySelectorAll('.so-btn').forEach(btn =>
    btn.addEventListener('click', () => {{
      activeSO = btn.dataset.so;
      document.querySelectorAll('.so-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      applyFilters();
    }})
  );

  document.querySelectorAll('.scenario-btn').forEach(btn =>
    btn.addEventListener('click', () => {{
      activeScenario = btn.dataset.scenario;
      document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      applyFilters();
    }})
  );
}})();

function copyText(text, btn) {{
  navigator.clipboard.writeText(String(text)).then(() => {{
    const orig = btn.textContent;
    btn.textContent = '✓';
    btn.classList.add('copied');
    setTimeout(() => {{ btn.textContent = orig; btn.classList.remove('copied'); }}, 1200);
  }});
}}
</script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Spatial discrepancy report")
    parser.add_argument("--structures-gpkg", required=True)
    parser.add_argument("--catchments", required=True)
    parser.add_argument("--agreement-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-pct", type=float, default=5.0)
    parser.add_argument("--units", default="feet", choices=["feet", "meters"])
    parser.add_argument("--scenario", default="100yr")
    parser.add_argument("--title", default="Triage")
    args = parser.parse_args()

    t_total = time.time()

    structures_all, structures_worst = load_structures(
        args.structures_gpkg, args.top_pct, args.units
    )
    catchments_all, catchments_worst, da = load_catchments(
        args.catchments, args.agreement_map, args.top_pct, args.units
    )

    html = build_html(
        structures_all, structures_worst,
        catchments_all, catchments_worst, da,
        args.top_pct, args.units, args.title, args.scenario,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"Report saved to {args.output} ({size_kb:.0f} KB) "
          f"[{time.time()-t_total:.1f}s total]")


if __name__ == "__main__":
    main()
