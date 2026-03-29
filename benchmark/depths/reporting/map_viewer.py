"""
Generate an interactive Leaflet web map for triage review.

Converts the agreement raster to a Cloud-Optimized GeoTIFF, exports vectors
to GeoJSON, and creates a self-contained map.html that loads everything via
a local HTTP server.

Example usage:
    python map_viewer.py \
        --structures-gpkg outputs/structures.gpkg \
        --catchments inputs/catchments.fgb \
        --agreement-map outputs/agreement_map.tif \
        --output-dir outputs/map \
        --units feet
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import signal
import socketserver
import sys
import threading
import time
import webbrowser

import geopandas as gpd
import numpy as np
import rasterio
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

M_TO_FT = 3.28084


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _convert_to_cog(src_path, dst_path, units):
    """Reproject agreement map to EPSG:3857 COG with overviews."""
    from rasterio.shutil import copy as rio_copy
    import tempfile

    print("Converting agreement map to COG (EPSG:3857 with overviews)...")
    t0 = time.time()

    # First reproject to 3857 into a temp file, then COG-ify
    with rasterio.open(src_path) as src:
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        dst_crs = "EPSG:3857"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            driver="GTiff",
        )

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with rasterio.open(tmp_path, "w", **profile) as dst:
                data = src.read(1)
                if units == "feet":
                    valid = ~np.isnan(data) if np.issubdtype(data.dtype, np.floating) else np.ones_like(data, dtype=bool)
                    data = np.where(valid, data * M_TO_FT, data)
                reproject(
                    source=data,
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=src.nodata,
                )

            # COG translate
            output_profile = cog_profiles.get("deflate")
            cog_translate(
                tmp_path,
                dst_path,
                output_profile,
                overview_level=5,
                overview_resampling="average",
                quiet=True,
            )
        finally:
            os.unlink(tmp_path)

    print(f"  COG saved: {dst_path} [{time.time()-t0:.1f}s]")


def _export_geojson(gdf, path, cols_keep, units_col=None, units="feet"):
    """Export a GeoDataFrame to GeoJSON in EPSG:4326, keeping only needed columns."""
    t0 = time.time()
    out = gdf.to_crs("EPSG:4326")[cols_keep + ["geometry"]].copy()
    if units_col and units == "feet":
        out[units_col] = out[units_col] * M_TO_FT
    out.to_file(path, driver="GeoJSON")
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  {path} ({size_mb:.1f} MB) [{time.time()-t0:.1f}s]")


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def _build_map_html(cog_filename, structures_filename, catchments_filename,
                    s_vmax, c_vmax, center_lat, center_lon, worst_pct_threshold,
                    nwm_flows_filename=None):
    """Return the Leaflet map HTML as a string."""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Triage Map Viewer</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d0f14; }}
  #map {{ width: 100vw; height: 100vh; }}
  .leaflet-popup-content-wrapper {{
    background: #1a1d26; color: #e0e0e0; border-radius: 6px;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 12px;
  }}
  .leaflet-popup-tip {{ background: #1a1d26; }}
  .popup-title {{ font-weight: 600; font-size: 13px; margin-bottom: 4px; color: #f0f0f0; }}
  .popup-row {{ display: flex; justify-content: space-between; gap: 12px; padding: 1px 0; }}
  .popup-label {{ color: #6a6d75; }}
  .popup-val {{ font-weight: 500; }}
  .popup-val.over {{ color: #5888d0; }}
  .popup-val.under {{ color: #c05060; }}
  .info-panel {{
    position: fixed; top: 10px; left: 55px; z-index: 1000;
    background: rgba(17,19,24,0.92); border: 1px solid #252830; border-radius: 6px;
    padding: 10px 14px; color: #9098b0; font-size: 11px;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    pointer-events: none;
  }}
  .legend {{
    position: fixed; bottom: 24px; left: 12px; z-index: 1000;
    background: rgba(17,19,24,0.92); border: 1px solid #252830; border-radius: 6px;
    padding: 10px 14px; color: #9098b0; font-size: 11px;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  }}
  .legend-title {{ font-weight: 600; color: #d8d8e8; margin-bottom: 6px; }}
  .legend-bar {{
    width: 180px; height: 12px; border-radius: 2px;
    background: linear-gradient(to right, #b2182b, #d6604d, #f4a582, #fddbc7, #f7f7f7, #d1e5f0, #92c5de, #4393c3, #2166ac);
  }}
  .legend-labels {{ display: flex; justify-content: space-between; font-size: 10px; margin-top: 2px; }}
</style>
</head>
<body>
<div id="map"></div>
<div class="info-panel" id="info">Triage Map Viewer</div>
<div class="legend">
  <div class="legend-title">Mean Depth Difference</div>
  <div class="legend-bar"></div>
  <div class="legend-labels"><span>Under (candidate &lt; benchmark)</span><span>Over</span></div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/georaster@1.6.0/dist/georaster.browser.bundle.min.js"></script>
<script src="https://unpkg.com/georaster-layer-for-leaflet@3.10.0/dist/georaster-layer-for-leaflet.min.js"></script>

<script>
(function() {{
  // --- Parse URL params for deep-linking ---
  const params = new URLSearchParams(window.location.search);
  const initLat  = parseFloat(params.get('lat'))  || {center_lat};
  const initLon  = parseFloat(params.get('lon'))  || {center_lon};
  const initZoom = parseInt(params.get('zoom'))    || 12;

  const map = L.map('map', {{
    center: [initLat, initLon],
    zoom: initZoom,
    zoomControl: true,
  }});

  // --- Basemaps ---
  const darkGray = L.tileLayer(
    'https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}',
    {{ attribution: 'Esri', maxZoom: 16 }}
  );

  const osm = L.tileLayer(
    'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
    {{ attribution: '&copy; OpenStreetMap contributors', maxZoom: 19 }}
  );

  const imagery = L.tileLayer(
    'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
    {{ attribution: 'Esri', maxZoom: 19 }}
  );

  const imageryLabels = L.layerGroup([
    L.tileLayer(
      'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{ attribution: 'Esri', maxZoom: 19 }}
    ),
    L.tileLayer(
      'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{ maxZoom: 19, pane: 'overlayPane' }}
    ),
  ]).addTo(map);

  const baseMaps = {{
    "Dark Gray": darkGray,
    "OpenStreetMap": osm,
    "Satellite": imagery,
    "Satellite + Labels": imageryLabels,
  }};

  // --- RdBu colormap (matching report) ---
  // Attempt to replicate matplotlib RdBu via manual stops
  const RdBu_stops = [
    [0.0,  [103, 0, 31]],
    [0.1,  [178, 24, 43]],
    [0.2,  [214, 96, 77]],
    [0.3,  [244, 165, 130]],
    [0.4,  [253, 219, 199]],
    [0.5,  [247, 247, 247]],
    [0.6,  [209, 229, 240]],
    [0.7,  [146, 197, 222]],
    [0.8,  [67, 147, 195]],
    [0.9,  [33, 102, 172]],
    [1.0,  [5, 48, 97]],
  ];

  function rdbuColor(val, vmax) {{
    const normed = (Math.max(-1, Math.min(1, val / vmax)) + 1) / 2;
    // Find the two stops bracketing normed
    let lo = RdBu_stops[0], hi = RdBu_stops[RdBu_stops.length - 1];
    for (let i = 0; i < RdBu_stops.length - 1; i++) {{
      if (normed >= RdBu_stops[i][0] && normed <= RdBu_stops[i+1][0]) {{
        lo = RdBu_stops[i];
        hi = RdBu_stops[i+1];
        break;
      }}
    }}
    const t = (hi[0] === lo[0]) ? 0 : (normed - lo[0]) / (hi[0] - lo[0]);
    const r = Math.round(lo[1][0] + t * (hi[1][0] - lo[1][0]));
    const g = Math.round(lo[1][1] + t * (hi[1][1] - lo[1][1]));
    const b = Math.round(lo[1][2] + t * (hi[1][2] - lo[1][2]));
    return [r, g, b];
  }}

  // --- Agreement raster (COG loaded as ArrayBuffer) ---
  const S_VMAX = {s_vmax};
  const C_VMAX = {c_vmax};
  const WORST_THRESH = {worst_pct_threshold};

  const infoEl = document.getElementById('info');
  infoEl.textContent = 'Loading agreement raster...';

  fetch('{cog_filename}')
    .then(r => r.arrayBuffer())
    .then(buf => parseGeoraster(buf))
    .then(georaster => {{
      const rasterLayer = new GeoRasterLayer({{
        georaster: georaster,
        opacity: 0.65,
        pixelValuesToColorFn: function(vals) {{
          const v = vals[0];
          if (v === null || v === undefined || isNaN(v)) return null;
          if (v <= -9990 || v === georaster.noDataValue) return null;
          const rgb = rdbuColor(v, C_VMAX);
          return 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',0.82)';
        }},
        resolution: 256,
      }});
      rasterLayer.addTo(map);
      overlays["Agreement Raster"] = rasterLayer;
      updateLayerControl();
      infoEl.textContent = 'Triage Map Viewer';
    }})
    .catch(err => {{
      console.warn('Could not load raster:', err);
      infoEl.textContent = 'Raster failed to load: ' + err.message;
    }});

  // --- Overlay tracking ---
  const overlays = {{}};
  let layerControl = null;

  function updateLayerControl() {{
    if (layerControl) map.removeControl(layerControl);
    layerControl = L.control.layers(baseMaps, overlays).addTo(map);
  }}

  // --- Structures GeoJSON ---
  fetch('{structures_filename}')
    .then(r => r.json())
    .then(data => {{
      const layer = L.geoJSON(data, {{
        style: function(feature) {{
          const val = feature.properties.mean_depth_diff || 0;
          const rgb = rdbuColor(val, S_VMAX);
          return {{
            fillColor: 'rgb(' + rgb.join(',') + ')',
            fillOpacity: 0.85,
            color: '#282c38',
            weight: 1,
          }};
        }},
        onEachFeature: function(feature, layer) {{
          const p = feature.properties;
          const val = (p.mean_depth_diff || 0).toFixed(1);
          const sign = p.mean_depth_diff >= 0 ? '+' : '';
          const dir = p.mean_depth_diff >= 0 ? 'over' : 'under';
          layer.bindPopup(
            '<div class="popup-title">Structure</div>' +
            '<div class="popup-row"><span class="popup-label">Mean Diff</span>' +
            '<span class="popup-val ' + dir + '">' + sign + val + ' ft</span></div>'
          );
        }},
      }}).addTo(map);
      overlays["Structures"] = layer;
      updateLayerControl();
    }})
    .catch(err => console.warn('Could not load structures:', err));

  // --- Catchments GeoJSON (two passes: normal first, worst on top) ---
  fetch('{catchments_filename}')
    .then(r => r.json())
    .then(data => {{
      function catchPopup(feature, layer) {{
        const p = feature.properties;
        const val = p.mean_diff !== null ? p.mean_diff.toFixed(1) : 'N/A';
        const sign = (p.mean_diff || 0) >= 0 ? '+' : '';
        const dir = (p.mean_diff || 0) >= 0 ? 'over' : 'under';
        const so = p.stream_order || '?';
        const fid = p.feature_id || '';
        layer.bindPopup(
          '<div class="popup-title">Catchment ' + fid + '</div>' +
          '<div class="popup-row"><span class="popup-label">Stream Order</span>' +
          '<span class="popup-val">' + so + '</span></div>' +
          '<div class="popup-row"><span class="popup-label">Mean Diff</span>' +
          '<span class="popup-val ' + dir + '">' + sign + val + ' ft</span></div>'
        );
      }}

      // Normal catchments (rendered first, underneath)
      const normalLayer = L.geoJSON(data, {{
        filter: function(feature) {{
          const val = feature.properties.mean_diff;
          return !(val !== null && val !== undefined && Math.abs(val) >= WORST_THRESH);
        }},
        style: {{ fillOpacity: 0, color: '#262832', weight: 1 }},
        onEachFeature: catchPopup,
      }}).addTo(map);

      // Worst catchments (rendered on top)
      const worstLayer = L.geoJSON(data, {{
        filter: function(feature) {{
          const val = feature.properties.mean_diff;
          return val !== null && val !== undefined && Math.abs(val) >= WORST_THRESH;
        }},
        style: {{ fillOpacity: 0, color: '#ffd700', weight: 2.5 }},
        onEachFeature: catchPopup,
      }}).addTo(map);

      // Group them so the layer control toggles both together
      const group = L.layerGroup([normalLayer, worstLayer]).addTo(map);
      overlays["Catchments"] = group;
      updateLayerControl();
    }})
    .catch(err => console.warn('Could not load catchments:', err));

  // --- NWM Flows GeoJSON ---
  {f"""fetch('{nwm_flows_filename}')
    .then(r => r.json())
    .then(data => {{
      const layer = L.geoJSON(data, {{
        style: function(feature) {{
          const so = feature.properties.stream_order || 1;
          const weights = {{ 1: 0.6, 2: 0.9, 3: 1.3, 4: 1.8, 5: 2.5, 6: 3.5, 7: 4.5, 8: 6, 9: 8, 10: 10 }};
          return {{ color: '#2fb8a0', weight: weights[so] || 1, opacity: 0.75 }};
        }},
        onEachFeature: function(feature, layer) {{
          const p = feature.properties;
          const fid = p.feature_id || '';
          const so = p.stream_order || '?';
          layer.bindPopup(
            '<div class="popup-title">NWM Flowline ' + fid + '</div>' +
            '<div class="popup-row"><span class="popup-label">Stream Order</span>' +
            '<span class="popup-val">' + so + '</span></div>'
          );
        }},
      }});
      layer.addTo(map);
      overlays["NWM Flowlines"] = layer;
      updateLayerControl();
    }})
    .catch(err => console.warn('Could not load NWM flowlines:', err));""" if nwm_flows_filename else "// NWM flowlines not provided"}

  // Initial layer control (basemaps only until overlays load)
  updateLayerControl();

  // --- Deep-link marker (fades out after 4s) ---
  if (params.get('lat') && params.get('lon')) {{
    const marker = L.circleMarker([initLat, initLon], {{
      radius: 8, fillColor: '#ffd700', fillOpacity: 0.9,
      color: '#fff', weight: 2,
    }}).addTo(map);
    setTimeout(() => {{ map.removeLayer(marker); }}, 4000);
  }}

}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def _make_handler(serve_root):
    """Create an HTTP handler that supports PUT for saving reviews.json."""

    class TriageHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_PUT(self):
            # Only allow writing reviews.json
            path = self.translate_path(self.path)
            if not os.path.basename(path) == "reviews.json":
                self.send_error(403, "Only reviews.json can be written")
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            # Validate it's valid JSON
            try:
                json.loads(body)
            except Exception:
                self.send_error(400, "Invalid JSON")
                return
            with open(path, "wb") as f:
                f.write(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, PUT, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(200)
            self.end_headers()

    return TriageHandler


def _serve(output_dir, port):
    """Start a simple HTTP server serving the parent of output_dir.

    This way both the triage report (in the parent) and the map files
    (in output_dir) are accessible. The browser opens the map directly.
    Supports PUT requests to save reviews.json.
    """
    # Serve from the parent of the map output dir so that both
    # triage_report.html and map/ are accessible at the same origin.
    serve_root = os.path.dirname(os.path.abspath(output_dir))
    map_subdir = os.path.basename(os.path.abspath(output_dir))
    os.chdir(serve_root)

    handler = _make_handler(serve_root)

    with socketserver.TCPServer(("", port), handler) as httpd:
        map_url = f"http://localhost:{port}/{map_subdir}/map.html"
        report_url = f"http://localhost:{port}/triage_report.html"
        print(f"\n  Map:    {map_url}")
        print(f"  Report: {report_url}")
        print("  Press Ctrl+C to stop.\n")
        webbrowser.open(report_url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive Leaflet map viewer for triage")
    parser.add_argument("--structures-gpkg", required=True)
    parser.add_argument("--catchments", required=True)
    parser.add_argument("--agreement-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--units", default="feet", choices=["feet", "meters"])
    parser.add_argument("--port", type=int, default=0,
                        help="Port for local server (0 = auto-assign)")
    parser.add_argument("--nwm-flows", default=None,
                        help="Path to NWM flows FlatGeobuf")
    parser.add_argument("--no-serve", action="store_true",
                        help="Generate files only, don't start server")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t_total = time.time()

    # 1. Convert agreement map to COG
    cog_path = os.path.join(args.output_dir, "agreement_cog.tif")
    if not os.path.exists(cog_path):
        _convert_to_cog(args.agreement_map, cog_path, args.units)
    else:
        print(f"  COG already exists: {cog_path}")

    # 2. Export structures to GeoJSON
    print("Exporting structures...")
    structs = gpd.read_file(args.structures_gpkg)
    structs_path = os.path.join(args.output_dir, "structures.geojson")
    _export_geojson(structs, structs_path, ["mean_depth_diff"], "mean_depth_diff", args.units)

    # 3. Export catchments to GeoJSON
    print("Exporting catchments...")
    # Compute zonal stats for catchments (same as report_triage)
    import rioxarray as rxr
    from exactextract import exact_extract
    from shapely.geometry import box as shapely_box
    import tempfile

    da = rxr.open_rasterio(args.agreement_map, masked=True).squeeze().compute()
    if args.units == "feet":
        da = da * M_TO_FT
    bounds = da.rio.bounds()
    domain_mask = gpd.GeoDataFrame(geometry=[shapely_box(*bounds)], crs=da.rio.crs)
    catchments = gpd.read_file(args.catchments, mask=domain_mask)

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

    cols = ["mean_diff", "stream_order"]
    if "feature_id" in catchments.columns:
        cols.append("feature_id")
    catch_path = os.path.join(args.output_dir, "catchments.geojson")
    _export_geojson(catchments, catch_path, cols)

    # 4. Export NWM flows (clipped to domain)
    nwm_flows_filename = None
    if args.nwm_flows:
        print("Exporting NWM flows (clipped to domain)...")
        nwm = gpd.read_file(args.nwm_flows, mask=domain_mask)
        nwm_cols = [c for c in ["feature_id", "stream_order"] if c in nwm.columns]
        nwm_path = os.path.join(args.output_dir, "nwm_flows.geojson")
        _export_geojson(nwm, nwm_path, nwm_cols)
        nwm_flows_filename = "nwm_flows.geojson"

    # Compute vmax values for colormap scaling
    s_vmax = float(np.percentile(structs["mean_depth_diff"].abs().dropna(), 95))
    if args.units == "feet":
        s_vmax *= M_TO_FT
    c_vmax = float(np.percentile(catchments["mean_diff"].abs().dropna(), 95))

    # Worst-percentile threshold for catchment highlighting (top 5%)
    top_pct = 5.0
    worst_pct_threshold = float(np.percentile(catchments["mean_diff"].abs().dropna(), 100 - top_pct))

    # Center of data
    all_bounds = catchments.to_crs("EPSG:4326").total_bounds  # xmin, ymin, xmax, ymax
    center_lon = (all_bounds[0] + all_bounds[2]) / 2
    center_lat = (all_bounds[1] + all_bounds[3]) / 2

    # 4. Generate map HTML
    html = _build_map_html(
        "agreement_cog.tif", "structures.geojson", "catchments.geojson",
        s_vmax, c_vmax, center_lat, center_lon, worst_pct_threshold,
        nwm_flows_filename=nwm_flows_filename,
    )
    html_path = os.path.join(args.output_dir, "map.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\nMap files written to {args.output_dir}/ [{time.time()-t_total:.1f}s]")

    # 5. Serve
    if not args.no_serve:
        if args.port == 0:
            # Find a free port
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port = s.getsockname()[1]
        else:
            port = args.port
        _serve(args.output_dir, port)


if __name__ == "__main__":
    main()
