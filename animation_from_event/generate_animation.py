#!/usr/bin/env python3
"""
Generate Animation from Flood Inundation Maps
==============================================

Creates an animated video showing flood progression over time from FIM outputs.

Reads all FIM GeoTIFF files from the outputs/fims/ directory and creates a video
animation showing flood extent evolution with basemap and county boundary (optional).

All dependencies are pre-installed in the Docker container.

Usage:
    python generate_animation.py --config config.yaml

Configuration is loaded from config.yaml which specifies:
    - FIM directory location
    - Frame rate, resolution, and visualization parameters
    - Basemap and overlay settings
    - Geographic extent and county boundaries
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Patch
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
import imageio
from datetime import datetime
import warnings
import gc
import argparse
from config_utils import load_config, get_paths, get_animation_config
import fiona
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform
import contextily as ctx
import geopandas as gpd
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Load configuration
parser = argparse.ArgumentParser(description="Generate flood animation from FIM files")
parser.add_argument('--config', default='config.yaml', help="Path to config file")
parser.add_argument('--dpi', type=int, help="Override DPI")
parser.add_argument('--fps', type=float, help="Override FPS")
parser.add_argument('--start-time', help="Start time filter (YYYY-MM-DD HH:MM)")
parser.add_argument('--end-time', help="End time filter (YYYY-MM-DD HH:MM)")
parser.add_argument('--output', help="Override output video filename")
args = parser.parse_args()

config = load_config(args.config)
paths = get_paths(config)
anim_cfg = get_animation_config(config)

# Set paths from config
FIMS_DIR = paths['fims_dir']
OUTPUT_VIDEO = Path(args.output) if args.output else paths['output_video']

# Date range filter from command-line args (format: YYYY-MM-DD HH:MM)
# Convert to internal format YYYYMMDD_HHMM for filtering
def convert_time_format(time_str):
    """Convert 'YYYY-MM-DD HH:MM' to 'YYYYMMDD_HHMM'"""
    if not time_str:
        return None
    try:
        dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
        return dt.strftime('%Y%m%d_%H%M')
    except ValueError:
        # If already in YYYYMMDD_HHMM format, return as-is
        return time_str

START_TIME = convert_time_format(args.start_time) if args.start_time else None
END_TIME = convert_time_format(args.end_time) if args.end_time else None

# Visual settings from config
extent_cfg = anim_cfg['extent']
visual_cfg = anim_cfg['visual']
overlay_cfg = anim_cfg['overlay']
basemap_cfg = anim_cfg['basemap']
county_cfg = anim_cfg['county']
lake_cfg = anim_cfg['lake_fill']

# Animation settings
FPS = args.fps if args.fps else visual_cfg['fps']
DPI = args.dpi if args.dpi else visual_cfg['dpi']
DURATION_LAST_FRAME = visual_cfg['duration_last_frame']

# Visualization settings
DEPTH_MIN = visual_cfg['depth_min']
DEPTH_MAX = visual_cfg['depth_max']
COLORMAP = visual_cfg['colormap']
FIGSIZE = visual_cfg['figsize']
DOWNSAMPLE_FACTOR = visual_cfg.get('downsample_factor', 1)  # Default: no downsampling

# Overlay settings
SHOW_TIMESTAMP = overlay_cfg['show_timestamp']
SHOW_COLORBAR = overlay_cfg['show_colorbar']
SHOW_EXTENT_INFO = False  # Not in config
SHOW_DISCLAIMER = overlay_cfg['show_disclaimer']
TITLE_PREFIX = overlay_cfg['title_prefix']
DISCLAIMER_TEXT = overlay_cfg['disclaimer_text']

# Basemap settings
SHOW_BASEMAP = basemap_cfg['enabled']
# Parse basemap source string into contextily provider
if SHOW_BASEMAP:
    basemap_source_str = basemap_cfg['source']
    if basemap_source_str == 'OpenStreetMap.Mapnik':
        BASEMAP_SOURCE = ctx.providers.OpenStreetMap.Mapnik
    elif basemap_source_str == 'OpenTopoMap':
        BASEMAP_SOURCE = ctx.providers.OpenTopoMap
    elif basemap_source_str == 'Esri.WorldImagery':
        BASEMAP_SOURCE = ctx.providers.Esri.WorldImagery
    else:
        BASEMAP_SOURCE = ctx.providers.OpenStreetMap.Mapnik  # Default
else:
    BASEMAP_SOURCE = None
BASEMAP_ALPHA = basemap_cfg['alpha']

# County boundary settings
SHOW_COUNTY_BOUNDARY = county_cfg['show_boundary']
SHOW_COUNTY_LABEL = county_cfg['show_label']
COUNTY_NAME = county_cfg['name']
STATE_NAME = county_cfg['state']

# Custom map extent
USE_CUSTOM_EXTENT = extent_cfg['use_custom']
CUSTOM_EXTENT_CENTER = (extent_cfg['center_lon'], extent_cfg['center_lat'])
CUSTOM_EXTENT_SIZE_KM = (extent_cfg['size_km_ew'], extent_cfg['size_km_ns'])

# Lake masking settings
LAKE_FILL_FILE = Path(lake_cfg['file_path']) if lake_cfg.get('enabled') and lake_cfg.get('file_path') else None
LAKE_FILL_DEPTH = lake_cfg.get('depth', 5.0)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_state_abbreviation(state_name: str) -> str:
    """
    Convert full state name to 2-letter abbreviation, or return as-is if already abbreviated.

    Args:
        state_name: Full state name (e.g., "Texas") or abbreviation (e.g., "TX")

    Returns:
        2-letter state abbreviation
    """
    state_map = {
        'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
        'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
        'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
        'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
        'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
        'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
        'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
        'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
        'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
        'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
        'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
        'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
        'wisconsin': 'WI', 'wyoming': 'WY', 'district of columbia': 'DC'
    }

    state_lower = state_name.lower().strip()

    # If already an abbreviation (2 chars), return uppercase
    if len(state_lower) == 2:
        return state_name.upper()

    # Otherwise look up in map
    return state_map.get(state_lower, state_name.upper())


def get_county_boundary(county_name: str, state_name: str):
    """
    Fetch county boundary from US Census Bureau for any US county.

    Args:
        county_name: Name of county (e.g., "Kerr", "Los Angeles")
        state_name: Full state name or 2-letter abbreviation (e.g., "Texas" or "TX")

    Returns:
        GeoDataFrame with county boundary in original CRS, or None if not found
    """
    try:
        # Fetch county boundaries from US Census Bureau
        url = "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_500k.zip"
        counties = gpd.read_file(url)

        # Convert state name to abbreviation
        state_abbr = get_state_abbreviation(state_name)

        # Filter for specific county and state
        county = counties[
            (counties['NAME'].str.lower() == county_name.lower()) &
            (counties['STUSPS'] == state_abbr)
        ]

        if county.empty:
            print(f"Warning: Could not find {county_name} County, {state_name} ({state_abbr})")
            return None

        # Keep in original CRS - will be reprojected to match FIM later
        return county

    except Exception as e:
        print(f"Warning: Could not fetch county boundary: {e}")
        return None


def compute_custom_extent(center_lon, center_lat, size_km_ew, size_km_ns, target_crs):
    """
    Compute custom map extent from center point and size in kilometers.

    Args:
        center_lon: Center longitude (degrees)
        center_lat: Center latitude (degrees)
        size_km_ew: East-west extent in kilometers
        size_km_ns: North-south extent in kilometers
        target_crs: Target CRS (e.g., 'EPSG:5070')

    Returns:
        Tuple of (minx, miny, maxx, maxy) in target CRS, or None if conversion fails
    """
    try:
        from pyproj import Transformer

        # Create transformer from WGS84 to target CRS
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

        # Convert center point to target CRS
        center_x, center_y = transformer.transform(center_lon, center_lat)

        # In EPSG:5070 (Albers), units are meters
        # Convert km to meters
        half_width = (size_km_ew / 2) * 1000
        half_height = (size_km_ns / 2) * 1000

        # Compute bounds
        minx = center_x - half_width
        maxx = center_x + half_width
        miny = center_y - half_height
        maxy = center_y + half_height

        return (minx, miny, maxx, maxy)

    except Exception as e:
        print(f"Warning: Could not compute custom extent: {e}")
        return None


def get_fim_files(fims_dir: Path, pattern: str = "*_extent.tif",
                  start_time: str = None, end_time: str = None) -> list:
    """
    Get all FIM files sorted by timestamp, optionally filtered by date range.

    Args:
        fims_dir: Directory containing FIM files
        pattern: Glob pattern to match (default extent TIFFs, also supports "*_depth.tif")
        start_time: Optional start time filter (format: YYYYMMDD_HHMM)
        end_time: Optional end time filter (format: YYYYMMDD_HHMM)

    Returns:
        List of Path objects sorted by extracted timestamp
    """
    files = list(fims_dir.glob(pattern))

    # If no files found with default pattern, try depth pattern as fallback
    if not files and pattern == "*_extent.tif":
        files = list(fims_dir.glob("*_depth.tif"))
        if files:
            print(f"Note: No extent files found, using {len(files)} depth file(s) instead")

    # Sort by timestamp extracted from filename
    def extract_timestamp(filepath):
        # Format: YYYYMMDD_HHMM_extent.tif or YYYYMMDD_HHMM_depth.tif
        name = filepath.stem.replace('_depth', '').replace('_extent', '')
        try:
            return datetime.strptime(name, '%Y%m%d_%H%M')
        except ValueError:
            return datetime.min

    files = sorted(files, key=extract_timestamp)

    # Filter by date range if specified
    if start_time or end_time:
        filtered_files = []
        for f in files:
            timestamp_str = f.stem.replace('_depth', '').replace('_extent', '')

            # Check if within range
            if start_time and timestamp_str < start_time:
                continue
            if end_time and timestamp_str > end_time:
                continue

            filtered_files.append(f)

        return filtered_files

    return files


def format_timestamp(filename: str) -> str:
    """
    Extract and format timestamp from filename.

    Args:
        filename: Filename like '20250703_0700_depth.tif'

    Returns:
        Formatted string like '2025-07-03 07:00'
    """
    name = filename.replace('_depth.tif', '').replace('_extent.tif', '')
    try:
        dt = datetime.strptime(name, '%Y%m%d_%H%M')
        return dt.strftime('%Y-%m-%d %H:%M')
    except ValueError:
        return name


def apply_lake_fill(data, transform, bounds, crs, is_extent_file):
    """
    Apply lake fill mask to FIM data to fill gaps for permanent waterbodies.

    Args:
        data: FIM data array (masked array)
        transform: Rasterio affine transform
        bounds: Raster bounds
        crs: Raster CRS
        is_extent_file: Whether this is an extent file (vs depth file)

    Returns:
        Modified data array with lake pixels filled
    """
    try:
        from rasterio.features import rasterize
        from rasterio.warp import reproject, Resampling

        # Check file extension to determine how to read
        file_ext = LAKE_FILL_FILE.suffix.lower()

        if file_ext in ['.tif', '.tiff']:
            # Read as raster and reproject to match FIM
            with rasterio.open(LAKE_FILL_FILE) as lake_src:
                lake_mask = np.zeros(data.shape, dtype=np.uint8)
                reproject(
                    source=rasterio.band(lake_src, 1),
                    destination=lake_mask,
                    src_transform=lake_src.transform,
                    src_crs=lake_src.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.nearest
                )

        elif file_ext in ['.gpkg', '.geojson', '.shp']:
            # Read as vector and rasterize
            with fiona.open(LAKE_FILL_FILE) as lake_src:
                # Reproject geometries to match FIM CRS if needed
                from pyproj import Transformer

                src_crs_str = lake_src.crs_wkt if hasattr(lake_src, 'crs_wkt') else str(lake_src.crs)
                dst_crs_str = crs.to_string() if hasattr(crs, 'to_string') else str(crs)

                # Read all features and prepare for rasterization
                geometries = []
                for feature in lake_src:
                    geom = feature['geometry']

                    # If CRS differs, reproject the geometry
                    if src_crs_str != dst_crs_str:
                        transformer = Transformer.from_crs(
                            lake_src.crs,
                            crs,
                            always_xy=True
                        )

                        geom_shape = shape(geom)
                        geom_transformed = shapely_transform(transformer.transform, geom_shape)
                        geom = mapping(geom_transformed)

                    geometries.append((geom, 1))

                # Rasterize geometries
                if geometries:
                    lake_mask = rasterize(
                        geometries,
                        out_shape=data.shape,
                        transform=transform,
                        fill=0,
                        dtype=np.uint8
                    )
                else:
                    lake_mask = np.zeros(data.shape, dtype=np.uint8)
        else:
            print(f"    Warning: Unsupported lake fill file format: {file_ext}")
            return data

        # Apply the lake mask to fill gaps
        lake_pixels = lake_mask == 1

        if np.any(lake_pixels):
            # Unmask the data array if needed to modify it
            if np.ma.is_masked(data):
                data_filled = data.filled(0)
            else:
                data_filled = data.copy()

            # Set lake pixels to appropriate value
            if is_extent_file:
                data_filled[lake_pixels] = 1  # Mark as flooded
            else:
                data_filled[lake_pixels] = LAKE_FILL_DEPTH  # Set to specified depth

            # Re-create masked array preserving original mask except for lake pixels
            if np.ma.is_masked(data):
                new_mask = data.mask.copy()
                new_mask[lake_pixels] = False  # Unmask lake pixels
                data = np.ma.masked_array(data_filled, mask=new_mask)
            else:
                data = data_filled

            pixel_count = np.sum(lake_pixels)
            print(f"    Applied lake fill: {pixel_count:,} pixels added")

        return data

    except Exception as e:
        print(f"    Warning: Could not apply lake fill: {e}")
        import traceback
        traceback.print_exc()
        return data


def read_fim(fim_path: Path) -> tuple:
    """
    Read FIM GeoTIFF and return data array and metadata.

    Args:
        fim_path: Path to FIM file

    Returns:
        Tuple of (data_array, transform, bounds, crs, is_extent_file)
    """
    with rasterio.open(fim_path) as src:
        # Apply downsampling if configured (reduces memory usage for large rasters)
        if DOWNSAMPLE_FACTOR > 1:
            # Calculate new dimensions
            out_shape = (
                src.height // DOWNSAMPLE_FACTOR,
                src.width // DOWNSAMPLE_FACTOR
            )
            # Read with downsampling
            data = src.read(
                1,
                out_shape=out_shape,
                resampling=rasterio.enums.Resampling.nearest
            )
            # Adjust transform for downsampled data
            transform = src.transform * src.transform.scale(
                (src.width / out_shape[1]),
                (src.height / out_shape[0])
            )
            print(f"    Downsampled from {src.height}x{src.width} to {out_shape[0]}x{out_shape[1]} (factor: {DOWNSAMPLE_FACTOR}x)")
        else:
            data = src.read(1)
            transform = src.transform

        bounds = src.bounds
        crs = src.crs
        nodata = src.nodata

        # Check if this is an extent or depth file
        is_extent_file = '_extent' in fim_path.name

        # Mask nodata values
        if nodata is not None:
            data = np.ma.masked_equal(data, nodata)

        # For extent files, mask zeros but keep positive values (typically 1)
        # For depth files, mask values <= small threshold (0.1 ft)
        if is_extent_file:
            # Extent files: keep any positive value (usually 1 = flooded, 0 = not flooded)
            data = np.ma.masked_less_equal(data, 0)
        else:
            # Depth files: mask very small depths (< 0.1 ft) to avoid noise
            data = np.ma.masked_less(data, 0.1)

        # Apply lake fill if configured
        if LAKE_FILL_FILE is not None and LAKE_FILL_FILE.exists():
            data = apply_lake_fill(data, transform, bounds, crs, is_extent_file)

        # Debug: print statistics about the data
        valid_data = data[~data.mask] if np.ma.is_masked(data) else data
        if valid_data.size > 0:
            print(f"    Data range: {valid_data.min():.3f} - {valid_data.max():.3f}, "
                  f"Flooded cells: {valid_data.size:,}, "
                  f"Type: {'extent' if is_extent_file else 'depth'}")
        else:
            print(f"    WARNING: No valid flood data in this file!")

    return data, transform, bounds, crs, is_extent_file


def create_frame(fim_path: Path, frame_num: int, total_frames: int,
                 depth_range: tuple, cmap, figsize: tuple,
                 county_boundary=None) -> np.ndarray:
    """
    Create a single animation frame.

    Args:
        fim_path: Path to FIM file
        frame_num: Current frame number (1-indexed)
        total_frames: Total number of frames
        depth_range: Tuple of (min_depth, max_depth)
        cmap: Matplotlib colormap
        figsize: Figure size tuple
        county_boundary: GeoDataFrame with county boundary (optional)

    Returns:
        RGB image array
    """
    # Read FIM data
    data, transform, bounds, crs, is_extent_file = read_fim(fim_path)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

    # Set axis limits based on: custom extent > county boundary > FIM extent
    if USE_CUSTOM_EXTENT and crs:
        # Use custom extent centered on specified location
        custom_bounds = compute_custom_extent(
            CUSTOM_EXTENT_CENTER[0],
            CUSTOM_EXTENT_CENTER[1],
            CUSTOM_EXTENT_SIZE_KM[0],
            CUSTOM_EXTENT_SIZE_KM[1],
            crs.to_string()
        )
        if custom_bounds:
            ax.set_xlim(custom_bounds[0], custom_bounds[2])
            ax.set_ylim(custom_bounds[1], custom_bounds[3])
        else:
            # Fallback to FIM extent if custom extent fails
            ax.set_xlim(bounds.left, bounds.right)
            ax.set_ylim(bounds.bottom, bounds.top)
    elif county_boundary is not None and crs:
        # Reproject county to FIM CRS and get its bounds with padding
        county_in_fim_crs = county_boundary.to_crs(crs)
        county_bounds = county_in_fim_crs.total_bounds  # [minx, miny, maxx, maxy]

        # Add 5% padding around county boundary
        width = county_bounds[2] - county_bounds[0]
        height = county_bounds[3] - county_bounds[1]
        padding_x = width * 0.05
        padding_y = height * 0.05

        ax.set_xlim(county_bounds[0] - padding_x, county_bounds[2] + padding_x)
        ax.set_ylim(county_bounds[1] - padding_y, county_bounds[3] + padding_y)
    else:
        # Fallback to FIM extent
        ax.set_xlim(bounds.left, bounds.right)
        ax.set_ylim(bounds.bottom, bounds.top)

    # Add basemap
    if SHOW_BASEMAP and BASEMAP_SOURCE:
        try:
            # FIM files are typically in EPSG:5070
            # Contextily needs to know the CRS to properly fetch and reproject tiles
            target_crs = crs.to_string() if crs else 'EPSG:5070'
            ctx.add_basemap(ax,
                          crs=target_crs,
                          source=BASEMAP_SOURCE,
                          alpha=BASEMAP_ALPHA,
                          zoom='auto')
        except Exception as e:
            print(f"  Warning: Could not add basemap: {e}")

    # Plot flood depth with blue colormap
    # Set colormap to start with transparent for zero/no flooding
    import matplotlib
    flood_cmap = matplotlib.colormaps.get_cmap(cmap).copy()
    flood_cmap.set_bad(color='none', alpha=0)  # Transparent for masked values

    # For extent files, use fixed color (all flooded areas same blue)
    # For depth files, use gradient based on depth
    if is_extent_file:
        # Extent files: single blue color for all flooded areas
        im = ax.imshow(data,
                       extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                       cmap=flood_cmap,
                       vmin=0,
                       vmax=1,  # Binary: 0 or 1
                       interpolation='nearest',
                       origin='upper',
                       alpha=0.8,  # More opaque for visibility
                       zorder=2)
    else:
        # Depth files: gradient based on depth values
        im = ax.imshow(data,
                       extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                       cmap=flood_cmap,
                       vmin=depth_range[0],
                       vmax=depth_range[1],
                       interpolation='nearest',
                       origin='upper',
                       alpha=0.7,
                       zorder=2)

    # Format timestamp
    timestamp_str = format_timestamp(fim_path.name)

    # Add title
    if SHOW_TIMESTAMP:
        title = f"{TITLE_PREFIX} - {timestamp_str}"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    # Add county boundary if enabled
    if SHOW_COUNTY_BOUNDARY and county_boundary is not None:
        try:
            # Reproject county boundary to match FIM CRS
            if crs:
                county_in_fim_crs = county_boundary.to_crs(crs)
                county_in_fim_crs.boundary.plot(ax=ax, edgecolor='red', linewidth=3,
                                                label=f'{COUNTY_NAME} County', zorder=3)
        except Exception as e:
            print(f"  Warning: Could not plot county boundary: {e}")

    # Add colorbar
    if SHOW_COLORBAR:
        cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04, shrink=0.5)
        if is_extent_file:
            cbar.set_label('Flood Extent', rotation=270, labelpad=15, fontsize=10)
        else:
            cbar.set_label('Depth (ft)', rotation=270, labelpad=15, fontsize=10)

    # Add frame counter
    if SHOW_EXTENT_INFO:
        info_text = f"Frame {frame_num}/{total_frames}"
        ax.text(0.02, 0.98, info_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add disclaimer
    if SHOW_DISCLAIMER:
        ax.text(0.98, 0.02, DISCLAIMER_TEXT,
                transform=ax.transAxes,
                fontsize=11,
                fontweight='bold',
                horizontalalignment='right',
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, edgecolor='red', linewidth=2))

    # Add county label in lower left corner
    if SHOW_COUNTY_LABEL:
        county_label = f"{COUNTY_NAME} County, {STATE_NAME}"
        ax.text(0.02, 0.02, county_label,
                transform=ax.transAxes,
                fontsize=12,
                fontweight='bold',
                color='red',
                horizontalalignment='left',
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red', linewidth=2))

    # Format axes - remove labels and ticks for cleaner map view
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)  # Remove grid for cleaner appearance

    # Remove axis spines for cleaner map appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Tight layout
    plt.tight_layout()

    # Convert figure to RGB array
    fig.canvas.draw()
    # Use buffer_rgba() instead of deprecated tostring_rgb()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    image = buf.reshape(h, w, 4)  # RGBA format
    image = image[:, :, :3]  # Convert RGBA to RGB by dropping alpha channel

    # Aggressively clean up matplotlib objects to prevent memory leaks
    plt.close(fig)
    plt.clf()
    plt.cla()

    return image


# ==============================================================================
# MAIN ANIMATION GENERATION
# ==============================================================================

def main():
    print("=" * 70)
    print("FLOOD ANIMATION GENERATION")
    print("=" * 70)
    print()

    # Validate paths
    if not FIMS_DIR.exists():
        print(f" ERROR: FIMs directory not found: {FIMS_DIR}")
        return 1

    # Get all FIM files with date range filter
    fim_files = get_fim_files(FIMS_DIR, start_time=START_TIME, end_time=END_TIME)

    if not fim_files:
        print(f" ERROR: No FIM files found in {FIMS_DIR}")
        if START_TIME or END_TIME:
            print(f"  Date range filter: {START_TIME or 'start'} to {END_TIME or 'end'}")
        print("  Looking for files matching pattern: *_extent.tif or *_depth.tif")
        return 1

    print(f"Found {len(fim_files)} FIM file(s)")
    if START_TIME or END_TIME:
        print(f"Date range filter: {START_TIME or 'start'} to {END_TIME or 'end'}")
    print(f"Time range: {format_timestamp(fim_files[0].name)} to {format_timestamp(fim_files[-1].name)}")
    print()

    # Create output directory
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    print(f"Configuration:")
    print(f"  Input directory: {FIMS_DIR}")
    print(f"  Output video: {OUTPUT_VIDEO}")
    if START_TIME or END_TIME:
        print(f"  Time window: {START_TIME or 'start'} to {END_TIME or 'end'}")
    print(f"  Frame rate: {FPS} fps")
    print(f"  Resolution: {DPI} dpi")
    print(f"  Depth range: {DEPTH_MIN} - {DEPTH_MAX} ft")
    print(f"  Colormap: {COLORMAP}")
    if DOWNSAMPLE_FACTOR > 1:
        print(f"  Downsampling: {DOWNSAMPLE_FACTOR}x (reduces memory usage)")
    print(f"  Basemap: {'Enabled' if SHOW_BASEMAP else 'Disabled'}")
    print(f"  County boundary: {'Enabled' if SHOW_COUNTY_BOUNDARY else 'Disabled'}")
    print(f"  Disclaimer: {'Enabled' if SHOW_DISCLAIMER else 'Disabled'}")
    print()

    # Fetch county boundary if enabled
    county_boundary = None
    if SHOW_COUNTY_BOUNDARY:
        print(f"Fetching {COUNTY_NAME} County, {STATE_NAME} boundary...")
        county_boundary = get_county_boundary(COUNTY_NAME, STATE_NAME)
        if county_boundary is not None:
            print(f"   County boundary loaded")
        else:
            print(f"   Could not load county boundary")
        print()

    # Create colormap
    cmap = plt.get_cmap(COLORMAP)

    # Open video writer to write frames directly (avoid storing all in memory)
    print(f"Opening video writer: {OUTPUT_VIDEO}")
    try:
        writer = imageio.get_writer(
            OUTPUT_VIDEO,
            fps=FPS,
            codec='libx264',
            pixelformat='yuv420p',
            ffmpeg_params=['-crf', '18', '-preset', 'fast']  # Use 'fast' instead of 'slow' for speed
        )
    except Exception as e:
        print(f" ERROR: Could not open video writer: {e}")
        return 1

    # Generate and write frames directly to video
    print("Generating and writing frames...")
    frame_count = 0
    last_frame = None

    for i, fim_file in enumerate(fim_files, 1):
        print(f"  [{i}/{len(fim_files)}] Processing {fim_file.name}...", end='')

        try:
            frame = create_frame(
                fim_file,
                frame_num=i,
                total_frames=len(fim_files),
                depth_range=(DEPTH_MIN, DEPTH_MAX),
                cmap=cmap,
                figsize=FIGSIZE,
                county_boundary=county_boundary
            )
            writer.append_data(frame)
            last_frame = frame.copy()  # Make a copy for holding at end
            del frame  # Explicitly delete frame to free memory
            frame_count += 1
            print(" ")

            # Force garbage collection every 10 frames to prevent memory buildup
            if i % 10 == 0:
                gc.collect()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    if frame_count == 0:
        print("\n ERROR: No frames were generated")
        writer.close()
        return 1

    # Add extra frames at the end to hold last frame
    if DURATION_LAST_FRAME > 0 and last_frame is not None:
        extra_frames = int(FPS * DURATION_LAST_FRAME)
        print(f"\nHolding last frame for {DURATION_LAST_FRAME}s ({extra_frames} frames)...")
        for i in range(extra_frames):
            writer.append_data(last_frame)
            frame_count += 1

    # Close the writer
    print("\nFinalizing video...")
    writer.close()

    # Clean up memory
    del last_frame
    gc.collect()

    print(" Video created successfully!")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total frames: {frame_count}")
    print(f"Duration: {frame_count / FPS:.1f} seconds")
    print(f"Output: {OUTPUT_VIDEO}")
    print()

    # File size
    if OUTPUT_VIDEO.exists():
        size_mb = OUTPUT_VIDEO.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB")

    print("\n Animation generation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
