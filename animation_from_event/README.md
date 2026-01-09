# Flood Animation from Event

Automated flood inundation mapping and animation tool using NWM streamflow data, RIPPLE-FIM libraries, and flows2fim.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Lake Fill Feature](#lake-fill-feature)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## Overview

This containerized tool automates the complete workflow for creating flood inundation animations from hydro-meteorological events:

```
NWM Data (S3) → Flow Files → flows2fim → FIM GeoTIFFs → Animation Video
```

### Workflow Steps

1. **Download Data** - RIPPLE collection (ripple.gpkg, start_reaches.csv) from S3
2. **Generate Flows** - Extract NWM streamflow data for event period
3. **Generate FIMs** - Create flood inundation maps using flows2fim
4. **Create Animation** - Render video with basemap, timestamps, and overlays

---

## Features

- **Make Build System** - Simple commands for build, run, and clean workflows
- **Fully Containerized** - Docker-based, runs anywhere
- **S3 Integration** - Auto-downloads RIPPLE data and accesses NWM data
- **Config-Driven** - Single YAML file for all settings
- **Dynamic Paths** - Collection ID-based S3 paths (change once, update all)
- **County Agnostic** - Works with any US county boundary
- **Lake Fill** - Fill permanent water bodies in animations
- **Customizable Visualization** - Basemaps, colormaps, extents, overlays
- **Parallel Processing** - Multi-threaded FIM generation

---

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- 8GB+ RAM, 20GB+ disk space
- AWS credentials (optional, for private S3 buckets)

### 1. Initial Setup

```bash
cd animation_from_event

# Create directories and environment file
make setup

# Paste AWS Creds to `.env`
export AWS_ACCESS_KEY_ID= <your_access_key_id>
export AWS_SECRET_ACCESS_KEY= <your_secret_access_key>
export AWS_SESSION_TOKEN= <your_session_token>

# Edit config with your collection and event dates
vim config.yaml
```

### 2. Configure Your Event

Edit `config.yaml` - **only need to change 3 lines**:

```yaml
collection:
  id: "ble_12100201_UpperGuadalupe"  # ← Change to your collection ID

event:
  start_date: "2025-07-04 00:00"  # ← Your event start time
  end_date: "2025-07-08 14:00"    # ← Your event end time
```

**Note:** For events longer than 20 hours, see [Long-Duration Events](#long-duration-events-20-hours) section for best practices on breaking animations into segments.

### 3. Build Container

```bash
make build
```

This takes 5+ minutes the first time.

### 4. Run Workflow

```bash
# Run complete workflow (all steps)
make run-workflow

# OR run individual steps:
make generate-flows     # Generate flow files
make generate-fims      # Generate FIMs
make generate-animation # Create video
```

### 5. Get Your Video

```bash
ls -lh data/output/flood_animation.mp4
```

---

## Installation

### Using Make Commands (Recommended)

```bash
# Initial setup
make setup      # Create directories, copy .env template
make build      # Build Docker image
```
### Verify Setup (Optional)

Before running the workflow, you can verify your environment:
```bash
./test_setup.sh
```

### Manual Setup

```bash
# Create directories
mkdir -p data/input data/output data/cache

# Copy environment template
cp .env.example .env

# Insert AWS credentials
vim .env

# Build container
docker-compose build

# Run workflow
docker-compose run --rm flood-animation python run_workflow.py --config config.yaml
```

---

## Configuration

All settings are in [`config.yaml`](config.yaml). The file has detailed comments for each option.

### Essential Settings

#### 1. Collection Configuration

```yaml
collection:
  id: "ble_12100201_UpperGuadalupe"  # CHANGE THIS to your collection
  name: "Upper Guadalupe Basin"

  s3:
    bucket: "fimc-data"
    base_path: "ripple/fim_100_domain/collections"
    # Paths auto-constructed: {base_path}/{collection.id}/{filename}
```

Change `collection.id` and all S3 paths update automatically:
- `{base_path}/{id}/ripple.gpkg`
- `{base_path}/{id}/start_reaches.csv`
- `{base_path}/{id}/library_extent`

#### 2. Event Time Period

```yaml
event:
  start_date: "2025-07-04 02:00"  # Format: YYYY-MM-DD HH:MM (or just YYYY-MM-DD for midnight)
  end_date: "2025-07-08"    # Format: YYYY-MM-DD HH:MM (or just YYYY-MM-DD for midnight)

  nwm:
    bucket: "noaa-nwm-pds"                      # For dates after 2023
    # bucket: "noaa-nwm-retrospective-3-0-pds"  # For 1979-2023
    configuration: "analysis_assim"
    use_anonymous: true  # No credentials needed for public buckets
```

#### 3. FIM Settings

```yaml
fim:
  type: "extent"              # "depth" or "extent"
  output_format: "COG"        # "VRT", "COG", or "GTIFF"
  boundary_condition: "nd"    # "nd" (normal depth) or "kwse"

  # Starting reach IDs (boundary conditions for flows2fim)
  starting_reaches: "/data/input/start_reaches.csv"  # Options:
                              #   "/data/input/start_reaches.csv" - Use CSV file (default)
                              #   "auto" - Auto-detect upstream reaches from ripple.gpkg
                              #   "123456,789012" - Comma-separated reach IDs
```

**Starting Reaches Explained:**

The `starting_reaches` setting controls which reaches are used as upstream boundary conditions for FIM generation:

- **CSV file path** (default: `"/data/input/start_reaches.csv"`) - Use a CSV file with two columns: `reach_id,control_stage`. This allows you to specify custom water surface elevations for each starting reach. The workflow attempts to download this file from S3 along with ripple.gpkg.

- **`"auto"`** - Automatically detects the upstream-most reaches (headwaters) from the ripple.gpkg database by finding reaches with no upstream connections. This is a reliable fallback when CSV files aren't available.

- **Comma-separated IDs** - Manually specify reach IDs (e.g., `"123456,789012,345678"`). Useful when you know specific reaches to use as boundary conditions.

**Automatic Fallback:** If the configured CSV file doesn't exist locally (e.g., not available in your S3 collection), the tool automatically falls back to `"auto"` mode and detects upstream reaches from ripple.gpkg.

#### 4. Animation Settings

```yaml
animation:
  # Map extent (customize for your area)
  extent:
    use_custom: true          # false = use full FIM extent
    center_lon: -99.23        # Center longitude (degrees)
    center_lat: 30.12         # Center latitude (degrees)
    size_km_ew: 10            # East-west extent (km)
    size_km_ns: 6             # North-south extent (km)

  # Visual settings
  visual:
    fps: 1.0                  # Frames per second
    dpi: 250                  # Resolution
    duration_last_frame: 3    # Hold last frame (seconds)
    depth_min: 0.0            # Min depth to display (ft)
    depth_max: 30.0           # Max depth for colormap (ft)
    colormap: "GnBu"          # Matplotlib colormap
    figsize: [14, 10]         # Figure size (inches)

  # Overlays
  overlay:
    show_timestamp: true
    show_colorbar: false
    show_disclaimer: true
    disclaimer_text: "Disclaimer: Experimental Guidance"
    title_prefix: "Flood Extent"

  # Basemap
  basemap:
    enabled: true
    source: "OpenStreetMap.Mapnik"  # or "OpenTopoMap", "Esri.WorldImagery", etc.
    alpha: 0.5                      # Transparency (0-1)

  # County boundary (works for any US county)
  county:
    show_boundary: false
    show_label: false
    name: "Kerr"                    # Any county name
    state: "Texas"                  # Full name or abbreviation (TX)

  # Lake fill (optional - see Lake Fill Feature section)
  lake_fill:
    enabled: false
    # file_path: "/data/input/lake_polygons.geojson"
    depth: 5.0  # Depth to assign (ft)
```

#### 5. Processing Settings

```yaml
processing:
  flows2fim_executable: "flows2fim"  # Binary installed in container
  max_workers: 4                     # Parallel FIM workers
```

---

## Usage

### Run Complete Workflow

```bash
make run-workflow
```

This runs all four steps:
1. Downloads RIPPLE data from S3
2. Generates hourly flow files from NWM
3. Generates FIMs using flows2fim
4. Creates animation video

### Workflow Efficiency Tips

**Important:** Steps 1-2 (downloading NWM NetCDF files and generating flow/control files) are **time-consuming**.

**Best Practice:** Pull data for a **longer time period initially** (e.g., full week), then re-run only the animation step (Step 4) with different visualization parameters, time windows, or extents.

```bash
# Initial run: Pull full event data (e.g., 7 days)
# Edit config.yaml: start_date: 2025-07-04, end_date: 2025-07-11
make run-workflow --skip-animation

# Later: Create different animations (subset of full timeserires from comprehensive data)
# Just modify animation settings (and start / end times) in config.yaml (extent, colormap, etc.)
make generate-animation

# Or create animation for subset of time period
docker-compose run --rm flood-animation python generate_animation.py \
  --config config.yaml \
  --start-time "2025-07-04 06:00" \
  --end-time "2025-07-04 18:00"
```

**Smart File Skipping:**
- **Step 2 (Flow generation)** automatically checks if flow files exist for the configured time range and skips regeneration if all files are present.
- **Step 3 (FIM generation)** automatically skips already-existing controls and FIM files. If your workflow is interrupted or you need to regenerate only new timesteps, simply re-run `make generate-fims` - it will skip completed files and only generate missing ones.

This approach saves significant time when experimenting with different visualizations or creating multiple animations for different time windows.

To force regeneration of existing files:
```bash
# Force regenerate flow files
docker-compose run --rm flood-animation python generate_flow_files.py --config config.yaml --force

# Force regenerate FIM files
docker-compose run --rm flood-animation python generate_batch_fims.py --config config.yaml --force
```

### Long-Duration Events (>20 hours)

**Important:** For events longer than 20 hours, it's recommended to create multiple shorter animations and stitch them together. This approach:
- Reduces memory pressure and OOM risk
- Allows parallel processing of segments
- Makes it easier to recover from failures
- Provides checkpoint progress for long events

**Workflow for multi-day events:**

```bash
# Step 1: Generate all flow and FIM files for the full event period
# Edit config.yaml: start_date: 2025-07-04, end_date: 2025-07-11 (7 days)
make run-workflow --skip-animation

# Step 2: Create animations for each day separately
docker-compose run --rm flood-animation python generate_animation.py \
  --config config.yaml \
  --start-time "2025-07-04 00:00" \
  --end-time "2025-07-04 20:00" \
  --output video_1.mp4

docker-compose run --rm flood-animation python generate_animation.py \
  --config config.yaml \
  --start-time "2025-07-04 20:00" \
  --end-time "2025-07-05 16:00" \
  --output video_2.mp4

docker-compose run --rm flood-animation python generate_animation.py \
  --config config.yaml \
  --start-time "2025-07-05 16:00" \
  --end-time "2025-07-06 12:00" \
  --output video_3.mp4

docker-compose run --rm flood-animation python generate_animation.py \
  --config config.yaml \
  --start-time "2025-07-06 12:00" \
  --end-time "2025-07-07 8:00" \
  --output video_4.mp4

# Step 3: Stitch videos together using ffmpeg
docker-compose run --rm flood-animation bash -c "\
  ffmpeg -i /data/output/video_1.mp4 \
         -i /data/output/video_2.mp4 \
         -i /data/output/video_3.mp4 \
         -i /data/output/video_4.mp4 \
         -filter_complex 'concat=n=4:v=1:a=0' \
         /data/output/flood_animation_full.mp4"
```

**Alternative: Using a file list for many segments**

```bash
# Create a file list for ffmpeg
cat > data/output/video_list.txt <<EOF
file 'video_day1.mp4'
file 'video_day2.mp4'
file 'video_day3.mp4'
file 'video_day4.mp4'
file 'video_day5.mp4'
file 'video_day6.mp4'
file 'video_day7.mp4'
EOF

# Concatenate using file list
docker-compose run --rm flood-animation bash -c "\
  ffmpeg -f concat -safe 0 -i /data/output/video_list.txt \
         -c copy /data/output/flood_animation_full.mp4"
```

**Memory considerations:**
- Each day (24 hours) = ~24-48 frames at 1-2 hour intervals
- Peak memory usage scales with: `frames × DPI × figure_size × downsample_factor`
- Recommended segment length: 12-20 hours per video
- For very high resolution (DPI > 200), consider 6-12 hour segments

### Run Individual Steps

```bash
# Step 1: Download RIPPLE data
docker-compose run --rm flood-animation python utils_s3.py --config config.yaml --download-ripple

# Step 2: Generate flow files
make generate-flows

# Step 3: Generate FIMs
make generate-fims

# Step 4: Create animation
make generate-animation
```

### Skip Steps

```bash
# Skip download (use existing local files)
docker-compose run --rm flood-animation python run_workflow.py --config config.yaml --skip-download

# Skip flows (use existing flow files)
docker-compose run --rm flood-animation python run_workflow.py --config config.yaml --skip-flows

# Skip FIMs (use existing FIM files)
docker-compose run --rm flood-animation python run_workflow.py --config config.yaml --skip-fims
```

### Cleanup Commands

Remove generated files to free disk space or start fresh:

```bash
make clean
```

This removes:
- `data/output/*` - All generated outputs (flows, controls, FIMs, videos)
- `data/cache/*` - Temporary files

**Use when:**
- Starting a new event analysis
- Freeing disk space
- Troubleshooting issues with stale data

**Note:** Input files (`data/input/*`) are preserved. Downloaded RIPPLE data and lake polygons remain intact.

```bash
# Complete cleanup - remove all Docker artifacts
make clean-all
```

This performs `make clean` plus:
- Removes Docker images
- Removes Docker volumes
- Stops all containers

**Use when:**
- Rebuilding from scratch
- Freeing maximum disk space (~3.5GB from Docker image)
- Resolving Docker-related issues

**Warning:** After `make clean-all`, you'll need to rebuild: `make build` (takes 5-10 minutes)

**Example workflow:**
```bash
# Clean output for new analysis
make clean
make run-workflow  # Uses existing Docker image

# Complete fresh start
make clean-all
make build         # Rebuild Docker image
make run-workflow
```

### Command-Line Overrides

```bash
# Override dates
docker-compose run --rm flood-animation python generate_flow_files.py \
  --config config.yaml \
  --start-date 2025-07-01 \
  --end-date 2025-07-03

# Override visualization
docker-compose run --rm flood-animation python generate_animation.py \
  --config config.yaml \
  --dpi 300 \
  --fps 2.0
```

### Interactive Shell

```bash
make shell
# Now inside container:
python generate_flow_files.py --config config.yaml
exit
```

---

## Lake Fill Feature

Fill permanent water bodies (lakes, reservoirs) in your animations to show continuous flooding, even where FIM data has gaps.

### Quick Start

```bash
# Download lake polygon interactively
make download-lake

# Follow prompts:
#   Lake name: Lake Mead
#   Bounding box: 36.0,-114.8,36.3,-114.4
#   Output: lake_mead.geojson

# In config.yaml
vim config.yaml
# Set lake_fill.enabled = true 
# Set lake_fill.file_path: "/data/input/<lake name>.geojson"
```

### Finding Bounding Box Coordinates

For simplicity, you can use https://boundingbox.klokantech.com/:
1. Navigate to your lake
2. Draw a box around it
3. Select "CSV" format
4. Copy coordinates: `south,west,north,east`

### Usage Examples

**Interactive Mode:**
```bash
make download-lake
```

**Command Line:**
```bash
# Lake Mead, Nevada/Arizona
docker-compose run --rm flood-animation python download_lake_polygon.py \
  --name "Lake Mead" \
  --bbox 36.0,-114.8,36.3,-114.4 \
  --output /data/input/lake_mead.geojson

# Lake Travis, Texas
docker-compose run --rm flood-animation python download_lake_polygon.py \
  --name "Travis" \
  --bbox 30.3,-98.1,30.5,-97.9 \
  --output /data/input/lake_travis.geojson

# Lake Ingram (multiple lakes with partial name match)
docker-compose run --rm flood-animation python download_lake_polygon.py \
  --name "Ingram" \
  --bbox 29.9,-99.5,30.3,-99.0 \
  --output /data/input/lake_ingram.geojson
```

### Configuration

Edit `config.yaml`:

```yaml
animation:
  lake_fill:
    enabled: true
    file_path: "/data/input/lake_<name>.geojson"
    depth: 5.0  # For extent: any value works; for depth: controls color
```

### Supported File Formats

- GeoJSON (`.geojson`) - Recommended
- GeoPackage (`.gpkg`)
- Shapefile (`.shp`)
- GeoTIFF (`.tif`)

### Multiple Lakes

Single file can contain multiple lakes (FeatureCollection). All will be filled:

```json
{
  "type": "FeatureCollection",
  "features": [
    {"properties": {"name": "Lake New Ingram"}, "geometry": {...}},
    {"properties": {"name": "Lake Old Ingram"}, "geometry": {...}}
  ]
}
```

### Alternative Data Sources

If OpenStreetMap doesn't have your lake:

1. **USGS National Map**
   - Visit: https://apps.nationalmap.gov/downloader/
   - Download NHD (National Hydrography Dataset)

2. **Manual Digitization**
   - Open QGIS
   - Load basemap
   - Digitize lake boundary
   - Export as GeoJSON

3. **NWM Lakes Dataset**
   - Extract from nwm_lakes.gpkg by location

---

## Advanced Configuration

### Custom Map Extent

Zoom to a specific area instead of full county:

```yaml
animation:
  extent:
    use_custom: true
    center_lon: -99.23    # Your center point
    center_lat: 30.12
    size_km_ew: 10        # Width (km)
    size_km_ns: 6         # Height (km)
```

### Basemap Options

```yaml
basemap:
  source: "OpenStreetMap.Mapnik"  # Street map (default)
  # source: "OpenTopoMap"         # Topographic
  # source: "Esri.WorldImagery"   # Satellite
  alpha: 0.5  # Adjust transparency
```

### Custom Colormaps

```yaml
visual:
  colormap: "GnBu"      # Default: Green-Blue
  # colormap: "Blues"   # Blue shades
  # colormap: "YlOrRd"  # Yellow-Orange-Red
  # colormap: "viridis" # Perceptually uniform
```

See Matplotlib colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html

### County Boundaries (Any US County)

```yaml
county:
  show_boundary: true
  show_label: true
  name: "Los Angeles"
  state: "California"  # or "CA"

  # Also works:
  # name: "Cook", 
  # state: "IL"
  # OR
  # name: "Miami-Dade"
  # state: "Florida"
```

State names or abbreviations both work (e.g., "Texas" or "TX").

### Parallel Processing

Adjust based on your CPU cores and RAM:

```yaml
processing:
  max_workers: 4  # Default
  # max_workers: 8  # More cores = faster (if you have RAM)
  # max_workers: 2  # Less RAM usage
```

Rule of thumb: `max_workers = CPU cores - 1`, but watch RAM usage.

---

## Troubleshooting

### Docker Issues

**Can't connect to Docker?**
```bash
# Start Docker
sudo systemctl start docker  # Linux
# Or start Docker Desktop (Mac/Windows)
```

**Permission denied?**
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Log out and back in
```

### AWS Credentials

**Credentials not working?**
```bash
# Check environment variables
docker-compose run --rm flood-animation env | grep AWS

# Verify .env file
cat .env

# For public buckets, use anonymous access:
# In config.yaml: use_anonymous: true
```

### Memory Issues

**Out of memory / container killed?**

The Docker memory limits have been removed by default (see `docker-compose.yml`). Common memory related errors are `exit code -9` or  `Error 137`, If still experiencing issues:

Increasing the downsample_factor:
```yaml
visual:
  downsample_factor: 4  # Downsample large rasters
```

Reduce animation resolution:
```yaml
visual:
  dpi: 150  # Reduce from 250
```

Or reduce parallel workers in `config.yaml`:
```yaml
processing:
  max_workers: 2  # Reduce from 4
```

**For events longer than 20 hours:** Break the animation into segments and stitch them together. See [Long-Duration Events](#long-duration-events-20-hours). This is the recommended approach for multi-day events:

### Disk Space Issues

**Running out of disk space?**

```bash
# Check disk usage
du -sh data/output/*

# Remove output files
make clean

# For more space, remove Docker artifacts (~3.5GB)
make clean-all
docker system prune -a  # Remove unused Docker data
```

### flows2fim Errors

**flows2fim not found?**

Rebuild container (flows2fim should be installed):
```bash
make build

# Verify installation
docker-compose run --rm flood-animation flows2fim --version
```

**flows2fim fails on FIM generation?**

Check paths in config.yaml:
- `ripple.gpkg` exists
- `start_reaches.csv` exists
- FIM library path correct on S3

### S3 Download Fails

**Can't download RIPPLE data?**

Check:
1. Collection ID correct in config.yaml
2. S3 paths exist: `s3://fimc-data/ripple/fim_100_domain/collections/{collection.id}/`
3. Valid AWS credentials in `.env`
4. Internet connection

**Anonymous access fails?**
```yaml
# Try with credentials in .env
nwm:
  use_anonymous: false
```

### Animation Issues

**No FIM files found?**
```bash
# Check FIM directory
ls -lh data/output/fims/

# If empty, run FIM generation:
make generate-fims
```

**Lake fill not working?**

Verify:
1. `lake_fill.enabled: true` in config.yaml
2. File path correct: `/data/input/<your_lake>.geojson`
3. File exists: `ls data/input/<your_lake>.geojson`
4. File is valid GeoJSON (check with QGIS or geojson.io)

**Basemap not showing?**

Check internet connection - basemap tiles download at runtime.

Or disable basemap:
```yaml
basemap:
  enabled: false
```

### No Output Video

**Video not created?**

Check logs:
```bash
docker-compose logs
```

Common issues:
- No FIM files (run `make generate-fims`)
- Memory issue (reduce DPI or workers)
- FFmpeg error (check logs)

### Common Error Messages

| Error | Solution |
|-------|----------|
| `OOM killed` | Increase `downsample_factor`, reduce `max_workers` or `dpi` in config.yaml |
| `No space left on device` | `make clean` then `docker system prune -a` |
| `ripple.gpkg not found` | Run download step: `make generate-flows` |
| `flows2fim command not found` | `make clean-all` then `make build` |
| `Access Denied (S3)` | Check AWS credentials or use anonymous |
| `No NWM data for date` | Check date range and NWM bucket |
| `Stale output data` | `make clean` to remove old files |
| `Docker build fails` | `make clean-all` then `make build` |

---

## Development

### Directory Structure

```
animation_from_event/
├── config.yaml              # Main configuration
├── .env                     # AWS credentials (gitignored)
├── .env.example             # Credentials template
├── Dockerfile               # Container definition
├── docker-compose.yml       # Docker Compose config
├── requirements.txt         # Python dependencies
├── Makefile                 # Convenience commands
│
├── run_workflow.py          # Workflow orchestrator
├── config_utils.py          # Config loading utilities
├── utils_s3.py              # S3 download functions
│
├── test_setup.sh            # Verifies installation & environment
│
├── generate_flow_files.py   # Step 1: Flow generation
├── generate_batch_fims.py   # Step 2: FIM generation
├── generate_animation.py    # Step 3: Animation creation
├── download_lake_polygon.py # Utility: Lake download
│
├── data/                    # Mounted volume (gitignored)
│   ├── input/               # Downloaded RIPPLE data
│   │   ├── ripple.gpkg
│   │   ├── start_reaches.csv
│   │   └── lake_*.geojson
│   ├── output/              # Generated outputs
│   │   ├── flows/           # NWM flow CSVs
│   │   ├── controls/        # flows2fim control files
│   │   ├── fims/            # FIM GeoTIFFs
│   │   └── flood_animation.mp4
│   └── cache/               # Temporary files
│
└── README.md                # This file
```

### Container Paths

All data uses standard container paths (not user-specific):

| Purpose | Container Path | Host Path |
|---------|---------------|-----------|
| Input data | `/data/input/` | `./data/input/` |
| Output data | `/data/output/` | `./data/output/` |
| Cache | `/data/cache/` | `./data/cache/` |
| Config | `/app/config.yaml` | `./config.yaml` |
| Scripts | `/app/*.py` | `./*.py` |

### Make Commands Reference

```bash
# Setup
make setup              # Create directories, copy .env
make build              # Build Docker image

# Run
make run-workflow       # Complete workflow (all steps)
make shell              # Interactive shell in container

# Generate individual steps
make generate-flows     # Generate flow files from NWM data
make generate-fims      # Generate FIM GeoTIFFs
make generate-animation # Generate animation video

# Utilities
make download-lake      # Download lake polygon (interactive)

# Cleanup
make clean              # Remove output files (data/output, data/cache)
make clean-all          # Remove output + Docker images and volumes
make logs               # Show Docker logs
make help               # Show all commands
```

### Python Script Arguments

All three main scripts support config-based execution:

**generate_flow_files.py**
```bash
python generate_flow_files.py --config config.yaml \
  [--start-date YYYY-MM-DD] \
  [--end-date YYYY-MM-DD]
```

**generate_batch_fims.py**
```bash
python generate_batch_fims.py --config config.yaml \
  [--max-workers N]
```

**generate_animation.py**
```bash
python generate_animation.py --config config.yaml \
  [--dpi N] \
  [--fps N.N]
```

### Extending the Tool

#### Add Custom Processing Step

1. Create Python script in project root
2. Import config utilities:
```python
from config_utils import load_config, get_paths
```
3. Add to `run_workflow.py` if desired
4. Add Make command in `Makefile`

#### Custom Basemap Source

Edit `generate_animation.py` basemap parsing (lines 96-110) to add new providers.

#### Custom Output Formats

Modify `generate_animation.py` to support additional video formats (MP4, AVI, GIF, etc.).

---

## Recommended System Requirements

- **CPU:** 4+ cores (for parallel processing)
- **RAM:** 16 GB
- **Disk:** 50 GB+ SSD
- **Network:** Broadband (for S3 downloads and basemap tiles)

---

## Dependencies

### System (in Container)

- Python 3.11
- ffmpeg (video encoding)
- GDAL 3.x (geospatial library)
- libhdf5, libnetcdf (NWM data)
- flows2fim v0.4.1 (FIM generation)

### Python Packages

- **Data:** boto3, pandas, numpy, xarray, netCDF4
- **Geospatial:** rasterio, fiona, geopandas, shapely, pyproj
- **Visualization:** matplotlib, imageio, contextily
- **Utilities:** pyyaml, python-dotenv

See [`requirements.txt`](requirements.txt) for full list.

---
