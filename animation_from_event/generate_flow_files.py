#!/usr/bin/env python3
"""
Generate Hourly Flow Files from NWM Data
==========================================

Fetches NWM streamflow data from S3 for all reaches in a RIPPLE rating
curve database (ripple.gpkg) and writes one CSV file per hourly timestep.

Output format (per file):
    nwm_feature_id,discharge
    3458519,1250.5
    3458691,4320.0
    ...

Configuration:
    - Output directory: config.yaml -> output.base_dir/flows
    - Filename format: YYYYMMDD_HHMM_{suffix}.csv (suffix from config.yaml)
    - Time period: config.yaml -> event.start_date to event.end_date

Usage:
    python generate_flow_files.py --config config.yaml
    python generate_flow_files.py --config config.yaml --force  # Regenerate even if files exist
    python generate_flow_files.py --start-date "2025-07-04 06:00" --end-date "2025-07-05 18:00"
"""

import pandas as pd
import sqlite3
import xarray as xr
import boto3
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config
import argparse
from dataclasses import dataclass
from config_utils import load_config, get_paths, get_nwm_config

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Load configuration
parser = argparse.ArgumentParser(description="Generate hourly flow files from NWM data")
parser.add_argument('--config', default='config.yaml', help="Path to config file")
parser.add_argument('--start-date', help="Override start date/time (YYYY-MM-DD or YYYY-MM-DD HH:MM)")
parser.add_argument('--end-date', help="Override end date/time (YYYY-MM-DD or YYYY-MM-DD HH:MM)")
parser.add_argument('--force', action='store_true', help="Force regeneration even if files exist")
args = parser.parse_args()

config = load_config(args.config)
paths = get_paths(config)
nwm_cfg = get_nwm_config(config)

# Set paths from config
RIPPLE_DB_PATH = paths['ripple_db']
OUTPUT_DIR = paths['flows_dir']
FLOW_FILE_SUFFIX = paths['flow_file_suffix']
START_DATE = args.start_date if args.start_date else nwm_cfg['start_date']
END_DATE = args.end_date if args.end_date else nwm_cfg['end_date']
S3_BUCKET = nwm_cfg['bucket']
NWM_CONFIG = nwm_cfg['config']
USE_ANONYMOUS_S3 = nwm_cfg['use_anonymous']


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

@dataclass
class ProcessingStats:
    """Track processing statistics."""
    files_processed: int = 0
    files_failed: int = 0
    files_written: int = 0
    files_skipped: int = 0
    reaches_with_data: int = 0


def parse_datetime_string(date_str: str) -> datetime:
    """
    Parse a datetime string that can be in multiple formats.

    Supports:
        - "YYYY-MM-DD" (defaults to 00:00:00)
        - "YYYY-MM-DD HH:MM"
        - "YYYY-MM-DD HH:MM:SS"

    Args:
        date_str: Date/datetime string

    Returns:
        datetime object
    """
    date_str = date_str.strip()

    # Try different formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Could not parse date string: {date_str}. Expected format: YYYY-MM-DD or YYYY-MM-DD HH:MM")


def extract_timestamp_from_nwm_filename(filename: str, date_str: str, bucket_type: str) -> pd.Timestamp:
    """
    Extract timestamp from NWM NetCDF filename.

    Args:
        filename: NetCDF filename
        date_str: Date string in YYYY-MM-DD format
        bucket_type: 'operational' or 'retrospective'

    Returns:
        pandas Timestamp with UTC timezone, or None if parsing fails
    """
    try:
        if bucket_type == 'operational':
            # Operational format: nwm.tHHz.analysis_assim.channel_rt.tm00.conus.nc
            parts = filename.split('.')
            for part in parts:
                if part.startswith('t') and part.endswith('z') and len(part) == 4:
                    hour = part[1:3]  # Extract HH from tHHz
                    return pd.Timestamp(f"{date_str} {hour}:00:00", tz='UTC')
        else:
            # Retrospective format: YYYYMMDDHHMI.CHRTOUT_DOMAIN1
            if len(filename) >= 12:
                date_part = filename[:8]  # YYYYMMDD
                hour_part = filename[8:10]  # HH
                return pd.Timestamp(
                    f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {hour_part}:00:00",
                    tz='UTC'
                )
    except Exception:
        pass
    return None


def build_s3_prefix_and_pattern(date: datetime, bucket: str, config: str) -> tuple[str, str]:
    """
    Build S3 prefix and file pattern for listing NWM files.

    Args:
        date: Date to build path for
        bucket: S3 bucket name
        config: NWM configuration (e.g., 'analysis_assim')

    Returns:
        Tuple of (prefix, file_pattern)
    """
    year_month_day = date.strftime("%Y%m%d")

    if bucket == "noaa-nwm-pds":
        # Operational bucket structure
        prefix = f"nwm.{year_month_day}/{config}/"
        file_pattern = f".{config}.channel_rt.tm00.conus.nc"
    else:
        # Retrospective bucket structure
        year = date.strftime("%Y")
        prefix = f"CONUS/netcdf/CHRTOUT/{year}/"
        file_pattern = year_month_day

    return prefix, file_pattern


# ==============================================================================
# DATABASE FUNCTIONS
# ==============================================================================

def get_reach_ids_from_database(db_path: Path) -> list:
    """
    Read all unique reach IDs from the rating_curves table.
    These reach_ids are the NWM feature_ids.

    Returns:
        List of reach_ids (NWM feature IDs)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all unique reach_ids from rating_curves table
    cursor.execute('SELECT DISTINCT reach_id FROM rating_curves ORDER BY reach_id')
    reach_ids = [row[0] for row in cursor.fetchall()]

    conn.close()
    return reach_ids


# ==============================================================================
# S3 ACCESS FUNCTIONS
# ==============================================================================

def get_s3_client():
    """Create S3 client with appropriate configuration."""
    if USE_ANONYMOUS_S3:
        # Use unsigned requests for public bucket
        return boto3.client('s3', config=Config(signature_version=UNSIGNED))
    else:
        # Use default credentials
        return boto3.client('s3')


def list_nwm_files_for_date(s3_client, date_str: str) -> list:
    """
    List NWM files for a specific date.

    Args:
        s3_client: boto3 S3 client
        date_str: Date string in YYYY-MM-DD format

    Returns:
        List of S3 keys for NetCDF files
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    prefix, file_pattern = build_s3_prefix_and_pattern(dt, S3_BUCKET, NWM_CONFIG)

    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

        if 'Contents' not in response:
            return []

        # Filter for channel_rt (channel routing - streamflow) files
        files = [
            obj['Key'] for obj in response['Contents']
            if file_pattern in obj['Key'] and obj['Key'].endswith('.nc')
        ]

        return sorted(files)

    except Exception as e:
        print(f"  Error listing files for {date_str}: {e}", file=sys.stderr)
        return []


def load_nwm_streamflow_for_hour(s3_client, s3_key: str, feature_ids: list) -> dict:
    """
    Load streamflow data from a single NWM NetCDF file for specific feature IDs.

    Args:
        s3_client: boto3 S3 client
        s3_key: S3 key to NetCDF file
        feature_ids: List of NWM feature IDs to extract

    Returns:
        Dict mapping feature_id to streamflow value (cms)
    """
    try:
        # Open dataset directly from S3 using xarray and s3fs
        s3_path = f"s3://{S3_BUCKET}/{s3_key}"

        # Use anonymous access if needed
        storage_options = {}
        if USE_ANONYMOUS_S3:
            storage_options = {'anon': True}

        with xr.open_dataset(s3_path, engine='h5netcdf', storage_options=storage_options) as ds:
            # Get streamflow variable (typically 'streamflow' or 'qSfcLatRunoff')
            if 'streamflow' in ds.variables:
                streamflow_var = 'streamflow'
            elif 'qSfcLatRunoff' in ds.variables:
                streamflow_var = 'qSfcLatRunoff'
            else:
                # Try to find the right variable
                print(f"  Warning: streamflow variable not found in {s3_key}", file=sys.stderr)
                return {}

            # Get feature_id dimension
            if 'feature_id' in ds.variables:
                all_feature_ids = ds['feature_id'].values
            else:
                print(f"  Warning: feature_id not found in {s3_key}", file=sys.stderr)
                return {}

            # Find indices for our feature IDs
            feature_id_to_idx = {fid: idx for idx, fid in enumerate(all_feature_ids)}

            results = {}
            for fid in feature_ids:
                if fid in feature_id_to_idx:
                    idx = feature_id_to_idx[fid]
                    # Get streamflow value (convert from cms to cfs: 1 cms = 35.3147 cfs)
                    flow_cms = float(ds[streamflow_var][idx].values)
                    flow_cfs = flow_cms * 35.3147  # Convert to cubic feet per second

                    if not np.isnan(flow_cfs):
                        results[fid] = flow_cfs

            return results

    except Exception as e:
        # Silently skip files that can't be read
        # print(f"  Error loading {s3_key}: {e}", file=sys.stderr)
        return {}


# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def process_nwm_file(s3_client, nc_file: str, date_str: str, reach_ids: list,
                      reach_data: dict, bucket_type: str) -> bool:
    """
    Process a single NWM NetCDF file and store results.

    Args:
        s3_client: boto3 S3 client
        nc_file: S3 key to NetCDF file
        date_str: Date string in YYYY-MM-DD format
        reach_ids: List of reach IDs to extract
        reach_data: Dictionary to store results in
        bucket_type: 'operational' or 'retrospective'

    Returns:
        True if successful, False otherwise
    """
    try:
        filename = nc_file.split('/')[-1]
        timestamp = extract_timestamp_from_nwm_filename(filename, date_str, bucket_type)

        if timestamp is None:
            return False

        # Load streamflow data for this hour
        hour_data = load_nwm_streamflow_for_hour(s3_client, nc_file, reach_ids)

        # Store in reach_data
        for fid, flow in hour_data.items():
            reach_data[fid][timestamp] = flow

        return True

    except Exception:
        return False


def fetch_all_reach_data(reach_ids: list) -> dict:
    """
    Fetch discharge data for all reaches from NWM on S3.

    Args:
        reach_ids: List of NWM feature IDs (reach_ids from database)

    Returns:
        Dict mapping nwm_feature_id to DataFrame of hourly discharge
    """
    print("=" * 70)
    print("FETCHING NWM DATA FROM S3")
    print("=" * 70)
    print(f"\nDatabase: {RIPPLE_DB_PATH}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"S3 Bucket: s3://{S3_BUCKET}/")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Total reaches: {len(reach_ids)}")
    print(f"Configuration: {NWM_CONFIG}")
    print()

    # Create S3 client
    print("Connecting to S3...")
    s3_client = get_s3_client()

    # Generate date range
    start_dt = parse_datetime_string(START_DATE)
    end_dt = parse_datetime_string(END_DATE)
    date_range = pd.date_range(start=start_dt.date(), end=end_dt.date(), freq='D', inclusive='left')

    # Dictionary to store data: {feature_id: {timestamp: flow}}
    reach_data = {fid: {} for fid in reach_ids}
    stats = ProcessingStats()

    # Determine bucket type
    bucket_type = 'operational' if S3_BUCKET == "noaa-nwm-pds" else 'retrospective'

    print(f"Processing {len(date_range)} day(s) of data...")

    # Process each day
    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        print(f"\n  Processing {date_str}...")

        nc_files = list_nwm_files_for_date(s3_client, date_str)

        if not nc_files:
            print(f"    No files found for {date_str}")
            continue

        print(f"    Found {len(nc_files)} NetCDF files")

        # Process each hourly file
        for nc_file in nc_files:
            if process_nwm_file(s3_client, nc_file, date_str, reach_ids, reach_data, bucket_type):
                stats.files_processed += 1
                if stats.files_processed % 10 == 0:
                    print(f"    Processed {stats.files_processed} files...")
            else:
                stats.files_failed += 1

    print(f"\n  Files processed: {stats.files_processed}")
    print(f"  Files failed: {stats.files_failed}")

    # Convert to DataFrame format
    print("\nConverting to DataFrames...")
    all_data = {}

    for fid, time_series in reach_data.items():
        if time_series:
            df = pd.DataFrame.from_dict(time_series, orient='index', columns=['discharge'])
            df.index.name = 'datetime'
            df = df.sort_index()
            all_data[fid] = df
            stats.reaches_with_data += 1

    print(f"  Reaches with data: {stats.reaches_with_data}/{len(reach_ids)}")

    return all_data


def generate_hourly_timestamps() -> list:
    """
    Generate list of hourly timestamps for the event period.
    Uses the exact start and end times specified in config.
    """
    start_dt = parse_datetime_string(START_DATE)
    end_dt = parse_datetime_string(END_DATE)

    start = pd.Timestamp(start_dt, tz='UTC')
    end = pd.Timestamp(end_dt, tz='UTC')

    timestamps = pd.date_range(start=start, end=end, freq='h', inclusive='left')
    return list(timestamps)


def write_timestep_csvs(all_data: dict) -> ProcessingStats:
    """
    Write one CSV file per hourly timestep.

    Each CSV has format:
        nwm_feature_id,discharge
        3458519,1250.5
        3458691,4320.0
        ...

    Filename format: YYYYMMDD_HHMM_{suffix}.csv (suffix from config)

    Returns:
        ProcessingStats with files_written and files_skipped counts
    """
    print("\n" + "=" * 70)
    print("WRITING PER-TIMESTEP CSV FILES")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    timestamps = generate_hourly_timestamps()
    print(f"Writing {len(timestamps)} hourly files...")
    print()

    stats = ProcessingStats()

    for ts in timestamps:
        # Build data for this timestep
        rows = [
            {'nwm_feature_id': feature_id, 'discharge': df.loc[ts, 'discharge']}
            for feature_id, df in all_data.items()
            if ts in df.index and pd.notna(df.loc[ts, 'discharge'])
        ]

        if rows:
            ts_df = pd.DataFrame(rows)
            filename = ts.strftime('%Y%m%d_%H%M') + f'_{FLOW_FILE_SUFFIX}.csv'
            filepath = OUTPUT_DIR / filename

            ts_df.to_csv(filepath, index=False)
            stats.files_written += 1

            # Show progress for key timestamps
            if ts.hour in [0, 4, 5, 6, 12]:
                print(f"  ✓ {filename} ({len(rows)} gauges)")
        else:
            stats.files_skipped += 1

    print()
    print(f"Files written: {stats.files_written}")
    print(f"Files skipped (no data): {stats.files_skipped}")

    return stats


def print_sample_output(all_data: dict):
    """Print a sample of what the CSV files will contain."""
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT FORMAT")
    print("=" * 70)
    
    # Find a timestep with data
    sample_ts = None
    for feature_id, df in all_data.items():
        if not df.empty:
            sample_ts = df.index[len(df)//2]  # Middle of the data
            break
    
    if sample_ts:
        print(f"\nExample file: {sample_ts.strftime('%Y%m%d_%H%M')}_{FLOW_FILE_SUFFIX}.csv")
        print("-" * 40)
        print("nwm_feature_id,discharge")

        lines_printed = 0
        for feature_id, df in all_data.items():
            if sample_ts in df.index:
                discharge = df.loc[sample_ts, 'discharge']
                if pd.notna(discharge):
                    print(f"{feature_id},{discharge:.1f}")
                    lines_printed += 1
                    if lines_printed >= 10:
                        print("...")
                        break

        print("-" * 40)


def check_existing_flow_files() -> tuple[bool, int, int]:
    """
    Check if flow files already exist for the configured time range.

    Returns:
        Tuple of (all_exist, existing_count, expected_count)
    """
    # Generate expected timestamps
    start_dt = parse_datetime_string(START_DATE)
    end_dt = parse_datetime_string(END_DATE)

    start = pd.Timestamp(start_dt, tz='UTC')
    end = pd.Timestamp(end_dt, tz='UTC')

    timestamps = pd.date_range(start=start, end=end, freq='h', inclusive='left')
    expected_count = len(timestamps)

    if expected_count == 0:
        return False, 0, 0

    # Check if output directory exists
    if not OUTPUT_DIR.exists():
        return False, 0, expected_count

    # Check for existing files
    existing_count = 0
    for ts in timestamps:
        filename = ts.strftime('%Y%m%d_%H%M') + f'_{FLOW_FILE_SUFFIX}.csv'
        filepath = OUTPUT_DIR / filename
        if filepath.exists():
            existing_count += 1

    all_exist = (existing_count == expected_count)
    return all_exist, existing_count, expected_count


def print_database_info(reach_ids: list):
    """Print information about the rating curve database."""
    print("\n" + "=" * 70)
    print("RATING CURVE DATABASE INFO")
    print("=" * 70)
    print(f"\nDatabase path: {RIPPLE_DB_PATH}")
    print(f"Total reaches: {len(reach_ids)}")
    print(f"\nReach ID range: {min(reach_ids)} to {max(reach_ids)}")
    print(f"\nSample reach IDs:")
    for reach_id in reach_ids[:10]:
        print(f"  {reach_id}")
    if len(reach_ids) > 10:
        print(f"  ... and {len(reach_ids) - 10} more")
    print()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Check if database exists
    if not RIPPLE_DB_PATH.exists():
        print(f"\n ERROR: Database file not found: {RIPPLE_DB_PATH}")
        print("  Please update RIPPLE_DB_PATH in the configuration section.")
        return 1

    # Check if flow files already exist (unless --force flag is used)
    if not args.force:
        all_exist, existing_count, expected_count = check_existing_flow_files()

        if all_exist:
            print("\n" + "=" * 70)
            print("FLOW FILES ALREADY EXIST")
            print("=" * 70)
            print(f"\nAll {expected_count} flow files already exist for the configured time range:")
            print(f"  Start: {START_DATE}")
            print(f"  End: {END_DATE}")
            print(f"  Location: {OUTPUT_DIR}")
            print("\nSkipping flow file generation.")
            print("\nTo force regeneration, use: --force flag")
            print("Example: python generate_flow_files.py --config config.yaml --force")
            return 0
        elif existing_count > 0:
            print(f"\n⚠ Warning: Found {existing_count}/{expected_count} existing flow files.")
            print("  Will regenerate all files to ensure consistency.")

    # Read all reach IDs from database
    print("Reading reach IDs from database...")
    reach_ids = get_reach_ids_from_database(RIPPLE_DB_PATH)

    if not reach_ids:
        print(f"\n ERROR: No reach IDs found in database: {RIPPLE_DB_PATH}")
        return 1

    # Print database info
    print_database_info(reach_ids)

    # Fetch all reach data
    all_data = fetch_all_reach_data(reach_ids)

    if not all_data:
        print("\n⚠ WARNING: No data was retrieved from NWM AnA APIs.")
        print("  This could mean:")
        print("    - No data available for the specified time period")
        print("    - API connectivity issues")
        print("    - All reaches failed to return data")
        return 1

    # Show sample output format
    print_sample_output(all_data)

    # Write per-timestep CSV files
    write_stats = write_timestep_csvs(all_data)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nOutput location: {OUTPUT_DIR}")
    print(f"Files created: {write_stats.files_written}")
    print(f"Reaches with data: {len(all_data)}")
    print("\nFile format:")
    print("  Header: nwm_feature_id,discharge")
    print("  Values: NWM feature ID (reach_id), discharge in cfs")
    print("\nNext steps:")
    print("  Use flows2fim to generate flood inundation maps:")
    print("  1. flows2fim controls -db <ripple.gpkg> -f <flow.csv> -o <controls.csv> -sids <reach_ids>")
    print("  2. flows2fim fim -c <controls.csv> -lib <fim_library> -o <output.tif> -type depth")

    return 0


if __name__ == "__main__":
    sys.exit(main())