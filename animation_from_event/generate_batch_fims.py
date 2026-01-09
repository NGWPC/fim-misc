#!/usr/bin/env python3
"""
Batch Generate Flood Inundation Maps from Flow Files
=====================================================

Automates flows2fim workflow to generate FIMs for all timestep flow files.

For each flow CSV file in a time range specified in config.yaml:
  1. Runs flows2fim controls to create controls file (skips if exists)
  2. Runs flows2fim fim to generate flood inundation map (skips if exists)

The script intelligently checks for existing files and skips regeneration
unless the --force flag is used. This saves significant time when re-running
after interruptions or when only wanting to regenerate animations.

Usage:
    python generate_batch_fims.py [--config config.yaml] [--force]

    --config    Path to configuration file (default: config.yaml)
    --force     Force regeneration of existing files
"""

import subprocess
from pathlib import Path
import sys
import sqlite3
import argparse
from dataclasses import dataclass
from config_utils import load_config, get_paths, get_fim_config

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Load configuration
parser = argparse.ArgumentParser(description="Batch generate FIMs from flow files")
parser.add_argument('--config', default='config.yaml', help="Path to config file")
parser.add_argument('--max-workers', type=int, help="Override max parallel workers")
parser.add_argument('--force', action='store_true', help="Force regeneration of existing files")
args = parser.parse_args()

config = load_config(args.config)
paths = get_paths(config)
fim_cfg = get_fim_config(config)

# Set paths from config
RIPPLE_DB_PATH = paths['ripple_db']
FLOW_FILES_DIR = paths['flows_dir']
CONTROLS_DIR = paths['controls_dir']
FIMS_DIR = paths['fims_dir']
STARTING_REACH_IDS = str(paths['start_reaches'])

# FIM settings from config
FIM_LIBRARY = fim_cfg['library']
FIM_TYPE = fim_cfg['type']
OUTPUT_FORMAT = fim_cfg['format']
DEFAULT_BOUNDARY_CONDITION = fim_cfg['boundary_condition']

# flows2fim executable path
FLOWS2FIM_EXECUTABLE = config['processing'].get('flows2fim_executable', 'flows2fim')


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

@dataclass
class ProcessingStats:
    """Track processing statistics."""
    controls_success: int = 0
    controls_skipped: int = 0
    fim_success: int = 0
    fim_skipped: int = 0

    @property
    def total_success(self) -> int:
        """Total successfully completed FIMs."""
        return self.fim_success + self.fim_skipped


def get_fim_extension(output_format: str) -> str:
    """
    Get file extension for FIM output based on format.

    Args:
        output_format: 'VRT', 'COG', or 'GTIFF'

    Returns:
        File extension including dot (e.g., '.vrt', '.tif')
    """
    return '.vrt' if output_format == 'VRT' else '.tif'


def run_subprocess(cmd: list, error_prefix: str) -> bool:
    """
    Run subprocess command with standardized error handling.

    Args:
        cmd: Command list to execute
        error_prefix: Prefix for error messages

    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR {error_prefix}: {e.stderr}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"  ERROR: flows2fim executable not found: {cmd[0]}", file=sys.stderr)
        print(f"  Make sure flows2fim is installed and in your PATH", file=sys.stderr)
        return False


def get_upstream_reaches(db_path: Path, max_reaches: int = 10) -> list:
    """
    Find upstream-most reaches (reaches with no upstream connections).
    These are good candidates for starting reach IDs.

    Args:
        db_path: Path to ripple.gpkg database
        max_reaches: Maximum number of starting reaches to return

    Returns:
        List of reach IDs
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Find reaches that are not listed as downstream (nwm_to_id) of any other reach
    # These are the headwater reaches
    query = """
    SELECT DISTINCT r.reach_id
    FROM reaches r
    LEFT JOIN reaches r2 ON r.reach_id = r2.nwm_to_id
    WHERE r2.reach_id IS NULL
    ORDER BY r.reach_id
    LIMIT ?
    """

    cursor.execute(query, (max_reaches,))
    reach_ids = [row[0] for row in cursor.fetchall()]

    conn.close()
    return reach_ids


def get_all_flow_files(flow_dir: Path) -> list:
    """Get all flow CSV files sorted by timestamp."""
    flow_files = sorted(flow_dir.glob("*.csv"))
    return flow_files


def run_flows2fim_controls(flow_file: Path, controls_file: Path,
                           starting_ids: str, boundary_condition: str = "nd") -> bool:
    """
    Run flows2fim controls command.

    Args:
        flow_file: Path to input flow CSV
        controls_file: Path to output controls CSV
        starting_ids: Comma-separated reach IDs or path to CSV
        boundary_condition: 'nd' or 'kwse'

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        FLOWS2FIM_EXECUTABLE,
        "controls",
        "-db", str(RIPPLE_DB_PATH),
        "-f", str(flow_file),
        "-o", str(controls_file),
    ]

    # Add starting reach IDs
    if starting_ids.endswith('.csv'):
        cmd.extend(["-scsv", starting_ids])
    else:
        cmd.extend(["-sids", starting_ids, "-scs", boundary_condition])

    return run_subprocess(cmd, "running controls")


def run_flows2fim_fim(controls_file: Path, output_fim: Path,
                      fim_library: str, fim_type: str = "depth",
                      output_format: str = "VRT") -> bool:
    """
    Run flows2fim fim command.

    Args:
        controls_file: Path to input controls CSV
        output_fim: Path to output FIM file
        fim_library: Path to FIM library (local or S3)
        fim_type: 'depth' or 'extent'
        output_format: 'VRT', 'COG', or 'GTIFF'

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        FLOWS2FIM_EXECUTABLE,
        "fim",
        "-c", str(controls_file),
        "-lib", fim_library,
        "-o", str(output_fim),
        "-type", fim_type,
        "-fmt", output_format,
    ]

    return run_subprocess(cmd, "running fim")


def extract_timestamp_from_filename(filename: str) -> str:
    """
    Extract timestamp from flow filename.
    Format: YYYYMMDD_HHMM_*.csv -> YYYYMMDD_HHMM
    """
    parts = filename.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return filename.replace('.csv', '')


def process_flow_file(flow_file: Path, timestamp: str, starting_ids: str,
                      force: bool, stats: ProcessingStats) -> None:
    """
    Process a single flow file to generate controls and FIM.

    Args:
        flow_file: Path to flow CSV file
        timestamp: Extracted timestamp string
        starting_ids: Starting reach IDs configuration
        force: Force regeneration of existing files
        stats: ProcessingStats object to update
    """
    # Define output paths
    controls_file = CONTROLS_DIR / f"{timestamp}_controls.csv"
    fim_extension = get_fim_extension(OUTPUT_FORMAT)
    fim_file = FIMS_DIR / f"{timestamp}_{FIM_TYPE}{fim_extension}"

    # Step 1: Generate controls
    controls_generated = False
    if controls_file.exists() and not force:
        stats.controls_skipped += 1
        controls_generated = True
    else:
        action = "Regenerating" if controls_file.exists() else "Generating"
        print(f"{timestamp}: {action} controls...")
        if run_flows2fim_controls(flow_file, controls_file, starting_ids, DEFAULT_BOUNDARY_CONDITION):
            stats.controls_success += 1
            controls_generated = True
        else:
            print(f"  ✗ Failed to generate controls")

    # Step 2: Generate FIM (only if controls exist)
    if controls_generated:
        if fim_file.exists() and not force:
            stats.fim_skipped += 1
        else:
            action = "Regenerating" if fim_file.exists() else "Generating"
            print(f"{timestamp}: {action} FIM...")
            if run_flows2fim_fim(controls_file, fim_file, FIM_LIBRARY, FIM_TYPE, OUTPUT_FORMAT):
                stats.fim_success += 1
            else:
                print(f"  ✗ Failed to generate FIM")


# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def main():
    print("=" * 70)
    print("BATCH FIM GENERATION")
    print("=" * 70)
    print()

    # Validate paths
    if not RIPPLE_DB_PATH.exists():
        print(f" ERROR: Database not found: {RIPPLE_DB_PATH}")
        return 1

    if not FLOW_FILES_DIR.exists():
        print(f" ERROR: Flow files directory not found: {FLOW_FILES_DIR}")
        return 1

    # Create output directories
    CONTROLS_DIR.mkdir(parents=True, exist_ok=True)
    FIMS_DIR.mkdir(parents=True, exist_ok=True)

    # Get starting reach IDs
    if STARTING_REACH_IDS == "auto":
        print("Auto-detecting upstream starting reaches...")
        upstream_reaches = get_upstream_reaches(RIPPLE_DB_PATH)
        if not upstream_reaches:
            print(" ERROR: Could not detect upstream reaches")
            print("  Please set STARTING_REACH_IDS manually in the configuration")
            return 1
        starting_ids = ",".join(str(r) for r in upstream_reaches)
        print(f"  Found {len(upstream_reaches)} upstream reaches: {starting_ids}")
    else:
        starting_ids = STARTING_REACH_IDS
        print(f"Using starting reach IDs: {starting_ids}")

    print()
    print(f"Configuration:")
    print(f"  Database: {RIPPLE_DB_PATH}")
    print(f"  Flow files: {FLOW_FILES_DIR}")
    print(f"  Controls output: {CONTROLS_DIR}")
    print(f"  FIMs output: {FIMS_DIR}")
    print(f"  FIM library: {FIM_LIBRARY}")
    print(f"  FIM type: {FIM_TYPE}")
    print(f"  Output format: {OUTPUT_FORMAT}")
    print()

    # Get all flow files
    flow_files = get_all_flow_files(FLOW_FILES_DIR)

    if not flow_files:
        print(f" ERROR: No flow CSV files found in {FLOW_FILES_DIR}")
        return 1

    print(f"Found {len(flow_files)} flow file(s) to process")
    print()

    # Process each flow file
    stats = ProcessingStats()

    for flow_file in flow_files:
        timestamp = extract_timestamp_from_filename(flow_file.name)
        process_flow_file(flow_file, timestamp, starting_ids, args.force, stats)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total flow files: {len(flow_files)}")
    print(f"Controls generated: {stats.controls_success}")
    print(f"Controls skipped: {stats.controls_skipped} (already existed)")
    print(f"FIMs generated: {stats.fim_success}")
    print(f"FIMs skipped: {stats.fim_skipped} (already existed)")
    print(f"Complete: {stats.total_success}")
    print()

    if args.force:
        print("Note: Ran with --force flag (regenerated existing files)")
        print()

    if stats.total_success == len(flow_files):
        print("✓ All FIMs complete!")
        print(f"\nOutput location: {FIMS_DIR}")
        if stats.controls_skipped > 0 or stats.fim_skipped > 0:
            print(f"\nTip: Use --force flag to regenerate existing files")
        return 0
    elif stats.total_success > 0:
        print(f" {len(flow_files) - stats.total_success} FIM(s) failed")
        return 1
    else:
        print(" No FIMs were generated")
        return 1


if __name__ == "__main__":
    sys.exit(main())
