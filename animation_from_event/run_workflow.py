#!/usr/bin/env python3
"""
Flood Animation Workflow Orchestrator
======================================

Runs the complete workflow:
1. Download RIPPLE data from S3
2. Generate hourly flow files from NWM
3. Generate FIMs using flows2fim
4. Create flood animation video

Usage:
    python run_workflow.py --config config.yaml
    python run_workflow.py --config config.yaml --skip-flows --skip-fims
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time
from config_utils import load_config

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()

    start_time = time.time()

    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n {description} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n {description} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run complete flood animation workflow")
    parser.add_argument('--config', required=True, help="Path to config.yaml")
    parser.add_argument('--skip-download', action='store_true', help="Skip S3 download step")
    parser.add_argument('--skip-flows', action='store_true', help="Skip flow file generation")
    parser.add_argument('--skip-fims', action='store_true', help="Skip FIM generation")
    parser.add_argument('--skip-animation', action='store_true', help="Skip animation creation")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    print("=" * 70)
    print("FLOOD ANIMATION WORKFLOW")
    print("=" * 70)
    print(f"\nCollection: {config['collection']['id']}")
    print(f"Event: {config['event']['start_date']} to {config['event']['end_date']}")
    print()

    steps_run = 0
    steps_failed = 0

    # Step 1: Download RIPPLE data from S3
    if not args.skip_download:
        if run_command(
            ['python', 'utils_s3.py', '--config', args.config, '--download-ripple'],
            "Step 1: Download RIPPLE data from S3"
        ):
            steps_run += 1
        else:
            steps_failed += 1
            print(" Continuing with existing local files...")
    else:
        print("\n Skipping S3 download (using existing files)")

    # Step 2: Generate flow files
    if not args.skip_flows:
        if run_command(
            ['python', 'generate_flow_files.py', '--config', args.config],
            "Step 2: Generate hourly flow files from NWM"
        ):
            steps_run += 1
        else:
            steps_failed += 1
            print(" Cannot continue without flow files")
            return 1
    else:
        print("\n Skipping flow file generation")

    # Step 3: Generate FIMs
    if not args.skip_fims:
        if run_command(
            ['python', 'generate_batch_fims.py', '--config', args.config],
            "Step 3: Generate flood inundation maps"
        ):
            steps_run += 1
        else:
            steps_failed += 1
            print(" Cannot continue without FIMs")
            return 1
    else:
        print("\n Skipping FIM generation")

    # Step 4: Create animation
    if not args.skip_animation:
        if run_command(
            ['python', 'generate_animation.py', '--config', args.config],
            "Step 4: Create flood animation video"
        ):
            steps_run += 1
        else:
            steps_failed += 1
    else:
        print("\n Skipping animation creation")

    # Summary
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\nSteps completed: {steps_run}")
    print(f"Steps failed: {steps_failed}")

    if steps_failed == 0:
        output_file = Path(config['output']['base_dir']) / config['output']['video_filename']
        print(f"\n Animation ready: {output_file}")
        return 0
    else:
        print(f"\n Workflow completed with {steps_failed} error(s)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
