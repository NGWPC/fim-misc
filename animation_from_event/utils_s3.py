#!/usr/bin/env python3
"""
S3 Utility Functions
====================

Handles downloading RIPPLE collection data from S3.

Usage:
    python utils_s3.py --config config.yaml --download-ripple
"""

import argparse
import boto3
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from config_utils import load_config, get_s3_paths

# Load environment variables
load_dotenv()

def get_s3_client():
    """Create S3 client with credentials from environment."""
    return boto3.client('s3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    )

def download_from_s3(s3_client, bucket, key, local_path):
    """Download file from S3."""
    print(f"  Downloading: s3://{bucket}/{key}")
    print(f"  To: {local_path}")

    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        s3_client.download_file(bucket, key, str(local_path))
        file_size = local_path.stat().st_size / (1024 * 1024)
        print(f"   Downloaded {file_size:.1f} MB")
        return True
    except Exception as e:
        print(f"   Error: {e}")
        return False

def download_ripple_data(config):
    """Download RIPPLE database and start reaches from S3."""
    print("\n" + "=" * 70)
    print("DOWNLOADING RIPPLE DATA FROM S3")
    print("=" * 70)

    # Get dynamically constructed S3 paths
    s3_paths = get_s3_paths(config)
    collection_id = config['collection']['id']

    print(f"\nCollection: {collection_id}")
    print(f"Bucket: {s3_paths['bucket']}")

    s3_client = get_s3_client()

    # Download ripple.gpkg
    print("\n1. Downloading ripple.gpkg...")
    ripple_key = s3_paths['ripple_path']
    ripple_local = Path('/data/input/ripple.gpkg')

    if ripple_local.exists():
        print(f"   File already exists: {ripple_local}")
    else:
        if not download_from_s3(s3_client, s3_paths['bucket'], ripple_key, ripple_local):
            return False

    # Download start_reaches.csv (optional - will auto-detect if missing)
    print("\n2. Downloading start_reaches.csv...")
    reaches_key = s3_paths['start_reaches_path']
    reaches_local = Path('/data/input/start_reaches.csv')

    if reaches_local.exists():
        print(f"   File already exists: {reaches_local}")
    else:
        if not download_from_s3(s3_client, s3_paths['bucket'], reaches_key, reaches_local):
            print(f"   Warning: start_reaches.csv not found in S3")
            print(f"   Will use auto-detection of upstream reaches instead")

    print("\n RIPPLE data ready")
    return True

def main():
    parser = argparse.ArgumentParser(description="S3 utility functions")
    parser.add_argument('--config', required=True, help="Path to config.yaml")
    parser.add_argument('--download-ripple', action='store_true', help="Download RIPPLE data")

    args = parser.parse_args()

    config = load_config(args.config)

    if args.download_ripple:
        success = download_ripple_data(config)
        return 0 if success else 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
