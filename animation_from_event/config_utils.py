#!/usr/bin/env python3
"""
Configuration Utilities
========================

Helper functions for loading and accessing configuration.
"""

import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_config(config_path='config.yaml'):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_s3_paths(config):
    """Construct S3 paths using collection ID template."""
    collection_id = config['collection']['id']
    bucket = config['collection']['s3']['bucket']
    base_path = config['collection']['s3']['base_path']

    return {
        'bucket': bucket,
        'ripple_path': f"{base_path}/{collection_id}/ripple.gpkg",
        'start_reaches_path': f"{base_path}/{collection_id}/start_reaches.csv",
        'fim_library': f"/vsis3/{bucket}/{base_path}/{collection_id}/library_extent"
    }

def get_paths(config):
    """Get standardized paths from config."""
    base_dir = Path(config['output']['base_dir'])

    # Handle starting reaches configuration
    starting_reaches = config['fim'].get('starting_reaches', '/data/input/start_reaches.csv')

    # If it's "auto" or a comma-separated list, keep as string
    if starting_reaches == "auto" or (',' in str(starting_reaches) and not str(starting_reaches).endswith('.csv')):
        start_reaches_value = starting_reaches
    else:
        # Convert to Path, but check if file exists
        reaches_path = Path(starting_reaches)
        # If configured CSV file doesn't exist, fall back to auto-detection
        if not reaches_path.exists():
            start_reaches_value = "auto"
        else:
            start_reaches_value = reaches_path

    return {
        'base_dir': base_dir,
        'ripple_db': Path('/data/input/ripple.gpkg'),
        'start_reaches': start_reaches_value,
        'flows_dir': base_dir / 'flows',
        'controls_dir': base_dir / 'controls',
        'fims_dir': base_dir / 'fims',
        'output_video': base_dir / config['output']['video_filename'],
        'flow_file_suffix': config['output'].get('flow_file_suffix', 'flows'),
    }

def get_nwm_config(config):
    """Get NWM configuration."""
    return {
        'bucket': config['event']['nwm']['bucket'],
        'config': config['event']['nwm']['configuration'],
        'use_anonymous': config['event']['nwm']['use_anonymous'],
        'start_date': config['event']['start_date'],
        'end_date': config['event']['end_date'],
    }

def get_fim_config(config):
    """Get FIM generation configuration."""
    s3_paths = get_s3_paths(config)

    return {
        'library': s3_paths['fim_library'],
        'type': config['fim']['type'],
        'format': config['fim']['output_format'],
        'boundary_condition': config['fim']['boundary_condition'],
    }

def get_animation_config(config):
    """Get animation configuration."""
    anim_cfg = config['animation']

    return {
        'extent': anim_cfg['extent'],
        'visual': anim_cfg['visual'],
        'overlay': anim_cfg['overlay'],
        'basemap': anim_cfg['basemap'],
        'county': anim_cfg['county'],
        'lake_fill': anim_cfg['lake_fill'],
    }

def setup_aws_env():
    """Ensure AWS environment variables are set."""
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']

    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"Warning: Missing AWS environment variables: {', '.join(missing)}")
        print("Set these in .env file or use anonymous S3 access")

    return len(missing) == 0
