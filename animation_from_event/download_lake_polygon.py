#!/usr/bin/env python3
"""
Download Lake Polygon from OpenStreetMap
=========================================

Generic script to download any lake/reservoir polygon from OpenStreetMap
and save as GeoJSON for use in flood animation lake fill.

This script queries the OpenStreetMap Overpass API for water bodies matching
a given name within a specified bounding box, then saves the results as GeoJSON.

Requirements:
    pip install requests

Usage:
    # Using command-line arguments
    python download_lake_polygon.py --name "Lake Ingram" --bbox 29.9,-99.5,30.3,-99.0 --output lake_ingram.geojson

    # Using config file
    python download_lake_polygon.py --config config.yaml --lake-name "Lake Mead"

    # Interactive mode (prompts for inputs)
    python download_lake_polygon.py --interactive

Examples:
    # Lake Mead, Nevada/Arizona
    python download_lake_polygon.py --name "Lake Mead" --bbox 36.0,-114.8,36.3,-114.4 --output lake_mead.geojson

    # Lake Travis, Texas
    python download_lake_polygon.py --name "Lake Travis" --bbox 30.3,-98.1,30.5,-97.9 --output lake_travis.geojson

    # Reservoir with partial name match
    python download_lake_polygon.py --name "Ingram" --bbox 29.9,-99.5,30.3,-99.0 --output lake_ingram.geojson
"""

import requests
import json
import argparse
import sys
from pathlib import Path


def query_overpass(lake_name, bbox, timeout=25):
    """
    Query OpenStreetMap Overpass API for water bodies matching the lake name.

    Args:
        lake_name: Name of lake to search for (case-insensitive, partial match)
        bbox: Tuple of (south, west, north, east) in degrees
        timeout: Query timeout in seconds

    Returns:
        JSON response from Overpass API

    Raises:
        requests.exceptions.RequestException: If query fails
    """
    south, west, north, east = bbox

    # Build Overpass QL query
    # Searches for ways and relations with natural=water, water=lake, or water=reservoir
    # that match the lake name (case-insensitive)
    query = f"""
[out:json][timeout:{timeout}];
(
  way["natural"="water"]["name"~"{lake_name}",i]({south},{west},{north},{east});
  relation["natural"="water"]["name"~"{lake_name}",i]({south},{west},{north},{east});
  way["water"="lake"]["name"~"{lake_name}",i]({south},{west},{north},{east});
  relation["water"="lake"]["name"~"{lake_name}",i]({south},{west},{north},{east});
  way["water"="reservoir"]["name"~"{lake_name}",i]({south},{west},{north},{east});
  relation["water"="reservoir"]["name"~"{lake_name}",i]({south},{west},{north},{east});
);
out geom;
"""

    overpass_url = "http://overpass-api.de/api/interpreter"

    print(f"Querying OpenStreetMap for '{lake_name}'...")
    print(f"Search area: ({south}, {west}) to ({north}, {east})")
    print()

    response = requests.post(overpass_url, data=query, timeout=60)
    response.raise_for_status()

    return response.json()


def osm_to_geojson(osm_data):
    """
    Convert OSM data to GeoJSON FeatureCollection format.

    Args:
        osm_data: OSM JSON response from Overpass API

    Returns:
        GeoJSON FeatureCollection dictionary
    """
    features = []

    for element in osm_data.get('elements', []):
        if element['type'] == 'way' and 'geometry' in element:
            # Convert way to polygon
            coords = [[node['lon'], node['lat']] for node in element['geometry']]

            # Close the polygon if needed
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [coords]
                },
                'properties': {
                    'osm_id': element.get('id'),
                    'name': element.get('tags', {}).get('name', 'Unknown'),
                    'type': element.get('tags', {}).get('water', element.get('tags', {}).get('natural', 'water')),
                    'source': 'OpenStreetMap',
                    'osm_type': 'way'
                }
            }
            features.append(feature)

        elif element['type'] == 'relation' and 'members' in element:
            # Handle multipolygon relations
            # Note: Full multipolygon processing is complex, this is simplified
            print(f"  Note: Found relation '{element.get('tags', {}).get('name', 'Unknown')}' (ID: {element.get('id')})")
            print(f"        Relations require more complex processing. Extracting outer members only.")

            # Try to extract outer way geometries
            outer_coords = []
            for member in element.get('members', []):
                if member.get('role') == 'outer' and 'geometry' in member:
                    coords = [[node['lon'], node['lat']] for node in member['geometry']]
                    outer_coords.extend(coords)

            if outer_coords:
                # Close the polygon
                if outer_coords[0] != outer_coords[-1]:
                    outer_coords.append(outer_coords[0])

                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [outer_coords]
                    },
                    'properties': {
                        'osm_id': element.get('id'),
                        'name': element.get('tags', {}).get('name', 'Unknown'),
                        'type': element.get('tags', {}).get('water', element.get('tags', {}).get('natural', 'water')),
                        'source': 'OpenStreetMap',
                        'osm_type': 'relation'
                    }
                }
                features.append(feature)

    return {
        'type': 'FeatureCollection',
        'features': features
    }


def interactive_mode():
    """
    Interactive mode - prompts user for inputs.

    Returns:
        Tuple of (lake_name, bbox, output_path)
    """
    print("=" * 70)
    print("INTERACTIVE LAKE POLYGON DOWNLOAD")
    print("=" * 70)
    print()
    print("This tool will download lake polygons from OpenStreetMap.")
    print()

    # Get lake name
    lake_name = input("Enter lake name (e.g., 'Ingram'): ").strip()

    # Get bounding box
    print()
    print("Enter bounding box coordinates (in decimal degrees):")
    print("  Tip: Use https://boundingbox.klokantech.com/ to find coordinates")
    print("       Select 'CSV' format and copy the values")
    print()

    south = float(input("  South latitude (e.g., 29.9): "))
    west = float(input("  West longitude (e.g., -99.5): "))
    north = float(input("  North latitude (e.g., 30.3): "))
    east = float(input("  East longitude (e.g., -99.0): "))

    bbox = (south, west, north, east)

    # Get output path
    print()
    default_output = f"{lake_name.lower().replace(' ', '_')}.geojson"
    output_str = input(f"Output file path (default: {default_output}): ").strip()
    output_filename = output_str if output_str else default_output

    # Ensure file is saved to mounted volume (/data/input)
    output_path = Path(output_filename)
    if not output_path.is_absolute():
        output_path = Path('/data/input') / output_path

    return lake_name, bbox, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Download lake polygon from OpenStreetMap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --name "Lake Mead" --bbox 36.0,-114.8,36.3,-114.4 --output lake_mead.geojson
  %(prog)s --name "Ingram" --bbox 29.9,-99.5,30.3,-99.0 --output lake_ingram.geojson
  %(prog)s --interactive
        """
    )

    parser.add_argument('--name', '-n', help="Lake name to search for")
    parser.add_argument('--bbox', '-b', help="Bounding box: south,west,north,east (degrees)")
    parser.add_argument('--output', '-o', help="Output GeoJSON file path")
    parser.add_argument('--interactive', '-i', action='store_true',
                        help="Interactive mode (prompts for inputs)")
    parser.add_argument('--timeout', type=int, default=25,
                        help="Query timeout in seconds (default: 25)")

    args = parser.parse_args()

    # Interactive mode
    if args.interactive:
        lake_name, bbox, output_path = interactive_mode()
    else:
        # Validate required arguments
        if not args.name or not args.bbox or not args.output:
            parser.error("--name, --bbox, and --output are required (or use --interactive)")

        lake_name = args.name

        # Parse bounding box
        try:
            bbox_parts = [float(x.strip()) for x in args.bbox.split(',')]
            if len(bbox_parts) != 4:
                raise ValueError("Bounding box must have 4 values")
            bbox = tuple(bbox_parts)
        except ValueError as e:
            parser.error(f"Invalid bounding box format: {e}")

        # Ensure file is saved to mounted volume (/data/input) if relative path
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path('/data/input') / output_path

    # Display configuration
    print()
    print("=" * 70)
    print("DOWNLOAD LAKE POLYGON FROM OPENSTREETMAP")
    print("=" * 70)
    print(f"Lake name: {lake_name}")
    print(f"Bounding box: {bbox}")
    print(f"Output: {output_path}")
    print()

    try:
        # Query Overpass API
        osm_data = query_overpass(lake_name, bbox, timeout=args.timeout)

        # Check results
        if not osm_data.get('elements'):
            print("=" * 70)
            print("NO RESULTS FOUND")
            print("=" * 70)
            print(f"No water bodies matching '{lake_name}' found in the specified area.")
            print()
            print("Suggestions:")
            print("  1. Try a broader search area (larger bounding box)")
            print("  2. Try a partial name (e.g., 'Mead' instead of 'Lake Mead')")
            print("  3. Check the lake name spelling on OpenStreetMap.org")
            print("  4. The lake may not be mapped in OpenStreetMap")
            print()
            print("Alternative data sources:")
            print("  - USGS National Map: https://apps.nationalmap.gov/downloader/")
            print("  - NHD: https://www.usgs.gov/national-hydrography")
            return 1

        # Print found features
        print("=" * 70)
        print(f"FOUND {len(osm_data['elements'])} FEATURE(S)")
        print("=" * 70)
        for elem in osm_data['elements']:
            name = elem.get('tags', {}).get('name', 'Unnamed')
            osm_type = elem.get('type', 'unknown')
            osm_id = elem.get('id', 'N/A')
            water_type = elem.get('tags', {}).get('water', elem.get('tags', {}).get('natural', 'N/A'))
            print(f"  • {name}")
            print(f"    Type: {osm_type} | ID: {osm_id} | Water: {water_type}")
        print()

        # Convert to GeoJSON
        geojson = osm_to_geojson(osm_data)

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        print("=" * 70)
        print("SUCCESS")
        print("=" * 70)
        print(f"Saved {len(geojson['features'])} feature(s) to: {output_path}")
        print()
        print("Next steps:")
        print("  1. Verify the lake polygon in QGIS or a GeoJSON viewer")
        print("  2. Add to config.yaml under animation.lake_fill:")
        print(f"     enabled: true")
        print(f"     file_path: \"/data/input/{output_path.name}\"")
        print("  3. Run the animation workflow")
        print()

        return 0

    except requests.exceptions.Timeout:
        print("ERROR: Query timed out. Try:")
        print("  - Smaller bounding box")
        print("  - Increase timeout with --timeout <seconds>")
        return 1

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to query OpenStreetMap: {e}")
        print()
        print("Possible issues:")
        print("  - No internet connection")
        print("  - Overpass API is down or overloaded")
        print("  - Invalid bounding box coordinates")
        print()
        print("Try alternative data sources:")
        print("  - USGS National Map")
        print("  - National Hydrography Dataset (NHD)")
        return 1

    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
