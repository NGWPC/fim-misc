
import os

import argparse
import pystac
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, mapping
from pyproj import CRS, transform
from tqdm import tqdm

def main(
    catalog_file_path: str, desired_crs: str | CRS = "EPSG:5070", output_file: str = "stac_bounding_boxes.gpkg"
):
    
        # 1. Load the STAC Catalog or Item
    catalog = pystac.Catalog.from_file(catalog_file_path)  # or ItemCollection

    # 2. Filter out GFM-related collections
    filtered_collections = [
        coll for coll in catalog.get_collections() 
        if "gfm" not in coll.id.lower()  # Exclude 'gfm', 'GFM', etc.
    ]

    # filter out hwm-related collections
    # hwm bounding areas are very large
    filtered_collections = [
        coll for coll in filtered_collections
        if "hwm" not in coll.id.lower()  # Exclude 'hwm', 'HWM', etc.
    ]


    all_items = []
    for collection in filtered_collections:
        # Load items from each collection
        items = collection.get_items()
        all_items.extend(items)

    # 2. Extract Bounding Boxes and Metadata
    features = []; failed_crs = 0
    for item in all_items:
        # Get the item's bounding box in the native CRS (usually WGS84)
        bbox = item.bbox  # [minx, miny, maxx, maxy]
        
        # Convert bbox to a Shapely Polygon
        bbox_geom = box(*bbox)

        # Get the CRS (default to EPSG:4326 if not specified)
        #crs = item.properties.get("proj:epsg", 4326)  # STAC common metadata
        crs = item.properties.get("proj:wkt2", 4326)  # Default to EPSG:4326 if not specified

        properties = item.properties.copy()
        properties.update({"geometry": bbox_geom, "id": item.id, 'collection_id' : item.collection_id})
        properties = {k: [v] for k, v in properties.items()}  # Ensure all values are lists

        # what's going on here?
        try:
            feature = gpd.GeoDataFrame(
                properties,
                #crs=crs  # Use the CRS from the item properties
                crs=4326 # TEMP
            ).to_crs(desired_crs)
        except Exception as e:
            feature = gpd.GeoDataFrame(
                properties,
                crs=4326
            )
            failed_crs += 1
            print(f"Failed to convert CRS for item {item.id}: {e}")
            pass
        
        # Append to features list
        features.append(feature)


    print(f"Failed to convert CRS for {failed_crs} items out of {len(all_items)} total items.")

    # 3. Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(pd.concat(features, ignore_index=True), geometry="geometry", crs=desired_crs)

    if os.path.exists(output_file):
        os.remove(output_file)  # Remove existing file if it exists

    # 5. Save to GeoPackage
    #gdf.to_file("stac_bounding_boxes.gpkg", layer="items", driver="GPKG")
    gdf.to_file(output_file, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract bounding boxes from STAC Catalog or Item.")
    parser.add_argument(
        '-f', "--catalog_file_path", type=str, help="Path to the STAC Catalog or Item JSON file.",
        required=False,
        default=os.path.join(os.path.expanduser('~'), 'data', 'foss_fim', 'misc', 'resolution_analysis', 'static_cat', 'catalog.json')
    )
    parser.add_argument(
        '-c', "--desired_crs", type=str, default="EPSG:5070", help="Desired CRS for the output bounding boxes."
    )
    parser.add_argument(
        '-o', "--output_file",
        type=str,
        default=os.path.join(os.path.expanduser('~'), 'data', 'foss_fim', 'misc', 'resolution_analysis', 'testing_domain_geom.gpkg'),

        help="Output GeoPackage file name."
    )

    # example: python get_stac_bounding_boxes.py -f ~/data/foss_fim/misc/resolution_analysis/static_cat/catalog.json -c EPSG:5070 -o ~/data/foss_fim/misc/resolution_analysis/testing_domain_geom.gpkg

    main(**vars(parser.parse_args()))