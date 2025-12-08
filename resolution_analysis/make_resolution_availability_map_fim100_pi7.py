"""
Make resolution availability map. Maps available DEM resolution by HUC8s and overall.
"""

import os
import geopandas as gpd
import pandas as pd
from tqdm import tqdm


def make_resolution_availability_map(
    regions_fn, tile_index_unary
):

    print('Reading regions...')
    regions = gpd.read_file(regions_fn).to_crs('EPSG:5070')

    # drop columns to be added below
    regions.drop(
        columns=['percent_covered_by_tiles'], errors='ignore', inplace=True
    )

    # make column that denotes if regions are covered by the tile index unary
    #print('Determine which HUCs are covered by the tile index...')
    #regions['covered_by_tiles'] = regions.covered_by(tile_index_unary)

    # make column that denotes if regions intersect with the tile index unary
    #print('Determine which HUCs intersect with the tile index...')
    #regions['intersects_tiles'] = regions.intersects(tile_index_unary)

    # compute percentage of each HUCs area that is covered by the tile index
    print('Compute percentage of each HUCs area that is covered by the tile index...')
    regions['percent_covered_by_tiles'] = 100* (regions.intersection(tile_index_unary).area / regions.area)

    regions.reset_index(drop=True, inplace=True)

    print('Write regions to file...')
    regions.to_file(regions_fn, driver='GPKG', index=False)


def prepare_tile_index(
    tile_index_1m_fn, tile_index_3m_fn, tile_index_fn
):
    
    print('Reading 1m and 3m tile index...')
    tile_index_1m = gpd.read_file(tile_index_1m_fn).to_crs('EPSG:5070')
    tile_index_3m = gpd.read_file(tile_index_3m_fn).to_crs('EPSG:5070')

    # join the 1m and 3m tile index and get unary union
    print('Joining 1m and 3m tile index...')
    tile_index = gpd.GeoDataFrame(
        pd.concat([tile_index_1m, tile_index_3m], ignore_index=True),
        crs='EPSG:5070'
    )

    print("Make tile index unary union...")
    tile_index_unary = tile_index.union_all()

    union_1m = tile_index.loc[tile_index.dem_resolution == 1].union_all()
    union_3m = tile_index.loc[tile_index.dem_resolution == 3].union_all()

    # make unary union of tile index by resolution
    tile_index = gpd.GeoDataFrame(
        {
            'geometry' : [
                union_1m,
                union_3m,
                tile_index_unary

            ]
            ,
            'resolution' : ['1', '3', 'either']
        },
        crs='EPSG:5070'
    )

    # write tile index to file
    print('Write tile index to file...')
    tile_index.to_file(tile_index_fn, driver='GPKG', index=False)

    return tile_index_unary


if __name__ == "__main__":


    data_dir = os.path.join(
        os.path.expanduser('~'), 'data', 'foss_fim', 'misc', 'resolution_analysis', 'metrics'
    )
    tile_index_dir = os.path.join(
        os.path.expanduser('~'), 'data', 'foss_fim', 'inputs', 'dems', '3dep_dems', 'lidar_tile_index'
    )

    regions_fns = [
        os.path.join(data_dir, 'skill','merged_skill_metrics_with_covariates.gpkg'),
        #os.path.join(data_dir, 'test_sites','test_sites_spatial_covariates.gpkg')
    ]

    tile_index_1m_fn = os.path.join(tile_index_dir, 'usgs_rocky_3dep_1m_tile_index_20240612.gpkg')
    tile_index_3m_fn = os.path.join(tile_index_dir, 'usgs_rocky_3dep_3m_tile_index_20240612.gpkg')
    tile_index_fn = os.path.join(data_dir, '..', 'tile_index_availability.gpkg')

    tile_index_unary = prepare_tile_index(
        tile_index_1m_fn, tile_index_3m_fn, tile_index_fn
    )

    for regions_fn in tqdm(regions_fns, desc='Adding availability to regions'):

        make_resolution_availability_map(
            regions_fn, tile_index_unary
        ) 