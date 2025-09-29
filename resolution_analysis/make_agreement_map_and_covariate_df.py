"""
Globs available benchmark maps and creates a geodataframe with the extent of the benchmark maps.
"""



import os
import gc

import geopandas as gpd
import pandas as pd
import rioxarray as rxr
from glob import glob
from tqdm import tqdm
from shapely.geometry import box

tqdm.pandas()

def make_agreement_map_and_covariate_df():
    """
    Globs available benchmark maps and creates a geodataframe with the extent of the benchmark maps.
    """

    test_cases_dir = os.path.join('data', 'foss_fim', 'test_cases')

    # find dirs in test_cases
    glob_pattern = os.path.join(os.path.expanduser('~'), test_cases_dir, '**', 'validation_data_*', '**', '*extent*.tif')
    extent_files = glob(glob_pattern, recursive=True)
                        
    
    benchmark_source = [ef.split('/')[6].split('_')[0] for ef in extent_files]

    # remove ras2fim benchmark source
    extent_files = [ef for ef, bs in zip(extent_files, benchmark_source) if bs != 'ras2fim']
    benchmark_source = [bs for bs in benchmark_source if bs != 'ras2fim']
    huc8s = [ef.split('/')[8] for ef in extent_files]

    # make bool mask for usgs and nws
    usgs_nws_bool = [bs in ['usgs', 'nws'] for bs in benchmark_source]
    ble_bool = [bs == 'ble' for bs in benchmark_source]

    # find extent_files and label 'ble', 'nws', 'usgs
    extent_files_df = pd.DataFrame(extent_files, columns=['extent_file'])
    extent_files_df['benchmark_source'] = benchmark_source
    extent_files_df['huc8'] = huc8s

    # usgs and nws extent files
    extent_files_usgs_nws = [ef for ef, bs in zip(extent_files, usgs_nws_bool) if bs]
    extent_files_ble = [ef for ef, bs in zip(extent_files, ble_bool) if bs]

    # for usgs and nws only
    extent_files_df.loc[usgs_nws_bool, 'test_case_id'] = [ef.split('/')[9] for ef in extent_files_usgs_nws]
    #extent_files_df.loc[usgs_nws_bool, 'magnitude'] = [ef.split('/')[10] for ef in extent_files_usgs_nws]
    #extent_files_df.loc[usgs_nws_bool, 'extent_file'] = [ef.split('/')[11] for ef in extent_files_usgs_nws]
    extent_files_df.loc[usgs_nws_bool, 'extent_file'] = extent_files_usgs_nws

    # for ble only
    extent_files_df.loc[ble_bool, 'test_case_id'] = [ef.split('/')[8] for ef in extent_files_ble]
    #extent_files_df.loc[ble_bool, 'magnitude'] = [ef.split('/')[9] for ef in extent_files_ble]
    #extent_files_df.loc[ble_bool, 'extent_file'] = [ef.split('/')[10] for ef in extent_files_ble]
    extent_files_df.loc[ble_bool, 'extent_file'] = extent_files_ble

    # remove duplicate magnitudes and extent_files per test_case_id
    extent_files_df = (
        extent_files_df
        .drop_duplicates(subset=['test_case_id'], keep='first')
        .reset_index(drop=True)
    )

    def get_bounding_box_and_crs(row):
        extent_file = row['extent_file']
        with rxr.open_rasterio(extent_file, parse_coordinates=False) as extent:
            bounds = extent.rio.bounds()
            crs = extent.rio.crs
            
        
        extent.close()
        del extent
        gc.collect()

        bounds = box(*bounds)

        # convert bounds to EPSG:5070
        bounds = gpd.GeoSeries(bounds, crs=crs).to_crs('EPSG:5070').iloc[0]
        
        row['geometry'] = bounds
        return row
    
    #extent_files_df = extent_files_df.apply(get_bounding_box_and_crs, axis=1)

    # do apply with progress bar tqdm
    extent_files_df = extent_files_df.progress_apply(
        get_bounding_box_and_crs, axis=1
    )

    # remove first three directories in extent_file column and overwrite column
    #extent_files_df['extent_file'] = extent_files_df['extent_file'].apply(lambda x: '/'.join(x.split('/')[3:]))

    # drop extent_file column
    extent_files_df.drop(columns='extent_file', inplace=True)
    
    # rename bounds to geometry
    #extent_files_df.rename(columns={'bounds': 'geometry'}, inplace=True)

    # make a geodataframe
    extent_files_gdf = gpd.GeoDataFrame(extent_files_df, geometry='geometry', crs='EPSG:5070')

    extent_file_fn = os.path.join('data', 'benchmarks_sp.gpkg')

    extent_files_gdf.to_file(extent_file_fn, driver='GPKG', index=False)

    
if __name__ == "__main__":
    make_agreement_map_and_covariate_df()



