

import os
import gc

import geopandas as gpd
import rioxarray as rxr
from tqdm import tqdm
from rioxarray.merge import merge_arrays

from compute_spatial_covariates_huc8s import compute_median_slope, compute_nlcd_high_dev


def close_xarray(xr):
    """
    Close xarray dataset
    """
    if xr is not None:
        xr.close()
    del xr
    gc.collect()

def main():

    # data directory
    data_dir = 'data'

    # read the spatial covariates
    #huc8s = gpd.read_file(os.path.join(data_dir, 'spatial_covariates.gpkg')).to_crs(5070)
    huc8s = gpd.read_file(os.path.join(data_dir, 'huc8s_sp.gpkg')).to_crs(5070)
    benchmark_extents = gpd.read_file(os.path.join(data_dir, 'benchmarks_sp.gpkg')).to_crs(5070)

    # spatial join
    print('Spatial join...')
    benchmark_extents_joined = (
        benchmark_extents
        .drop(columns=['HUC8'], errors='ignore')
        .sjoin(
            huc8s.drop(
                columns=['median_slope', 'missing_median_slope','freq_high_dev', 'missing_freq_high_dev', 'region'],
                errors='ignore'
            ),
            how='left',
            predicate='intersects'
        )
        .drop(columns=['index_right', 'huc8'], errors='ignore')
        .dropna(subset=['HUC8'])
        .sort_values(by=['test_case_id', 'HUC8'])
        .reset_index(drop=True)
    )

    # rewrite spatial join above on one line
    #be_joined = (be_orig.drop(columns=['HUC8'], errors='ignore').sjoin(huc8s.drop(columns=['median_slope', 'missing_median_slope','freq_high_dev', 'missing_freq_high_dev', 'region'], errors='ignore'), how='left', predicate='intersects').drop(columns=['index_right', 'huc8'], errors='ignore').dropna(subset=['HUC8']).sort_values(by=['test_case_id', 'HUC8']).reset_index(drop=True))

    # for every benchmark extent, compute unique HUC8s
    unique_huc8s_df = (
        benchmark_extents_joined
        .groupby('test_case_id')['HUC8']
        .unique()
        .reset_index(drop=False)
    )

    for _, row in tqdm(unique_huc8s_df.iterrows(), total=len(unique_huc8s_df), desc='Processing test cases'):

        # get x_min, y_min, x_max, y_max of benchmark extent
        x_min, y_min, x_max, y_max = benchmark_extents_joined.loc[benchmark_extents_joined['test_case_id'] == row['test_case_id'], 'geometry'].total_bounds.tolist()

        unique_huc8s = row['HUC8']

        slope_fns = [os.path.join(data_dir, 'huc8s', 'slope', f"slope_{huc8}.tif") for huc8 in unique_huc8s]

        # for every slope_fns, load xarray, aggregate to list
        slope_list = []
        for slope_fn in slope_fns:
            with rxr.open_rasterio(slope_fn, parse_coordinates=True, mask_and_scale=True) as slope:
                slope_list.append(slope)
            close_xarray(slope)

        # merge slope
        merged_slope = merge_arrays(slope_list)
        for slope in slope_list:
            close_xarray(slope)

        # select merged_slope within benchmark extent
        merged_slope = merged_slope.sel(x=slice(x_min, x_max), y=slice(y_max, y_min))

        # compute median slope
        test_case_bool = benchmark_extents['test_case_id'] == row['test_case_id']
        benchmark_extents.loc[test_case_bool, 'median_slope'] = compute_median_slope(merged_slope)
        close_xarray(merged_slope)

        # now for nlcd
        nlcd_fns = [os.path.join(data_dir, 'huc8s', 'nlcd', f"nlcd_{huc8}.tif") for huc8 in unique_huc8s]

        # for every nlcd_fns, load xarray, aggregate to list
        nlcd_list = []
        for nlcd_fn in nlcd_fns:
            with rxr.open_rasterio(nlcd_fn, parse_coordinates=True, mask_and_scale=True) as nlcd:
                nlcd_list.append(nlcd)
            close_xarray(nlcd)
        
        # merge nlcd
        merged_nlcd = merge_arrays(nlcd_list)
        for nlcd in nlcd_list:
            close_xarray(nlcd)

        # select merged_nlcd within benchmark extent
        merged_nlcd = merged_nlcd.sel(x=slice(x_min, x_max), y=slice(y_max, y_min))

        # compute nlcd high dev
        benchmark_extents.loc[test_case_bool, 'freq_high_dev'] = compute_nlcd_high_dev(merged_nlcd)
        close_xarray(merged_nlcd)
    
    # print number of missing values
    print('Number of missing median_slope: ', benchmark_extents['median_slope'].isnull().sum())
    print('Number of missing freq_high_dev: ', benchmark_extents['freq_high_dev'].isnull().sum())

    # remove benchmarks with missing HUC8 intersections
    benchmark_extents = (
        benchmark_extents
        .dropna(subset=['median_slope', 'freq_high_dev'])
        .drop(columns='huc8', errors='ignore')
        .sort_values(by=['test_case_id'])
        .reset_index(drop=True)
    )

    # write the benchmark extents
    benchmark_extents_fn = os.path.join(data_dir, 'benchmarks_sp.gpkg')
    benchmark_extents.to_file(benchmark_extents_fn, driver='GPKG', index=False)

if __name__ == "__main__":
    main()

