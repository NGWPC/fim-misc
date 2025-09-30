"""
Compute spatial covariates for the FIM resolution analysis at the HUC level.
"""


import os
import argparse
import gc
from glob import glob

import rioxarray as rxr
import geopandas as gpd
from tqdm import tqdm
import pygeohydro as gh
import numpy as np

# quiet future warnings on import 
import warnings
warnings.filterwarnings(
    'ignore', category=UserWarning, module='rioxarray._io', message='The dataset\'s nodata attribute is shadowing the alpha band'
)


os.environ["HYRIVER_CACHE_DISABLE"] = "true"


def compute_nlcd_high_dev(nlcd, nlcd_high_dev_class=24):
    """
    Compute the high developed NLCD class from NLCD data.
    """
    # find frequency of high developed NLCD class
    high_dev_count = nlcd.where(nlcd == nlcd_high_dev_class).count().values.item()
    total_count = nlcd.count().values.item()

    try:
        nlcd_high_dev_freq = high_dev_count / total_count
    except ZeroDivisionError:
        nlcd_high_dev_freq = None
    
    return nlcd_high_dev_freq

def compute_median_slope(slope):
    """
    Compute the median slope from slope data.
    """
    return slope.median().values.item()


def main(
    hucs,
    output_dir='data',
    spatial_covariates_gpkg_fn='spatial_covariates.gpkg',
    huc_col='HUC12',
):

    # open the hucs
    print('Reading hucs...')
    hucs = gpd.read_file(hucs).to_crs(5070)
    
    # debugging: limit hucs to first few rows
    #hucs = hucs.head(100)

    # testing_domain True if huc_col is 'id' (for testing domains), False otherwise
    testing_domain = True if 'id' in huc_col.lower() else False

    # convert above to list of strings
    hucs_to_drop = [
        '02080101',
        '04260000',
        '04280002',
        '04300109',
        '19010402',
        '19020702',
        '19030204',
        '19030205',
        '19030206',
        '19030401',
        '19030405',
        '19030407',
        '19070402',
        '19080301',
        '19080302',
        '19080304',
        '19080308',
        '21010007'
    ]

    # make hucs column for nlcd retrieval
    print('Adding region column to states ...')
    states = gh.get_us_states().to_crs(5070)
    states.loc[~states.STUSPS.isin(['AK','PR','HI']),'region'] = 'L48'
    states.loc[states.STUSPS == 'AK','region'] = 'AK'
    states.loc[states.STUSPS == 'PR','region'] = 'PR'
    states.loc[states.STUSPS == 'HI','region'] = 'HI'
    states = states[['region', 'geometry']]

    #spatial join
    print('Spatial join to get regions ...')
    hucs = (
        hucs
        .loc[:, [huc_col, 'geometry']]
        .dissolve(by=huc_col)
        .reset_index(drop=False)
        .sjoin(states, how='left', predicate='intersects')
        .drop(columns='index_right')
        .reset_index(drop=True)
        .dissolve(by=huc_col)
        .reset_index(drop=False)
        .loc[:, [huc_col, 'geometry', 'region']]
        .to_crs(5070)
    )

    # drop hucs that start with the hucs_to_drop list
    hucs = hucs[~hucs[huc_col].str.startswith(tuple(hucs_to_drop))].reset_index(drop=True)

    # set of hucs available in the spatial covariates
    available_hucs = set(hucs[huc_col].astype(str).to_list())

    del states

    # make hucs_df to store huc_code, median_slope, freq_high_dev
    slope_output_dir = os.path.join(output_dir, 'slope')
    nlcd_output_dir = os.path.join(output_dir, 'nlcd')

    # assert that slope_output_dir and nlcd_output_dir exist
    assert os.path.exists(slope_output_dir), f"Slope output directory does not exist: {slope_output_dir}"
    assert os.path.exists(nlcd_output_dir), f"NLCD output directory does not exist: {nlcd_output_dir}"

    # glob slope and nlcd
    slope_fns = glob(os.path.join(slope_output_dir, '*.tif'))
    nlcd_fns = glob(os.path.join(nlcd_output_dir, '*.tif'))

    # check if slope_fns and nlcd_fns are empty
    assert len(slope_fns) > 0, f"No slope files found in the specified directory, please check {slope_output_dir}."
    assert len(nlcd_fns) > 0, f"No NLCD files found in the specified directory, please check {nlcd_output_dir}."

    # drop slope_fns and nlcd_fns that do not contain hucs in available_hucs
    # only works with HUCs not test case domains
    if testing_domain:
        pass
    else:
        slope_fns = [fn for fn in slope_fns if os.path.basename(fn).split('_')[1].split('.')[0] in available_hucs]
        nlcd_fns = [fn for fn in nlcd_fns if os.path.basename(fn).split('_')[1].split('.')[0] in available_hucs]

    # loop through slope_fns. compute median slope and store in hucs_df
    for slope_fn in tqdm(slope_fns, desc='Processing slope files'):

        # get huc from filename
        if testing_domain:
            huc = '_'.join(os.path.basename(slope_fn).split('_')[1:]).split('.')[0]
        else:
            huc = os.path.basename(slope_fn).split('_')[1].split('.')[0]

        try:
            with rxr.open_rasterio(slope_fn, mask_and_scale=True) as slope:
                median_slope = compute_median_slope(slope)
        except FileNotFoundError:
            print(f"File {slope_fn} not found. Skipping...")
            continue

        if slope is not None:
            slope.close()
        del slope
        gc.collect()

        # store in hucs_df
        hucs.loc[hucs[huc_col] == huc, 'median_slope'] = median_slope

    # impute median_slope where missing with median of touching geometries
    print("Number of missing median_slope: ", num_missing_median_slope := hucs['median_slope'].isnull().sum())
    hucs['missing_median_slope'] = hucs['median_slope'].isnull()
    if num_missing_median_slope > 0:
        print('Imputing missing median_slope...')
        hucs.loc[hucs['median_slope'].isnull(), 'median_slope'] = (
            hucs.loc[hucs['median_slope'].isnull()]
            .sjoin(hucs, how='left', predicate='touches')
            #.groupby('HUC8_left')['median_slope_right']
            .groupby(f'{huc_col}_left')['median_slope_right']
            .median()
            .values
        )

    # loop through nlcd_fns. compute freq_high_dev and store in hucs_df
    for nlcd_fn in tqdm(nlcd_fns, desc='Processing nlcd files'):

        if testing_domain:
            huc = '_'.join(os.path.basename(nlcd_fn).split('_')[1:]).split('.')[0]
        else:
            huc = os.path.basename(nlcd_fn).split('_')[1].split('.')[0]

        # if huc is in AK, PR, or HI, continue
        if huc in available_hucs:
            if hucs.loc[hucs[huc_col] == huc, 'region'].values.item() in ['PR', 'HI']:
                nlcd_high_dev_class = 2
            else:
                nlcd_high_dev_class = 24
        else:
            continue

        try:
            with rxr.open_rasterio(nlcd_fn, mask_and_scale=True) as nlcd:
                freq_high_dev = compute_nlcd_high_dev(nlcd, nlcd_high_dev_class)
        except FileNotFoundError:
            print(f"File {nlcd_fn} not found. Skipping...")
            continue
        
        if nlcd is not None:
            nlcd.close()
        del nlcd
        gc.collect()

        # store in hucs_df
        hucs.loc[hucs[huc_col] == huc, 'freq_high_dev'] = freq_high_dev

    # impute freq_high_dev where missing with median of touching geometries
    print("Number of missing freq_high_dev: ", num_missing_freq_high_dev := hucs['freq_high_dev'].isnull().sum())
    hucs['missing_freq_high_dev'] = hucs['freq_high_dev'].isnull()
    if num_missing_freq_high_dev > 0:
        # missing: 21010007, 21020001, 21020002
        # all in PR with no LC data
        print('Imputing missing freq_high_dev...')
        hucs.loc[hucs['freq_high_dev'].isnull(), 'freq_high_dev'] = (
            hucs.loc[hucs['freq_high_dev'].isnull()]
            .sjoin(hucs, how='left', predicate='touches')
            #.groupby('HUC8_left')['freq_high_dev_right']
            .groupby(f'{huc_col}_left')['freq_high_dev_right']
            .median()
            .values
        )

    print('Remaining missing freq_high_dev: ', num_missing_freq_high_dev := hucs['freq_high_dev'].isnull().sum())
    if num_missing_freq_high_dev > 0:
        
        # for every missing freq_high_dev, compute the median of the n nearest geometries
        n = 3
        for idx, row in tqdm(
            hucs.loc[hucs['freq_high_dev'].isnull()].iterrows(),
            total=num_missing_freq_high_dev,
            desc=f'Imputing freq_high_dev with median of {n} nearest geometries'
        ):
            nearest_geometries = hucs.loc[hucs['freq_high_dev'].notnull()].distance(row['geometry'].centroid).nsmallest(n).index
            hucs.loc[idx, 'freq_high_dev'] = hucs.loc[nearest_geometries, 'freq_high_dev'].median()

    # drop region column
    hucs.drop(columns='region', inplace=True)

    # write hucs gpkg
    full_spatial_covariates_gpkg_fn = os.path.join(output_dir, spatial_covariates_gpkg_fn)
    hucs.to_file(
        full_spatial_covariates_gpkg_fn, driver='GPKG', index=False
    )

if __name__ == '__main__':
    

    #hucs = os.path.join('data', 'ALL_FIM60_HUC8s.gpkg')
    #output_dir = 'data'
    #spatial_covariates_gpkg_fn = 'hucs_sp.gpkg'
    
    # parse arguments
    parser = argparse.ArgumentParser(
        description='Compute spatial covariates for the FIM resolution analysis at the HUC level.'
    )

    parser.add_argument(
        '--hucs',
        type=str,
        required=True,
        help='Path to the HUC file.'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(
            os.path.expanduser('~'), 'data', 'foss_fim', 'misc', 'resolution_analysis', 'spatial_covariates','hucs'
        ),
        required=False,
        help='Output directory.'
    )
    parser.add_argument(
        '--spatial_covariates_gpkg_fn',
        type=str,
        default='hucs_spatial_covariates.gpkg',
        required=False,
        help='Outputs Spatial covariates GPKG filename.'
    )
    parser.add_argument(
        '--huc_col',
        type=str,
        default='HUC12',
        required=False,
        help='Column name for HUCs in the input file or "id" in test cases domain.'
    )


    # parse arguments and call main function
    main(**vars(parser.parse_args()))