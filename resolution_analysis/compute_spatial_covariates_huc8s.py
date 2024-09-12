"""
Compute spatial covariates for the FIM resolution analysis at the HUC8 level.
"""


import os
import gc
from glob import glob

import rioxarray as rxr
import geopandas as gpd
from tqdm import tqdm
import pygeohydro as gh

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
    huc8s,
    id,
    output_dir='data',
    huc8s_output_dir='huc8s',
    spatial_covariates_gpkg_fn='spatial_covariates.gpkg'
):

    # open the huc8s
    print('Reading huc8s...')
    huc8s = gpd.read_file(huc8s).to_crs(5070)
    #huc8s = huc8s.head(3)

    # convert above to list of strings
    huc8s_to_drop = [
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

    # make huc8s column for nlcd retrieval
    print('Adding region column to states ...')
    states = gh.get_us_states().to_crs(5070)
    states.loc[~states.STUSPS.isin(['AK','PR','HI']),'region'] = 'L48'
    states.loc[states.STUSPS == 'AK','region'] = 'AK'
    states.loc[states.STUSPS == 'PR','region'] = 'PR'
    states.loc[states.STUSPS == 'HI','region'] = 'HI'
    states = states[['region', 'geometry']]

    #spatial join
    print('Spatial join to get regions ...')
    huc8s = (
        huc8s
        .loc[:, ['HUC8', 'geometry']]
        .dissolve(by='HUC8')
        .reset_index(drop=False)
        .sjoin(states, how='left', predicate='intersects')
        .drop(columns='index_right')
        .reset_index(drop=True)
        .dissolve(by='HUC8')
        .reset_index(drop=False)
        .loc[:, ['HUC8', 'geometry', 'region']]
        .set_index('HUC8')
        .drop(index=huc8s_to_drop)
        .reset_index(drop=False)
        .to_crs(5070)
    )

    # set of huc8s available in the spatial covariates
    available_huc8s = set(huc8s[id].astype(str).to_list())

    del states

    # make hucs_df to store huc_code, median_slope, freq_high_dev
    huc8s_output_dir = os.path.join(output_dir, huc8s_output_dir)
    slope_output_dir = os.path.join(huc8s_output_dir, 'slope')
    nlcd_output_dir = os.path.join(huc8s_output_dir, 'nlcd')

    # glob slope and nlcd
    slope_fns = glob(os.path.join(slope_output_dir, '*.tif'))
    nlcd_fns = glob(os.path.join(nlcd_output_dir, '*.tif'))

    # loop through slope_fns. compute median slope and store in hucs_df
    for slope_fn in tqdm(slope_fns, desc='Processing slope files'):

        # get huc8 from filename
        huc8 = os.path.basename(slope_fn).split('_')[1].split('.')[0]

        with rxr.open_rasterio(slope_fn, mask_and_scale=True) as slope:
            median_slope = compute_median_slope(slope)
        
        if slope is not None:
            slope.close()
        del slope
        gc.collect()

        # store in hucs_df
        huc8s.loc[huc8s[id] == huc8, 'median_slope'] = median_slope

    # impute median_slope where missing with median of touching geometries
    print("Number of missing median_slope: ", num_missing_median_slope := huc8s['median_slope'].isnull().sum())
    huc8s['missing_median_slope'] = huc8s['median_slope'].isnull()
    if num_missing_median_slope > 0:
        print('Imputing missing median_slope...')
        huc8s.loc[huc8s['median_slope'].isnull(), 'median_slope'] = (
            huc8s.loc[huc8s['median_slope'].isnull()]
            .sjoin(huc8s, how='left', predicate='touches')
            .groupby('HUC8_left')['median_slope_right']
            .median()
            .values
        )

    # loop through nlcd_fns. compute freq_high_dev and store in hucs_df
    for nlcd_fn in tqdm(nlcd_fns, desc='Processing nlcd files'):

        huc8 = os.path.basename(nlcd_fn).split('_')[1].split('.')[0]

        # if huc8 is in AK, PR, or HI, continue
        if huc8 in available_huc8s:
            if huc8s.loc[huc8s[id] == huc8, 'region'].values.item() in ['PR', 'HI']:
                nlcd_high_dev_class = 2
            else:
                nlcd_high_dev_class = 24
        else:
            continue

        with rxr.open_rasterio(nlcd_fn, mask_and_scale=True) as nlcd:
            
            freq_high_dev = compute_nlcd_high_dev(nlcd, nlcd_high_dev_class)
        
        if nlcd is not None:
            nlcd.close()
        del nlcd
        gc.collect()

        # store in hucs_df
        huc8s.loc[huc8s[id] == huc8, 'freq_high_dev'] = freq_high_dev

    # impute freq_high_dev where missing with median of touching geometries
    print("Number of missing freq_high_dev: ", num_missing_freq_high_dev := huc8s['freq_high_dev'].isnull().sum())
    huc8s['missing_freq_high_dev'] = huc8s['freq_high_dev'].isnull()
    if num_missing_freq_high_dev > 0:
        # missing: 21010007, 21020001, 21020002
        # all in PR with no LC data
        print('Imputing missing freq_high_dev...')
        huc8s.loc[huc8s['freq_high_dev'].isnull(), 'freq_high_dev'] = (
            huc8s.loc[huc8s['freq_high_dev'].isnull()]
            .sjoin(huc8s, how='left', predicate='touches')
            .groupby('HUC8_left')['freq_high_dev_right']
            .median()
            .values
        )

    print('Remaining missing freq_high_dev: ', num_missing_freq_high_dev := huc8s['freq_high_dev'].isnull().sum())
    if num_missing_freq_high_dev > 0:
        
        # for every missing freq_high_dev, compute the median of the n nearest geometries
        n = 3
        for idx, row in tqdm(
            huc8s.loc[huc8s['freq_high_dev'].isnull()].iterrows(),
            total=num_missing_freq_high_dev,
            desc=f'Imputing freq_high_dev with median of {n} nearest geometries'
        ):
            nearest_geometries = huc8s.loc[huc8s['freq_high_dev'].notnull()].distance(row['geometry'].centroid).nsmallest(n).index
            huc8s.loc[idx, 'freq_high_dev'] = huc8s.loc[nearest_geometries, 'freq_high_dev'].median()

    # drop region column
    huc8s.drop(columns='region', inplace=True)

    # write hucs gpkg
    full_spatial_covariates_gpkg_fn = os.path.join(output_dir, spatial_covariates_gpkg_fn)
    huc8s.to_file(
        full_spatial_covariates_gpkg_fn, driver='GPKG', index=False
    )

if __name__ == '__main__':
    

    huc8s = os.path.join('data', 'ALL_FIM60_HUC8s.gpkg')
    id = 'HUC8'
    output_dir = 'data'
    huc8s_output_dir = 'huc8s'
    spatial_covariates_gpkg_fn = 'huc8s_sp.gpkg'

    main(
        huc8s=huc8s,
        id=id,
        output_dir=output_dir,
        huc8s_output_dir=huc8s_output_dir,
        spatial_covariates_gpkg_fn=spatial_covariates_gpkg_fn,
    )