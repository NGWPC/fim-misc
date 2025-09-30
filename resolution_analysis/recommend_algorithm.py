'''
Make a gpkg from compute_metrics.gpkg that recommends which algorithm to use.
'''

import os
import warnings

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from merge_metrics_csv import concat_compute_metrics


def recommend_algorithm():
    """
    Make a gpkg from compute_metrics.gpkg that recommends which algorithm to use.
    """

    print('Concatenate compute metrics ...')
    compute_metrics = concat_compute_metrics('data', only_zero_exit_status=False)

    # only keep the columns we need
    compute_metrics = (
        compute_metrics
        .loc[
            (compute_metrics.fim30 == 'yes') | (compute_metrics.fim60 == 'yes'),
            ['resolution', 'algorithm', 'Exit status', 'huc' ,'geometry']
        ]
        .rename(columns={'huc': 'HUC8', 'Exit status' : 'exit_status'})
        .reset_index(drop=True)
    )

    # get count of HUCs by 'algorithm', 'resolution', and 'Exit status'
    print("HUCs by algorithm, resolution, and exit status:")
    print(compute_metrics.groupby(['algorithm', 'resolution', 'exit_status'])['HUC8'].count())


    rec_fp = os.path.join('data', 'recommend_algorithm.gpkg')

    if os.path.exists(rec_fp):
        os.remove(rec_fp)

    # write to gpkg
    # for each resolution, write to a layer.
    for r in tqdm(compute_metrics['resolution'].unique(), desc='Algo recommendation by resolution'):

        print(f"Resolution: {r}m")

        cmr = (
            compute_metrics
            .loc[compute_metrics['resolution'] == r]
            .reset_index(drop=True)
        )

        # get set of completed (zero-exit code) HUC8 codes for the wbt algorithm
        wbt_bool = (cmr['exit_status'] == '0') & (cmr['algorithm'] == 'wbt')
        richdem_bool = (cmr['exit_status'] == '0') & (cmr['algorithm'] == 'richdem') & ~wbt_bool
        fail_bool = cmr['exit_status'] != '0'

        # get set of completed (zero-exit code) HUC8 codes for the richdem algorithm
        wbt_huc8_set = set(cmr.loc[wbt_bool, 'HUC8'])
        richdem_huc8_set = set(cmr.loc[richdem_bool, 'HUC8'])
        fail_huc8_set = set(cmr.loc[fail_bool, 'HUC8'])

        # remove wbt HUC8 codes from richdem and fail sets
        richdem_huc8_set -= wbt_huc8_set
        fail_huc8_set -= wbt_huc8_set
        fail_huc8_set -= richdem_huc8_set

        # reset bools with updated sets
        wbt_bool = cmr['HUC8'].isin(wbt_huc8_set)
        richdem_bool = cmr['HUC8'].isin(richdem_huc8_set)
        fail_bool = cmr['HUC8'].isin(fail_huc8_set)

        # add recommended_algorithm column
        cmr.loc[wbt_bool, 'recommended_algorithm'] = 'wbt'
        cmr.loc[richdem_bool, 'recommended_algorithm'] = 'richdem'
        cmr.loc[fail_bool, 'recommended_algorithm'] = pd.NA

        # drop fail_bool and reset index
        #cmr = cmr.loc[~fail_bool].reset_index(drop=True)

        # drop duplicates
        cmr.drop_duplicates(subset=['HUC8', 'recommended_algorithm'], inplace=True)

        # drop algorithm and exit_status
        cmr.drop(columns=['algorithm', 'exit_status'], inplace=True)
        
        cmr.to_file(
            rec_fp,
            driver='GPKG',
            index=False,
            layer=f'recommend_algorithm_{r}'
        )

        del cmr


if __name__ == '__main__':
    recommend_algorithm()