'''
Make a gpkg from compute_metrics.gpkg that recommends which algorithm to use.
'''

import os

import geopandas as gpd

from merge_metrics_csv import concat_compute_metrics


def recommend_algorithm():
    """
    Make a gpkg from compute_metrics.gpkg that recommends which algorithm to use.
    """

    compute_metrics = concat_compute_metrics('data', only_zero_exit_status=False)

    # only keep the columns we need
    compute_metrics = compute_metrics[['resolution', 'algorithm', 'Exit status', 'recommended_algorithm', 'geometry']]

    # drop duplicates
    compute_metrics.drop_duplicates(inplace=True)

    # if wbt algorithm has a zero exit status, recommend it.
    wbt_bool = (compute_metrics['Exit status'] == '0') & (compute_metrics['algorithm'] == 'wbt')
    
    # if wbt is not recommended, recommend richdem if it has a zero exit status.
    richdem_bool = (compute_metrics['Exit status'] == '0') & (compute_metrics['algorithm'] == 'richdem') & ~wbt_bool

    # if neither wbt nor richdem are recommended, recommend None
    none_bool = compute_metrics['Exit status'] != '0'

    compute_metrics.loc[wbt_bool, 'recommended_algorithm'] = 'wbt'
    compute_metrics.loc[richdem_bool, 'recommended_algorithm'] = 'richdem'
    compute_metrics.loc[none_bool, 'recommended_algorithm'] = 'none'

    # write to gpkg
    # for each resolution, write to a layer.
    for r in compute_metrics['resolution'].unique():
        compute_metrics[compute_metrics['resolution'] == r].to_file(os.path.join('data', 'recommend_algorithm.gpkg'), driver='GPKG', index=False, layer=f'recommend_algorithm_{r}')

    return compute_metrics


if __name__ == '__main__':
    recommend_algorithm()