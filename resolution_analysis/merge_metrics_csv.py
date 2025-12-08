'''
Merge benchmark_extents and compute metrics with metrics csv.
'''

import os

import pandas as pd
import geopandas as gpd


def concat_compute_metrics(source_dir='data', only_zero_exit_status=True):

    metrics = [
        'PI3_fim60_10m_richdem.gpkg',
        'PI3_fim60_10m_wbt.gpkg',
        'PI3_fim60_5m_richdem.gpkg',
        'PI3_fim60_5m_wbt.gpkg',
        'PI3_uat_and_alpha_domain_3m_richdem.gpkg',
        'PI3_uat_and_alpha_domain_3m_wbt.gpkg',
    ]

    parse_resolution = lambda x: int(x.split('_')[-2][:-1])
    parse_algorithm = lambda x: x.split('_')[-1].split('.')[0]

    # loop through geopackages, add resolution and algorithm columns, and concat
    metrics_list = []
    for m in metrics:
        metrics_df = gpd.read_file(os.path.join(source_dir, m)).to_crs('EPSG:5070')
        metrics_df['resolution'] = parse_resolution(m)
        metrics_df['algorithm'] = parse_algorithm(m)
        metrics_list.append(metrics_df)
        del metrics_df
    
    metrics_df = pd.concat(metrics_list, ignore_index=True)
    del metrics_list

    # convert to geopandas
    metrics_df = gpd.GeoDataFrame(metrics_df, geometry='geometry', crs='EPSG:5070')

    # rename HUC8 to huc
    metrics_df.rename(columns={'HUC8': 'huc'}, inplace=True)

    # only fim60 and exit status 0
    if only_zero_exit_status:
        metrics_df = metrics_df[(metrics_df['Exit status'] == '0') & (metrics_df['fim60'] == 'yes')]
    else:
        metrics_df = metrics_df[metrics_df['fim60'] == 'yes']

    # save to file
    metrics_df.to_file(os.path.join(source_dir, 'compute_metrics.gpkg'), driver='GPKG', index=False)
    
    return metrics_df



def merge_metrics(metrics_csvs, resolutions, benchmark_extents, merged_metrics_csv=None, compute_metrics=None):
    """
    Merge benchmark_extents with metrics csv.
    """

    benchmark_extents = gpd.read_file(benchmark_extents)

    if resolutions is None:
        metrics = pd.concat(
            [pd.read_csv(m, dtype={'huc': str}) for m in metrics_csvs],
            ignore_index=True
        )

        # add algorithm and resolution columns
        metrics['algorithm'] = metrics.apply(lambda x: x['version'].split('_')[-1], axis=1)
        metrics['resolution'] = metrics.apply(lambda x: x['version'].split('_')[-2].rstrip('m'), axis=1).astype(int)

    else:
        # read metrics by make sure huc is str format
        metrics_list = []
        for m, r in zip(metrics_csvs, resolutions):
            metrics = pd.read_csv(m, dtype={'huc': str})
            metrics['resolution'] = r
            metrics_list.append(metrics)
            del metrics

        metrics = pd.concat(metrics_list, ignore_index=True)
        del metrics_list

    # make test_case_id for metrics
    ble_bool = metrics.benchmark_source == 'ble'
    usgs_bool = metrics.benchmark_source == 'usgs'
    nws_bool = metrics.benchmark_source == 'nws'
    ras2fim_bool = metrics.benchmark_source == 'ras2fim'

    # make test_case_id for metrics
    metrics.loc[ble_bool, 'test_case_id'] = metrics.loc[ble_bool, 'huc']
    metrics.loc[usgs_bool, 'test_case_id'] = metrics.loc[usgs_bool, 'nws_lid']
    metrics.loc[nws_bool, 'test_case_id'] = metrics.loc[nws_bool, 'nws_lid']
    metrics.loc[ras2fim_bool, 'test_case_id'] = metrics.loc[ras2fim_bool, 'huc'].apply(lambda x: 'ras2fim_' + x)

    # drop 
    benchmark_extents.drop(
        columns=['benchmark_source', 'geometry'], inplace=True
    )

    merged_metrics = metrics.merge(
        benchmark_extents,
        how='left',
        on='test_case_id',
    )

    if compute_metrics is not None:
        
        # merge compute metrics
        merged_metrics = merged_metrics.merge(
            compute_metrics.drop(
                columns=[
                    'geometry', 'areaacres', 'areasqkm', 'states', 'name', 'shape_Length',
       'shape_Area', 'fimid', 'fossid', 'fim30', 'fim60'
                ]
            ),
            how='left',
            on=['huc', 'resolution', 'algorithm']
        )

    if merged_metrics_csv:
        merged_metrics.to_csv(merged_metrics_csv, index=False)

    return merged_metrics


if __name__ == '__main__':

    versions = 'PI3' # 'PI1-high-res' or 'PI1'
    #resolutions = [10, 5, 3]
    resolutions = None
    
    if versions == 'PI1-high-res':
        metrics_csvs = [
            os.path.join('data', '10m_lidar_ngwpc_pI1_high_res_metrics.csv'),
            os.path.join('data', '5m_lidar_ngwpc_PI1_high_res_metrics.csv'),
            os.path.join('data', '3m_lidar_ngwpc_PI1_high_res_metrics.csv'),
        ]
    elif versions == 'PI1':
        metrics_csvs = [
            os.path.join('data', '10m_lidar_ngwpc_pI1_metrics.csv'),
            os.path.join('data', '5m_lidar_ngwpc_PI1_metrics.csv'),
            os.path.join('data', '3m_lidar_ngwpc_metrics.csv'),
        ]
    elif versions == 'PI3':
        metrics_csvs = [
            os.path.join('data', 'combined_PI3_wbt_metrics.csv')
        ]
    
    benchmark_extents = os.path.join('data', 'benchmarks_sp.gpkg')
    merged_metrics_csv = os.path.join('data', 'merged_metrics.csv')

    # merge compute metrics
    metrics_df = concat_compute_metrics('data')

    merge_metrics(metrics_csvs, resolutions, benchmark_extents, merged_metrics_csv, compute_metrics=metrics_df)



