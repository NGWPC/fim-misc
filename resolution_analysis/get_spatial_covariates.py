"""
Get spatial covariates (slope and NLCD) for the FIM60 HUCs.
"""

import argparse
import os
import gc
from time import sleep

from pynhd import NLDI
import pygeohydro as gh
from py3dep import py3dep
import rioxarray as rxr
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

# quiet future warnings on import 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

with warnings.catch_warnings():
    import xrspatial

os.environ["HYRIVER_CACHE_DISABLE"] = "true"


def get_slope(geoseries, resolution=30, geo_crs=4326, crs=5070, m2m=True, to_file=None, id='huc'):
    """
    Get slope from a geometry using py3dep and xrspatial.
    """
    
    dem = py3dep.get_dem(geoseries.geometry, resolution, geo_crs)
    dem = dem.rio.reproject(crs)
    
    # slope
    slope = xrspatial.slope(dem)
    slope_attrs = 'degrees'
    if m2m:
        slope = py3dep.geoops.deg2mpm(slope)
        slope_attrs = 'm/m'
    
    # set slope attributes
    slope.attrs['units'] = slope_attrs

    # encoded nodata
    slope.rio.write_nodata(-10, inplace=True, encoded=True)

    # write to file
    if to_file is not None:
        slope.rio.to_raster(
            to_file.format(geoseries[id]), driver='GTiff', windowed=True, tiled=True, blockxsize=256, blockysize=256, compress='lzw'
        )

    return slope


def get_nlcd(geoseries, resolution=30, geo_crs=4326, crs=5070, region=None, to_file=None, id='huc'):
    """
    Get NLCD from a geometry using py3dep.
    """

    # get NLCD
    nlcd = gh.nlcd.nlcd_bygeom(
        gpd.GeoSeries([geoseries.geometry], crs=geo_crs), resolution, years={'cover':2021}, crs=geo_crs, region=region
    )
    
    # get the first item in the dictionary
    nlcd = next(iter(nlcd.values()))
    nlcd = nlcd['cover_2021']
    nlcd = nlcd.rio.reproject(crs)
    
    # write to file
    if to_file is not None:
        nlcd.rio.to_raster(
            to_file.format(geoseries[id]), driver='GTiff', windowed=True, tiled=True, blockxsize=256, blockysize=256, compress='lzw'
        )

    return nlcd

def main(
    hucs,
    id,
    hucs_layer=None,
    output_dir='data',
    hucs_output_dir='hucs',
    max_retries=3,
    num_jobs=1,
    resolution=30,
):

    # open the hucs
    print('Reading hucs...')
    hucs = gpd.read_file(hucs, layer=hucs_layer).to_crs(4326)

    # make hucs column for nlcd retrieval
    print('Adding region column to states ...')
    states = gh.get_us_states().to_crs(4326)
    states.loc[~states.STUSPS.isin(['AK','PR','HI']),'region'] = 'L48'
    states.loc[states.STUSPS == 'AK','region'] = 'AK'
    states.loc[states.STUSPS == 'PR','region'] = 'PR'
    states.loc[states.STUSPS == 'HI','region'] = 'HI'
    states = states[['region', 'geometry']]

    #spatial join
    print('Spatial join to get regions ...')
    hucs = (
        hucs
        .loc[:, [id, 'geometry']]
        .dissolve(by=id)
        .reset_index(drop=False)
        .sjoin(states, how='left', predicate='intersects')
        .drop(columns='index_right')
        .reset_index(drop=True)
        .dissolve(by=id)
        .reset_index(drop=False)
        .loc[:, [id, 'geometry', 'region']]
        .to_crs(4326)
    )

    del states

    hucs_output_dir = os.path.join(output_dir, hucs_output_dir)
    slope_output_dir = os.path.join(hucs_output_dir, 'slope')
    nlcd_output_dir = os.path.join(hucs_output_dir, 'nlcd')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(hucs_output_dir, exist_ok=True)
    os.makedirs(slope_output_dir, exist_ok=True)
    os.makedirs(nlcd_output_dir, exist_ok=True)

    if num_jobs == 1:

        for _, row in tqdm(hucs.iterrows(), total=len(hucs), desc='Processing hucs'):

            try:
                slope_fn = os.path.join(slope_output_dir, 'slope_{}.tif'.format(row[id]))
                retries = 0
                while retries < max_retries:
                    try:
                        # get slope
                        slope = get_slope(
                            row, to_file=slope_fn, id=id, geo_crs=4326, crs=5070, resolution=resolution
                        )
                    except:
                        retries += 1
                        if retries >= max_retries:
                            print(f'Failed to get slope for {row[id]}, retrying...')
                            slope = None
                        
                        sleep(10)
                        continue
                    else:
                        break

            except Exception as e:
                print(row[id], 'slope failed', e, e.__traceback__.tb_lineno)
                continue

            try:
                slope.close()
            except AttributeError:
                pass
            del slope
            gc.collect()
            

            try:

                nlcd_fn = os.path.join(nlcd_output_dir,'nlcd_{}.tif'.format(row[id]))
                retries = 0
                while retries < max_retries:
                    try:
                        # get NLCD
                        nlcd = get_nlcd(
                            row, to_file=nlcd_fn, id=id, region=row['region'], geo_crs=4326, crs=5070, resolution=resolution
                        )
                        
                    except:
                        retries += 1
                        if retries >= max_retries:
                            print(f'Failed to get NLCD for {row[id]}, retrying...')
                            nlcd = None
                        
                        sleep(10)
                        continue
                    else:
                        break
                        
            except Exception as e:
                print(row[id], 'nlcd failed', e, e.__traceback__.tb_lineno)
                continue

            try:
                nlcd.close()
            except AttributeError:
                pass
            del nlcd
            gc.collect()

    else:
        def _process_row(row):

            try:
                slope_fn = os.path.join(slope_output_dir, 'slope_{}.tif'.format(row[id]))
                retries = 0
                while retries < max_retries:
                    try:
                        # get slope
                        slope = get_slope(
                            row, to_file=slope_fn, id=id, geo_crs=4326, crs=5070, resolution=resolution
                        )
                    except:
                        retries += 1
                        if retries >= max_retries:
                            slope = None
                            print(f'Failed to get slope for {row[id]}, retrying...')
                        
                        sleep(10)
                        continue
                    else:
                        break

            except Exception as e:
                print(row[id], 'slope failed', e, e.__traceback__.tb_lineno)
                pass

            try:
                slope.close()
            except AttributeError:
                pass
            del slope
            gc.collect()
            

            try:

                nlcd_fn = os.path.join(nlcd_output_dir,'nlcd_{}.tif'.format(row[id]))
                retries = 0
                while retries < max_retries:
                    try:
                        # get NLCD
                        nlcd = get_nlcd(
                            row, to_file=nlcd_fn, id=id, region=row['region'], geo_crs=4326, crs=5070, resolution=resolution
                        )
                        
                    except:
                        retries += 1
                        if retries >= max_retries:
                            nlcd = None
                            print(f'Failed to get NLCD for {row[id]}, retrying...')
                        
                        sleep(10)
                        continue
                    else:
                        break
                        
            except Exception as e:
                print(row[id], 'nlcd failed', e, e.__traceback__.tb_lineno)
                pass

            try:
                nlcd.close()
            except AttributeError:
                pass
            del nlcd
            gc.collect()

        # parallel processing
        from joblib import Parallel, delayed
        Parallel(n_jobs=num_jobs, backend='loky')(
            delayed(_process_row)(row) for _, row in tqdm(hucs.iterrows(), total=len(hucs), desc='Processing hucs')
        )


if __name__ == "__main__":

    #hucs = os.path.join(os.path.join(os.path.expanduser('~'), 'data','foss_fim', 'inputs', 'wbd', 'ALL_FIM100_HUC12s.gpkg')) # os.path.join('data', 'benchmark_maps.gpkg')
    #id = 'HUC12' # test_case_id 
    #hucs_layer = None
    #output_dir = os.path.join(os.path.expanduser('~'), 'data','foss_fim', 'misc','resolution_analysis')
    #hucs_output_dir = 'hucs' # 'benchmarks'
    #max_retries = 10
    #num_jobs = 1
    #resolution = 100

    # above args with argparse
    parser = argparse.ArgumentParser(description='Get spatial covariates for HUCs.')
    parser.add_argument('--hucs', type=str, help='Path to HUCs file', required=True)
    parser.add_argument('--id', type=str, help='HUCs ID column name', required=True, default='HUC12')
    parser.add_argument('--hucs_layer', type=str, help='HUCs layer name', required=False, default=None)
    parser.add_argument('--output_dir', type=str, help='Output directory', required=False, default='data')
    parser.add_argument('--hucs_output_dir', type=str, help='HUCs output directory', required=False, default='hucs')
    parser.add_argument('--max_retries', type=int, help='Max retries for getting covariates', required=False, default=10)
    parser.add_argument('--num_jobs', type=int, help='Number of jobs to run in parallel', required=False, default=1)
    parser.add_argument('--resolution', type=int, help='Resolution of covariates', required=False, default=30)

    # example usage:
    # python3 get_spatial_covariates.py --hucs ~/foss_data/inputs/wbd/ALL_FIM100_HUC12s.gpkg --id HUC12 --output_dir ~/foss_data/misc/resolution_analysis --hucs_output_dir hucs --max_retries 10 --num_jobs 7 --resolution 30
    args = parser.parse_args()

    main(**vars(args))