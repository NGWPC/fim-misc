"""
Get spatial covariates (slope and NLCD) for the FIM60 HUC8s.
"""


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


def get_slope(geoseries, resolution=30, geo_crs=4326, crs=5070, m2m=True, to_file=None, id='huc8'):
    """
    Get slope from a geometry using py3dep and xrspatial.
    """
    
    dem = py3dep.get_dem(geoseries.geometry, resolution, geo_crs)
    dem = dem.rio.reproject(crs)
    
    # slope
    slope = xrspatial.slope(dem)
    slope_attrs = 'degrees'
    if m2m:
        slope = py3dep.utils.deg2mpm(slope)
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


def get_nlcd(geoseries, resolution=30, geo_crs=4326, crs=5070, region=None, to_file=None, id='huc8'):
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
    huc8s,
    id,
    huc8s_layer=None,
    output_dir='data',
    huc8s_output_dir='huc8s',
    max_retries=3,
):

    # open the huc8s
    print('Reading huc8s...')
    huc8s = gpd.read_file(huc8s, layer=huc8s_layer).to_crs(4326)

    # make huc8s column for nlcd retrieval
    print('Adding region column to states ...')
    states = gh.get_us_states().to_crs(4326)
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
        .to_crs(4326)
    )

    del states

    output_dir = 'data'
    huc8s_output_dir = os.path.join(output_dir, huc8s_output_dir)
    slope_output_dir = os.path.join(huc8s_output_dir, 'slope')
    nlcd_output_dir = os.path.join(huc8s_output_dir, 'nlcd')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(huc8s_output_dir, exist_ok=True)
    os.makedirs(slope_output_dir, exist_ok=True)
    os.makedirs(nlcd_output_dir, exist_ok=True)

    for _, row in tqdm(huc8s.iterrows(), total=len(huc8s), desc='Processing huc8s'):

        try:
            slope_fn = os.path.join(slope_output_dir, 'slope_{}.tif'.format(row[id]))
            retries = 0
            while retries < max_retries:
                try:
                    # get slope
                    slope = get_slope(
                        row, to_file=slope_fn, id=id, geo_crs=4326, crs=5070
                    )
                except:
                    retries += 1
                    if retries >= max_retries:
                        slope = None
                        #print(f'Failed to get slope for {row[id]}')
                    
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
                        row, to_file=nlcd_fn, id=id, region=row['region'], geo_crs=4326, crs=5070
                    )
                    
                except:
                    retries += 1
                    if retries >= max_retries:
                        #breakpoint()
                        nlcd = None
                        #print(f'Failed to get NLCD for {row[id]}')
                    
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

if __name__ == "__main__":

    huc8s = os.path.join('data', 'ALL_FIM60_HUC8s.gpkg') # os.path.join('data', 'benchmark_maps.gpkg')
    id = 'HUC8' # test_case_id
    huc8s_layer = None
    output_dir = 'data'
    huc8s_output_dir = 'huc8s' # 'benchmarks'
    max_retries = 10

    main(
        huc8s,
        id,
        huc8s_layer=huc8s_layer,
        output_dir=output_dir,
        huc8s_output_dir=huc8s_output_dir,
        max_retries=max_retries
    )