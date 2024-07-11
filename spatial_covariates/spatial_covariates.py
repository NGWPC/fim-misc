
import os

from pynhd import NLDI
import pygeohydro as gh
from py3dep import py3dep
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

# quiet future warnings on import 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

with warnings.catch_warnings():
    import xrspatial


def get_slope(geoseries, resolution=30, geo_crs=5070, crs=5070, m2m=True, to_file=None, id='huc8'):
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


def get_nlcd(geoseries, resolution=30, geo_crs=5070, crs=5070, to_file=None, id='huc8'):
    """
    Get NLCD from a geometry using py3dep.
    """

    # get NLCD
    nlcd = gh.nlcd.nlcd_bygeom(
        gpd.GeoSeries([geoseries.geometry], crs=geo_crs), resolution, years={'cover':2021}, crs=geo_crs
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


def compute_nlcd_high_dev(nlcd):
    """
    Compute the high developed NLCD class from NLCD data.
    """
    # find frequency of high developed NLCD class
    nlcd_high_dev = nlcd.where(nlcd == 24)
    return nlcd_high_dev.count().values.item() / nlcd.count().values.item()

def compute_median_slope(slope):
    """
    Compute the median slope from slope data.
    """
    return slope.median().values.item()

if __name__ == "__main__":
    # get the geometry
    #wbd = gh.watershed.WBD('huc8', crs=5070)
    #regions = wbd.byids('huc8', ['12090301', '12090302', '12040301'])
    #regions = regions.to_crs(5070)

    regions = os.path.join('data', 'WBD_National_EPSG_5070.gpkg')
    id = 'HUC8'

    # open the regions
    regions = gpd.read_file(regions, layer='WBDHU8')
    regions = regions.head(3)

    # make hucs_df to store huc_code, median_slope, freq_high_dev
    hucs_df = pd.DataFrame(columns=[id, 'median_slope', 'freq_high_dev'], index=range(len(regions)))

    for idx, row in tqdm(regions.iterrows(), total=len(regions)):

        # get slope
        slope = get_slope(row, to_file=os.path.join('data', 'regions','slope_{}.tif'), id=id)
        median_slope = compute_median_slope(slope)

        # get NLCD
        nlcd = get_nlcd(row, to_file=os.path.join('data', 'regions','nlcd_{}.tif'), id=id)
        freq_high_dev = compute_nlcd_high_dev(nlcd)

        # store in hucs_df
        hucs_df.loc[idx] = [row[id], median_slope, freq_high_dev]
        
    print(hucs_df)

    # merge in hucs_df to hucs
    regions = regions.merge(hucs_df, on=id)

    # write hucs gpkg
    regions.to_file(os.path.join('data', 'spatial_covariates.gpkg'), driver='GPKG', index=False)

    # write to csv
    hucs_df.to_csv(os.path.join('data', 'spatial_covariates.csv'), index=False)