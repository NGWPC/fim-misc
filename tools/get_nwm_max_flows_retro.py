import os
import argparse
import xarray as xr
import fsspec
import numpy as np
import dask
from pyproj import Transformer

#s3://noaa-nwm-retrospective-2-1-pds/

#URL = "s3://noaa-nwm-retrospective-2-1-zarr-pds/"
URL = "s3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr"

#arn:aws:s3:::noaa-nwm-retrospective-2-1-zarr-pds


def compute_max_flows(start_datetime, end_datetime, output_flow_file, bounds=""):

    if not os.access(os.path.dirname(output_flow_file), os.W_OK):
        print(f"You do not have permissions to write to the directory: " + os.path.dirname(output_flow_file))
        return

    print("Connecting to NWM Retro S3...")
    ds = xr.open_zarr(fsspec.get_mapper(URL, anon=True), consolidated=True)
    #print(ds)
    #quit()

    # crs = ds.attrs.get('crs', None)
    
    # print(ds)
    # print(bounds)
    # print(f"CRS from global attributes: {crs}")
    # #quit()
    # #print(ds.coords)
    # #print(ds.dims)
    
    # if len(bounds) == 4:
    #     print("Filtering results to provided bounds...")
    #     lon_min, lat_min, lon_max, lat_max = bounds

    #     # Create a transformer to convert from EPSG:4326 to the specified LCC projection
    #     print("Transforming bounding box coords...")
    #     transformer_to_lcc = Transformer.from_proj(
    #         proj_from='epsg:4326',  # Source CRS
    #         proj_to="+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@"
    #     )
    #     print(lon_min)
    #     print(lat_min)

    #     new_lat_min, new_lon_min = transformer_to_lcc.transform(lat_min, lon_min)
    #     new_lat_max, new_lon_max = transformer_to_lcc.transform(lat_max, lon_max)

    #     print(lon_max)
    #     print(lat_max)

    #     print(new_lat_min)
    #     print(new_lon_min)
    #     print(new_lat_max)
    #     print(new_lon_max)
    #     ds = ds.where((ds.latitude >= new_lat_min) & (ds.latitude <= new_lat_max) & (ds.longitude >= new_lon_min) & (ds.longitude <= new_lon_max), drop=True)
    #     #ds = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

    #print(ds)

    # Get max streamflow
    print("Extracting max flows...")
    period_streamflow = ds['streamflow'].sel(time=slice(start_datetime, end_datetime))
    print(f'Size of streamflow variable for provided time range: {period_streamflow.nbytes/1e9:.1f} GB')

    var_max = period_streamflow.max(dim='time').compute()

    # Assuming var_max is a pandas Series after the .compute()
    df = var_max.to_pandas().to_frame(name='discharge')

    print("Writing discharge file: " + output_flow_file)
    df.to_csv(output_flow_file)

    print("Max flows generation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produce CSV flow files from NWM 2.1 Retrospective.')
    parser.add_argument(
        '-s',
        '--start-datetime',
        help='The start date and time. Format: YYYY-MM-DD HH:MM',
        required=True,
    )
    parser.add_argument(
        '-e',
        '--end-datetime',
        help='The end date and time. Format: YYYY-MM-DD HH:MM',
        required=True,
    )
    parser.add_argument(
        '-o',
        '--output-flow-file',
        help='Full path to output flow file. A CSV with feature_id and discharge (in CMS) is written.',
        required=True,
    )
    args = vars(parser.parse_args())
    compute_max_flows(**args)