"""
Retrieve benchmark data from S3 and prepare a random variant for the candidate map

Must have AWS credentials configured to access the S3 bucket.

Example usage:
python prepare_test_data.py \
    --bucket fimc-data \
    --object-key benchmark/high_resolution_validation_data_ble/12090301/500yr/ble_huc_12090301_depth_500yr.tif \
    --local-input data/benchmark_ble_huc_12090301_depth_500yr.tif \
    --local-output data/candidate_dummy_huc_12090301_depth_500yr.tif \
    --num-workers 6
"""
from __future__ import annotations

import argparse
import os
import sys

import pyproj

# Set PROJ and GDAL data paths for odc-geo
os.environ["PROJ_LIB"] = os.path.join(sys.prefix, "share", "proj")
os.environ["GDAL_DATA"] = os.path.join(sys.prefix, "share", "gdal")

# Ensure pyproj uses the correct PROJ data directory
pyproj.datadir.set_data_dir(os.environ["PROJ_LIB"])

from tqdm import tqdm
import boto3
import numpy as np
import rioxarray as rxr
import dask.array as da
import dask
import odc.geo
from odc.geo.xr import ODCExtensionDa
from pyproj import CRS

ODCExtensionDa


def download_from_s3(
    bucket_name: str, object_key: str, local_file_path: str | os.PathLike
) -> str | os.PathLike:
    """
    Download a file from S3 with a progress bar.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket.
    object_key : str
        Key of the object to download.
    local_file_path : str or os.PathLike
        Local path to save the downloaded file.
    
    Returns
    -------
    local_file_path : str or os.PathLike
        Path to the downloaded file.    
    """
    s3 = boto3.client('s3')

    #s3.download_file(bucket_name, object_key, local_file_path)
    meta = s3.head_object(Bucket=bucket_name, Key=object_key)
    total = meta['ContentLength']

    # tqdm progress bar
    with tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading"
    ) as pbar:
        def progress_hook(bytes_amount):
            pbar.update(bytes_amount)

        with open(local_file_path, "wb") as f:
            s3.download_fileobj(bucket_name, object_key, f, Callback=progress_hook)

    print(f"Downloaded {object_key} from bucket {bucket_name} to {local_file_path}")
    return local_file_path

def prepare_candidate_map(
    input_raster_path: str | os.PathLike,
    output_raster_path: str | os.PathLike,
    random_seed: int | None = 42,
    noise_scale: float | int = 0.1,
    chunk_size: int | None = None,
) -> str | os.PathLike:
    """
    Out-of-core candidate map with rioxarray+dask.
    - Lazily reads with chunks
    - Adds Gaussian noise per chunk (no global noise array)
    - Writes window-by-window; tile size aligned to chunk size

    Parameters
    ----------
    input_raster_path : str or os.PathLike
        Path to the input raster file.
    output_raster_path : str or os.PathLike
        Path to save the output raster file.
    random_seed : int, optional
        Random seed for noise generation, by default 42
    noise_scale : float, optional
        Standard deviation of the Gaussian noise to add, by default 0.1

    Returns
    -------
    output_raster_path : str or os.PathLike
        Path to the output raster file.
    """
    # Open lazily; keep masked_and_scaled behavior
    da_in = rxr.open_rasterio(
        input_raster_path,
        masked_and_scaled=True,
        chunks="auto" if chunk_size is None else {"y": chunk_size, "x": chunk_size},
    ).squeeze()  # drop band if single-band

    # only for debugging. first five blocks of rows and columns
    #da_in = da_in.isel(y=slice(0, chunk_size), x=slice(0, chunk_size))

    def _add_noise_block(block, block_info=None, seed=None, nodata=da_in.rio.nodata):
        # block_info[None]['array-location'] is a tuple of (start, stop) per axis
        loc = block_info[None]["array-location"]
        if len(loc) == 3:         # (band, y, x)
            _, y_loc, x_loc = loc
        else:                     # (y, x)
            y_loc, x_loc = loc
        y0 = 0 if y_loc[0] is None else y_loc[0]
        x0 = 0 if x_loc[0] is None else x_loc[0]

        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, noise_scale, size=block.shape).astype("float32")

        # Add only to finite values; preserve NaNs
        out = np.float32(block + noise)

        # overwrite nans to ensure no precision issues
        out = np.where(block == nodata, nodata, out)

        return out

    # Apply per chunk (no full-size noise)
    data_noisy = da.map_blocks(_add_noise_block, da_in.data, dtype="float32", seed=random_seed, nodata=da_in.rio.nodata)

    # Only for debugging, used to test _add_noise_block
    #breakpoint()
    #_add_noise_block(da_in.isel(y=slice(0, chunk_size), x=slice(0, chunk_size)).data, block_info={
    #    None: {'array-location': ((0, chunk_size), (0, chunk_size))}, 
    #}, seed=random_seed, nodata=da_in.rio.nodata)
    # Keep coords/attrs/CRS by copying the original DataArray and swapping data

    # Create output DataArray
    da_out = da_in.copy(data=data_noisy)

    # Set nodata properly
    da_out.rio.write_nodata(np.float32(np.nan), inplace=True, encoded=True)
    da_out.rio.write_nodata(da_in.rio.nodata, inplace=True, encoded=False)

    # Remove _FillValue and missing_value to avoid conflicts
    for key in ("_FillValue", "missing_value"):
        if key in da_out.attrs:
            del da_out.attrs[key]
        if key in da_out.encoding:
            del da_out.encoding[key]

    # Write block-wise; align GeoTIFF tile size to chunk size to avoid rechunking
    da_out.rio.to_raster(
        output_raster_path,
        dtype="float32",
        driver="COG",
        tiled=True,
        blockxsize=da_out.chunks[1][0],
        blockysize=da_out.chunks[0][0],
        windowed=True,
        compress='LZW',
        BIGTIFF="IF_SAFER",
        nodata=da_in.rio.nodata
    )

    print(f"Candidate map saved to {output_raster_path}")
    return output_raster_path

def prepare_nonhomogenized_candidate_map(
    candidate_map_path: str | os.PathLike, chunk_size: int | None = None
) -> str | os.PathLike:
    """
    Prepare a perturbed candidate map that changes extent (by removing the top row),
    resolution to 10m, and CRS to WGS84.

    Note:
    Can be slow. Potential to use:
        gdalwarp -t_srs EPSG:4326 -r bilinear -srcnodata -9999 -dstnodata -9999 -multi -wo NUM_THREADS=6 -wm 1024 --config GDAL_CACHEMAX 512 -co COMPRESS=LZW -co TILED=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 -co COPY_SRC_OVERVIEWS=YES -co BIGTIFF=IF_SAFER -co SPARSE_OK=YES -co OVERVIEWS=AUTO -of COG -co PREDICTOR=2 -wo PROGRESS=YES data/candidate_dummy_huc_12090301_depth_500yr.tif data/candidate_dummy_huc_12090301_depth_500yr_nonhomogenized.tif

    Parameters
    ----------
    candidate_map_path : str or os.PathLike
        Path to the candidate map raster file.
    chunk_size : int, optional
        Chunk size for dask arrays, by default 512
    
    Returns
    -------
    str or os.PathLike
        Path to the non-homogenized candidate map raster file.
    """
    
    # Open candidate map lazily
    da_in = rxr.open_rasterio(
        candidate_map_path,
        masked_and_scaled=True,
        chunks="auto" if chunk_size is None else {"y": chunk_size, "x": chunk_size},
    )
    
    # Squeeze, remove top row and rightmost column and reproject to WGS84 in one step
    da_reprojected = (
        da_in
        .squeeze()  # drop band if single-band
        #.isel(y=slice(0, chunk_size*8), x=slice(0, chunk_size*8))  # only for debugging
        .isel(y=slice(1, None), x=slice(0, -1)) # remove top row and rightmost column
        .odc.reproject(CRS.from_epsg(4326), resampling="bilinear") # reproject to WGS84
    )

    # Write to file
    output_path = candidate_map_path.replace(".tif", "_nonhomogenized.tif")
    
    da_reprojected.rio.to_raster(
        output_path,
        dtype="float32",
        driver="COG",
        tiled=True,
        blockxsize=da_reprojected.chunks[1][0],
        blockysize=da_reprojected.chunks[0][0],
        windowed=True,
        compress='LZW',
        BIGTIFF="IF_SAFER",
        nodata=da_in.rio.nodata
    )
    print(f"Non-homogenized candidate map saved to {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Download benchmark data and prepare candidate map.")

    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--object-key", required=True, help="S3 object key for the benchmark raster")
    parser.add_argument(
        "--local-input", default="data/benchmark_map.tif"
    )
    parser.add_argument(
        "--local-output", default="data/candidate_map.tif", help="Local path to save the candidate map"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of Dask threads/workers to use"
    )
    args = parser.parse_args()

    dask.config.set(scheduler="threads", num_workers=args.num_workers)
    
    # Download benchmark data from S3
    download_from_s3(args.bucket, args.object_key, args.local_input)

    # Prepare candidate map
    prepare_candidate_map(args.local_input, args.local_output)

    prepare_nonhomogenized_candidate_map(args.local_output)

if __name__ == "__main__":
    main()