#!/usr/bin/env python

"""
This script processes depth raster data from GeoDatabases (GDB) and aligns them to extent rasters for different flood risk levels. It involves the following steps:

1. Reading HUC identifiers and corresponding GDB GDAL paths from a CSV file.
2. For each HUC, retrieving raster data representing different flood risk levels (e.g., "100yr" and "500yr").
3. Using GDAL commands to get reference info such as spatial reference system, resolution, and extent from reference extent rasters.
4. Warping the original depth rasters to reference raster extent, srs, and resolution.
5. The process runs in parallel across multiple HUCs to optimize performance, using multiprocessing.

Usage:
Command-line arguments are required for configuration, including directories output of depth raster and referencing extent raster, the number of parallel processes, and log level.
Example command:
`python align_depth_rasters.py -o /path/to/output -rd /path/to/reference -pp 4 -ll INFO`

Where:
- -o: Output directory path
- -rd: Base directory containing reference extent rasters
- -pp: Number of parallel processes to launch
- -ll: Log level (e.g., INFO, DEBUG)

Requirements:
- Python 3.6 or higher
- GDAL/OGR with Python bindings
- Access permissions for specified directories and files

The `bfe_hucs_gdal_paths.csv` file should be available adjacent to create.py and contain rows with HUC identifiers and
corresponding GDAL paths to their GeoDatabases.
"""

import argparse
import csv
import logging
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Manager
from typing import Tuple, Optional

from osgeo import gdal, osr


@dataclass
class HUCProcessingRecord:
    """Holds processing record for each huc."""

    huc: str
    start_time: datetime
    status: str = ""
    error: str = ""
    message: str = ""
    end_time: datetime = field(default_factory=datetime.now)

    def update_on_error(self, error_type: str, error_message: str) -> None:
        self.end_time = datetime.now()
        self.error = error_type
        self.status = "failed"
        self.message = error_message

    def update_on_success(self) -> None:
        self.end_time = datetime.now()
        self.status = "success"


def get_depth_raster_path(gdb_gdal_path: str, risk: list) -> str:
    main_dataset = gdal.Open(gdb_gdal_path, gdal.GA_ReadOnly)
    if not main_dataset:
        return ""
    subdatasets = main_dataset.GetSubDatasets()
    for sub in subdatasets:
        ds_name = sub[0].rpartition(":")[2].lower()
        if any(word in ds_name for word in risk) and "dep" in ds_name:
            return sub[0]
    return ""


def get_raster_info(gdal_path: str) -> Tuple[Optional[tuple], Optional[float], Optional[osr.SpatialReference]]:
    ds = gdal.Open(gdal_path, gdal.GA_ReadOnly)
    if not ds:
        return None, None, None

    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + width * gt[1] + height * gt[2]
    ymin = ymax + width * gt[4] + height * gt[5]
    extent = (xmin, ymin, xmax, ymax)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    ds = None
    return extent, max(gt[1], -gt[5]), srs


def process_huc(
    huc: str, gdb_gdal_path: str, reference_dir: str, output_dir: str, log_level: int, log_folder: str, lock
) -> None:
    start_time = datetime.now()
    logger = setup_logging(log_level, f"{log_folder}/{huc}")
    processing_record = HUCProcessingRecord(huc=huc, start_time=start_time)
    logger.info("Starting processing...")

    try:
        # 1. Get Source Raster Extent, Resolution and CRS
        # 100yr and 500yr have same resolution, extent etc
        extent, res, srs = get_raster_info(f"{reference_dir}/{huc}/100yr/ble_huc_{huc}_extent_100yr.tif")
        if res is None:
            logger.error(f"Incorrect Reference Raster path")
            processing_record.update_on_error("FileNotFound", "Reference Raster not found")
            return

        epsg_code_str = f"EPSG:{srs.GetAuthorityCode(None)}"
        # 2. Get Depth Rasters GDAL Paths from GDB

        risks = [
            ["500yr", ["0_2pct"], "output_raster", "output_tmp_raster", ""],
            ["100yr", ["1pct"], "output_raster", "output_tmp_raster", ""],
        ]
        output_hucdir_path = os.path.join(output_dir, huc)

        for risk_value in risks:
            risk_value[2] = os.path.join(output_hucdir_path, risk_value[0], f"ble_huc_{huc}_depth_{risk_value[0]}.tif")
            risk_value[3] = os.path.join(
                output_dir, "tmp", huc, risk_value[0], f"ble_huc_{huc}_depth_{risk_value[0]}.tif"
            )
            risk_value[4] = get_depth_raster_path(gdb_gdal_path, risk_value[1])

            if not risk_value[4]:
                logger.error("Incorrect GDB GDAL path")
                processing_record.update_on_error("FileNotFound", "GDB not found")
                return

        # 3. Align Raster
        for risk_value in risks:

            if os.path.exists(risk_value[2]):
                logger.info(f"Processing skipped as outputs already exist for {risk_value[0]}")
                continue

            logger.info(f"Aligning depth map for {risk_value[0]}...")
            output_tmp_raster = risk_value[3]
            os.makedirs(os.path.dirname(output_tmp_raster), exist_ok=True)
            gdal_warp_cmd = [
                "gdalwarp",
                "-overwrite",
                "-t_srs",
                epsg_code_str,
                "-tr",
                str(res),
                str(res),
                "-r",
                "bilinear",
                "-te",
                str(extent[0]),
                str(extent[1]),
                str(extent[2]),
                str(extent[3]),
                "-of",
                "COG",
                "-dstnodata",
                "-9999",
                risk_value[4],
                output_tmp_raster,
            ]
            try:
                result = subprocess.run(gdal_warp_cmd, check=True, text=True, capture_output=True)
                logger.info(f"Raster aligned successfully: {result.stdout}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to align raster: {e.stderr}")
                processing_record.update_on_error("SubprocessError", str(e))
                return

            # 4. Move rasters to final location after everything is complete
            # so as to distinguish from rasters that are incomplete
            logger.info(f"Moving Raster...")
            os.makedirs(os.path.dirname(risk_value[2]), exist_ok=True)
            shutil.move(output_tmp_raster, risk_value[2])

            # 5.Update Raster Statistics as raster statistics (not important)
            logger.info(f"Updating raster stats {risk_value[0]}...")
            gdalinfo_cmd = [
                "gdalinfo",
                "-stats",
                risk_value[2],
            ]
            try:
                result = subprocess.run(gdalinfo_cmd, check=True, text=True, capture_output=True)
                logger.info(f"Raster stats updated successfully: {result.stdout}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to update raster stats: {e.stderr}")

        logger.info(f"Completed in {datetime.now() - start_time}")
        processing_record.update_on_success()

    except Exception as e:
        logger.error(f"{huc}: {str(e)}")
        processing_record.update_on_error("UnknownError", str(e))

    finally:
        with lock:
            with open(f"{log_folder}/hucs.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        processing_record.huc,
                        processing_record.status,
                        processing_record.error,
                        processing_record.message,
                        processing_record.start_time,
                        processing_record.end_time,
                    ]
                )


def setup_logging(log_level: int, name: str) -> logging.Logger:
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Create a new logger for tihs name
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Configure file handler
    file_handler = logging.FileHandler(f"{name}.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    )

    logger.addHandler(file_handler)
    logger.propagate = False

    return logger


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Align BLE Benchmark Depth Rasters to Extent Rasters")
    parser.add_argument("-o", "--output_dir", required=True, type=str, help="Directory path for output data.")
    parser.add_argument(
        "-rd",
        "--reference_dir",
        required=True,
        type=str,
        help="Base directory path containing the reference extent rasters.",
    )
    parser.add_argument(
        "-pp", "--parallel_processes_count", default=None, type=int, help="Number of hucs to process simultaneously."
    )
    parser.add_argument(
        "-ll", "--log_level", default="INFO", type=str, help="Set the logging level (e.g., INFO, DEBUG)."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    run_time = datetime.now()
    run_time_str = run_time.strftime("%Y_%m_%d_%H_%M_%S")

    os.makedirs(run_time_str)
    log_level = getattr(logging, args.log_level.upper(), None)
    logger = setup_logging(log_level, f"{run_time_str}/main")

    # Setup huc processing records CSV
    with open(f"{run_time_str}/hucs.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["huc", "status", "error", "message", "start_time", "end_time"])

    # Load gdal_paths from CSV
    hucs_list = []
    with open("bfe_hucs_gdal_paths.csv", mode="r") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)  # skip the header
        hucs_list = [row for row in csvreader]

    lock = Manager().Lock()

    with ProcessPoolExecutor(max_workers=args.parallel_processes_count) as executor:
        for row in hucs_list:
            executor.submit(
                process_huc, row[0], row[1], args.reference_dir, args.output_dir, log_level, run_time_str, lock
            )

    logger.info(f"Completed in {datetime.now() - run_time}")


if __name__ == "__main__":
    main()
