#!/usr/bin/env python

"""
This script is designed to create Benchmark Inundation Maps for specific Hydrologic Unit Codes (HUCs) based on
geospatial data from GeoDatabases (GDB). The primary functionality includes:

1. Reading configurations and HUC paths from CSV files.
2. Checking the raster resolution and coordinate reference system (CRS) for compliance with provided parameters.
3. Reprojecting "FLD_HAZ_AR" vector layers to a desired output CRS.
4. Creating inundation maps for specified risk levels (e.g., "High" > 100yr and "Moderate" > 500yr).
5. Running the process in parallel for multiple HUCs to optimize performance.

The script employs multiprocessing to handle each HUC in a separate process, enhancing efficiency for large datasets.
GDAL/OGR compiled binaries are used to fast track the process.
Logging is implemented to track the process and record any errors or warnings.

Usage:
The script requires several command-line arguments for its configuration, including output directory, output CRS, output
resolution, maximum source resolution, source units, and the number of parallel processes. These arguments are specified
when running the script from the command line.

Requirements:
- Python 3.6 or higher
- GDAL/OGR with Python bindings
- Necessary permissions to access and write to the specified directories and files

The `bfe_hucs_gdal_paths.csv` file should be avialable adjacent to main.py and contain rows with HUC identifiers and
corresponding GDAL paths to their GeoDatabases.

Example:
`python main.py -o /path/to/output -oc EPSG:5070 -or 3 -smr 10 -su feet -pp 4 -ll INFO`

Where:
- -o: Output directory path
- -oc: Output CRS (e.g., EPSG:5070)
- -or: Output resolution in integer
- -smr: Source maximum resolution in integer
- -su: Source units (e.g., feet)
- -pp: Number of parallel processes to launch
- -ll: Log level (e.g., INFO, DEBUG)
"""


import argparse
import csv
import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Manager
from timeit import default_timer as timer

from osgeo import gdal, osr, ogr


@dataclass
class HUCProcessingRecord:
    """Holds processing record for each huc."""

    huc: str
    start_time: datetime
    status: str = ""
    error: str = ""
    message: str = ""
    end_time: datetime = field(default_factory=datetime.now)

    def update_on_error(self, error_type: str, error_message: str):
        self.end_time = datetime.now()
        self.error = error_type
        self.status = "failed"
        self.message = error_message

    def update_on_success(self):
        self.end_time = datetime.now()
        self.status = "success"

def has_specific_risk_values(gdb_path, layer_name, risk_values):
    dataSource = ogr.Open(gdb_path)
    if dataSource is None:
        return False

    sql_query = f"""SELECT EST_Risk FROM {layer_name} WHERE EST_Risk IN ({', '.join(f"'{val}'" for val in risk_values)}) LIMIT 1"""
    layer = dataSource.ExecuteSQL(sql_query)
    if layer is None:
        return False

    feature = layer.GetNextFeature()
    if feature is None:
        return False

    dataSource.ReleaseResultSet(layer)
    return True


def get_raster_resolution(gdal_path):
    main_dataset = gdal.Open(gdal_path, gdal.GA_ReadOnly)
    if not main_dataset:
        return None, None
    subdatasets = main_dataset.GetSubDatasets()
    dataset_name = subdatasets[0][0] # checking first subdataset
    dataset = gdal.Open(dataset_name, gdal.GA_ReadOnly)
    gt = dataset.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(dataset.GetProjection())
    dataset = None
    return max(gt[1], -gt[5]), srs


units_map = {"feet": ["us survey foot"], "meter": ["meter", "metre"]}


def is_resolution_in_units(srs, units):
    return srs.GetLinearUnitsName().lower() in units_map[units]


def process_huc(
    huc,
    gdb_gdal_path,
    source_max_resolution,
    source_max_resolution_units,
    output_crs,
    output_resolution,
    output_dir,
    log_level,
    log_folder: str,
    lock,
) -> None:

    start_time = datetime.now()

    logger = setup_logging(log_level, f"{log_folder}/{huc}")
    processing_record = HUCProcessingRecord(huc=huc, start_time=start_time)
    logger.info("Starting processing...")
    try:
        risks = [["500yr", ["M", "Moderate", "MODERATE", "Low or Moderate", "L"], ""], ["100yr", ["H", "High", "HIGH"], ""]]
        output_hucdir_path = os.path.join(output_dir, huc)
        tmp_dir_path = os.path.join(output_dir, "tmp")

        for risk_value in risks:
            output_raster = os.path.join(output_hucdir_path, risk_value[0], f"ble_huc_{huc}_extent_{risk_value[0]}.tif")
            risk_value[2] = output_raster

        if all([os.path.exists(risk_value[2]) for risk_value in risks]):
            logger.info(f"Processing skipped as outputs already exist")
            processing_record.update_on_success()
            return


        # 1. Check Raster Resolution and CRS
        res, srs = get_raster_resolution(gdb_gdal_path)
        if res is None:
            logger.error(f"Incorrect gdal path")
            processing_record.update_on_error("FileNotFound", "")
            return
        if not is_resolution_in_units(srs, source_max_resolution_units) or res > source_max_resolution:
            logger.error(f"Incorrect resolution or SRS: {res} | {srs}")
            processing_record.update_on_error("IncorrectResolution", "")
            return

        # 2. Check floodplain has at least one attribute with EST_Risk field having known risk_values
        for risk_value in risks:
            if not has_specific_risk_values(gdb_path=gdb_gdal_path, layer_name="FLD_HAZ_AR", risk_values=risk_value[1]):
                logger.error("ZeroValidRiskValues")
                processing_record.update_on_error("Does not have any feature with valid risk value", "")
                return

        logger.info("Reprojecting floodplain...")
        # 3. Reproject Vector Layer
        reprojected_vector = os.path.join(tmp_dir_path, f"{huc}.shp")
        os.makedirs(os.path.dirname(reprojected_vector), exist_ok=True)
        ogr_cmd = ["ogr2ogr", "-t_srs", output_crs, reprojected_vector, gdb_gdal_path, "FLD_HAZ_AR"]
        subprocess.run(ogr_cmd, check=True)


        # 4. Convert Vector to Raster for each EST_Risk value
        for risk_value in risks:
            logger.info(f"Creating inundation map for {risk_value[0]}...")
            output_raster = risk_value[2]
            os.makedirs(os.path.dirname(output_raster), exist_ok=True)
            gdal_rasterize_cmd = [
                "gdal_rasterize",
                "-burn",
                "1",
                "-where",
                f"""EST_Risk IN ({', '.join(f"'{val}'" for val in risk_value[1])})""",
                "-at",
                "-tr",
                str(output_resolution),
                str(output_resolution),
                "-ot",
                "Byte",
                "-co",
                "COMPRESS=LZW",
                reprojected_vector,
                output_raster,
            ]
            subprocess.run(gdal_rasterize_cmd, check=True)

        # 5. Merge 100yr floodplain into 500yr
        logger.info(f"Merging 100yr into 500yr...")
        gdal_merge_cmd = [
            "gdal_calc.py",
            "-A",
            risks[0][2],
            "-B",
            risks[1][2],
            f"--outfile={risks[0][2]}",
            "--co",
            "COMPRESS=LZW",
            '--calc="A+B>0"',
            "--overwrite",
        ]
        subprocess.run(gdal_merge_cmd, check=True)

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
    parser = argparse.ArgumentParser(description="Create BLE Benchmark Inundation Maps at Specified Resolution")
    parser.add_argument("-o", "--output_dir", required=True, type=str, help="Directory path for output data.")
    parser.add_argument("-oc", "--output_crs", required=True, type=str, help="")
    parser.add_argument("-or", "--output_resolution", required=True, type=int, help="")
    parser.add_argument("-smr", "--source_max_resolution", required=True, type=int, help="")
    parser.add_argument("-su", "--source_units", required=True, type=str, help="")
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
        next(csvreader, None)  # Skip the header
        for row in csvreader:
            hucs_list.append(row)

    m = Manager()
    lock = m.Lock()

    logger.info(f"Executing individual hucs...")
    with ProcessPoolExecutor(max_workers=args.parallel_processes_count) as executor:
        for row in hucs_list:
            executor.submit(
                process_huc,
                row[0],
                row[1],
                args.source_max_resolution,
                args.source_units,
                args.output_crs,
                args.output_resolution,
                args.output_dir,
                log_level,
                run_time_str,
                lock,
            )

    logger.info(f"Completed in {datetime.now() - run_time}")


if __name__ == "__main__":
    main()
