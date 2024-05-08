## Purpose
Provide scripts to create high resolution, standardized BLE inundation and depth rasters from source BLE Geodatabases. Read python files' docstrings for further information..

## Getting Started

It is assumed that docker is installed and running prior to execution of following commands.

1. **Create `.env`:**
   Use example env file provided to create the environment file if working with S3 data, else create a blank env file or comment out env part in the docker compose.

1. **Create `bfe_hucs_gdal_paths.csv`:**
   Use sample file provided to create `bfe_hucs_gdal_paths.csv`. This file is being used as input arguments. Each record in this CSV will be processed.

1. **Start Docker Compose:**
   Navigate to this folder and start the Docker container using Docker Compose:
   ```bash
   docker-compose up -d
   ```

1. **Access the Docker Container:**
   Once the container is running, you can access its bash shell:
   ```bash
   docker exec -it gdal_ops /bin/bash
   ```

1. **Run the Script:**
   Inside the Docker container, navigate to the `/src` directory (if not already there) and run the scripts:
   ```bash
   python create_extent_rasters.py -o /data/extent-outputs -oc EPSG:5070 -or 3 -smr 10 -su feet -pp 4 -ll INFO
   python align_depth_rasters.py -o /data/depth-outputs -rd /data/extent-outputs -pp 4 -ll INFO
   ```
   Replace the arguments with appropriate values as needed.