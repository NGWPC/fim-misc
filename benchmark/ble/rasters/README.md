## Purpose
Read python files' docstrings

## Getting Started

0. **Create `bfe_hucs_gdal_paths.csv`:**
   Use sample file provided to create `bfe_hucs_gdal_paths.csv`

1. **Start Docker Compose:**
   Navigate to this folder and start the Docker container using Docker Compose:
   ```bash
   docker-compose up -d
   ```

2. **Access the Docker Container:**
   Once the container is running, you can access its bash shell:
   ```bash
   docker exec -it gdal_ops /bin/bash
   ```

3. **Run the Script:**
   Inside the Docker container, navigate to the `/src` directory (if not already there) and run the `main.py` script:
   ```bash
   python create_extent_rasters.py -o /data/outputs -oc EPSG:5070 -or 3 -smr 10 -su feet -pp 4 -ll INFO
   python align_depth_rasters.py -o /data/outputs -rd /vsis3/fimc-data/benchmark/high_resolution_validation_data_ble  -pp 4 -ll INFO
   ```
   Replace the arguments with appropriate values as needed.