#!/bin/bash

# Clips the NLCD raster to the HUC8 boundaries for the specified region (PR, HI, AK)

# Paths to the input files
GPKG_FILE="data/spatial_covariates.gpkg"
LAYER_NAME="spatial_covariates"

# Region
REGION="HI"  # Specify the region value to filter by: AK, HI, PR

RASTER_FILE="data/manual_downloads/${REGION}/${REGION}_nlcd.tif"

# Temporary directory for storing intermediate files
TEMP_DIR="temp_gpkg"
mkdir -p "$TEMP_DIR"

# Get the list of HUC8 values for the specified region
HUC8_VALUES=$(ogrinfo -q -geom=NO -al -where "region = '$REGION'" -fields=YES "$GPKG_FILE" "$LAYER_NAME" | grep HUC8 | awk '{print $4}')

# Extract the nodata value from the raster
NODATA_VALUE=$(gdalinfo "$RASTER_FILE" | grep "NoData Value=" | awk -F= '{print $2}')

# check if NODATA_VALUE is empty
if [ -z "$NODATA_VALUE" ]; then
  NODATA_VALUE=0
fi

# Loop through each HUC8 value
for HUC8 in $HUC8_VALUES; do
  # Create a temporary GeoPackage for the current geometry
  TEMP_GPKG="$TEMP_DIR/temp_${HUC8}.gpkg"
  ogr2ogr -f "GPKG" -where "HUC8 = '$HUC8' AND region = '$REGION'" "$TEMP_GPKG" "$GPKG_FILE" "$LAYER_NAME"

  # Check if the GeoPackage has any features (i.e., it intersects with the raster)
  if [ $(ogrinfo "$TEMP_GPKG" | grep -c "Feature Count: 0") -eq 0 ]; then
    # Clip the raster using the geometry from the temporary GeoPackage
    OUTPUT_FILE="data/huc8s/nlcd/nlcd_${HUC8}.tif"
    gdalwarp -cutline "$TEMP_GPKG" -crop_to_cutline -overwrite -dstnodata "$NODATA_VALUE" "$RASTER_FILE" "$OUTPUT_FILE"
    echo "Created $OUTPUT_FILE"

  else
    echo "Skipping $HUC8 - No intersection with raster"
  fi
done

# Clean up temporary files
rm -rf "$TEMP_DIR"