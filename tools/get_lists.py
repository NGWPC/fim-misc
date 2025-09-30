import geopandas as gpd

# Path to the GeoPackage and layer name
gpkg_path = "/efs/fim-data/hand_fim/inputs/wbd/WBD_National_EPSG_5070.gpkg"
layer_name = "WBDHU12"

# List of HUC8 codes to filter by
huc8_list = {
    "01090004", "01090005", "03050109", "05100201",
    "07080203", "07080205", "07080206", "07080208",
    "07080209", "07140104", "08040207", "08050001", "08080103"
}

gdf = gpd.read_file("/efs/fim-data/hand_fim/inputs/wbd/WBD_National_EPSG_5070.gpkg", 
                    layer="WBDHU12", 
                    rows=1)  # Only read first row
print("Columns:", gdf.columns.tolist())

# Read the layer
gdf = gpd.read_file(gpkg_path, layer=layer_name)

# Ensure 'huc12' exists
if "HUC12" not in gdf.columns:
    raise ValueError("Column 'huc12' not found in the GeoPackage layer.")

# Force 'HUC12' to string and strip whitespace just in case
gdf['HUC12'] = gdf['HUC12'].astype(str).str.strip()

# Filter rows where first 8 characters of HUC12 are in the huc8 list
gdf_filtered = gdf[gdf['HUC12'].str[:8].isin(huc8_list)]

# Get unique, sorted lists of HUC12s and HUC10s
huc12_list = sorted(gdf_filtered['HUC12'].unique())
huc10_list = sorted(gdf_filtered['HUC12'].str[:10].unique())

# Write HUC12 list
with open("/efs/fim-data/hand_fim/temp/brad/hwm_mutual_huc12.lst", "w") as f:
    for huc12 in huc12_list:
        f.write(f"{huc12}\n")

# Write HUC10 list
with open("/efs/fim-data/hand_fim/temp/brad/hwm_mutual_huc10.lst", "w") as f:
    for huc10 in huc10_list:
        f.write(f"{huc10}\n")

print("Finished writing huc12.lst and huc10.lst.")
