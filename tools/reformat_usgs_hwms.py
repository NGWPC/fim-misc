import os
import argparse
import pandas as pd
import geopandas as gpd
import requests

import warnings
warnings.filterwarnings('ignore')

fields_to_remove = ['approvalMember', 'flagMemberName', 'surveyMemberName', 'flag_member_id', 'survey_member_id', 'last_updated_by', 'survey_member', 'flag_member']

#######################################################################
# Function to get conversion adjustment NGVD to NAVD in FEET
#######################################################################
def ngvd_to_navd_ft(lat, lon, region='contiguous'):
    '''
    Given the lat/lon, retrieve the adjustment from NGVD29 to NAVD88 in feet.
    Uses NOAA tidal API to get conversion factor. Requires that lat/lon is
    in NAD27 crs. If input lat/lon are not NAD27 then these coords are
    reprojected to NAD27 and the reproject coords are used to get adjustment.
    There appears to be an issue when region is not in contiguous US.

    Parameters
    ----------
    lat : FLOAT
        Latitude.
    lon : FLOAT
        Longitude.

    Returns
    -------
    datum_adj_ft : FLOAT
        Vertical adjustment in feet, from NGVD29 to NAVD88, and rounded to nearest hundredth.

    '''

    # Define url for datum API
    datum_url = 'https://vdatum.noaa.gov/vdatumweb/api/convert'

    # Define parameters. Hard code most parameters to convert NGVD to NAVD.
    params = {}
    params['lat'] = lat
    params['lon'] = lon
    params['region'] = region
    params['s_h_frame'] = 'NAD27'  # Source CRS
    params['s_v_frame'] = 'NGVD29'  # Source vertical coord datum
    params['s_vertical_unit'] = 'm'  # Source vertical units
    params['src_height'] = 0.0  # Source vertical height
    params['t_v_frame'] = 'NAVD88'  # Target vertical datum
    params['tar_vertical_unit'] = 'm'  # Target vertical height

    # Call the API
    response = requests.get(datum_url, params=params, verify=False)

    # If successful get the navd adjustment
    if response:
        results = response.json()
        # Get adjustment in meters (NGVD29 to NAVD88)
        adjustment = results['t_z']
        # convert meters to feet
        adjustment_ft = round(float(adjustment) * 3.28084, 2)
    else:
        adjustment_ft = None
    return adjustment_ft


def subset_hwms_by_event(usgs_hwm_csv, output_directory):

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Load the dataset
    data = pd.read_csv(usgs_hwm_csv)

    # Segregating the data based on unique values in "horizontalDatumName"
    unique_horizontal_datums = data['horizontalDatumName'].unique()

    # Example CRS mappings for common datums
    # These are general mappings and might need to be adjusted for specific cases
    crs_mappings = {
        'WGS84 (from Digital Map)': 'EPSG:4326',
        'NAD83': 'EPSG:4269',
        'local control point': None,  # Need specific EPSG code
        'NAD 83 (NSRS2007) epoch 2007': 'EPSG:4269',  # Need specific EPSG code
        'NAD27': 'EPSG:4267',
        'NAD 83 (2011) epoch 2010': 'EPSG:4269'  # Need specific EPSG code
        # Add additional mappings as needed
    }

    # Segregating the data based on unique values in "horizontalDatumName"
    unique_horizontal_datums = data['horizontalDatumName'].unique()

    # Loop through unique horizontal datums
    for unique_horiz_datum in unique_horizontal_datums:
        horiz_subset_df = data[data['horizontalDatumName'] == unique_horiz_datum]

        # Indentify the CRS associated with the event to skip control points
        event_crs = crs_mappings[unique_horizontal_datums[0]]
        if event_crs == None:
            continue  # Exclude control points

        # Create geodataframe
        horiz_subset_gdf = gpd.GeoDataFrame(horiz_subset_df, geometry=gpd.points_from_xy(horiz_subset_df.longitude_dd, horiz_subset_df.latitude_dd), crs=event_crs)
        
        # Reproject to EPSG:4326
        horiz_subset_gdf_proj = horiz_subset_gdf.to_crs('EPSG:4326')
        horiz_subset_handle = unique_horiz_datum.lower().replace(':','_')
        output_file = os.path.join(output_directory, horiz_subset_handle + ".gpkg")

        # Write file
        horiz_subset_gdf_proj.to_file(output_file, driver='GPKG')

    print("Merging all files...")
     # List all GeoPackage files in the directory
    gpkg_files = [os.path.join(output_directory, f) for f in os.listdir(output_directory) if f.endswith('.gpkg')]

    # Read all GeoPackage files and concatenate them into a single GeoDataFrame
    gdf_list = [gpd.read_file(gpkg) for gpkg in gpkg_files]
    combined_gdf = pd.concat(gdf_list, ignore_index=True)

    # Drop PII fields
    combined_gdf.drop(columns=fields_to_remove, axis=1, inplace=True)

    combined_gdf['lat4326'] = combined_gdf.geometry.y
    combined_gdf['lon4326'] = combined_gdf.geometry.x

    print("Standardizing vertical datums...")

    # -- Standardize vertical datums -- #

    combined_gdf['adj_elev_ft'] = combined_gdf['elev_ft']
    combined_gdf['datum_offset_ft'] = 0.0

    # Define url for datum API
    datum_url = 'https://vdatum.noaa.gov/vdatumweb/api/convert'
    print("Adjusting datums...")

    # Do datum offset
    for index, row in combined_gdf.iterrows():
    
        lat = row['lat4326']
        lon = row['lon4326']
        vert_datum = row['verticalDatumName']

        if vert_datum == None:
            continue

        if 'NGVD29' not in vert_datum:
            continue

        # Define parameters. Hard code most parameters to convert NGVD to NAVD.
        params = {}
        params['lat'] = lat
        params['lon'] = lon
        params['region'] = 'contiguous'
        params['s_h_frame'] = 'NAD27'  # Source CRS
        params['s_v_frame'] = 'NGVD29'  # Source vertical coord datum
        params['s_vertical_unit'] = 'm'  # Source vertical units
        params['src_height'] = 0.0  # Source vertical height
        params['t_v_frame'] = 'NAVD88'  # Target vertical datum
        params['tar_vertical_unit'] = 'm'  # Target vertical height

        # Call the API
        try:
            response = requests.get(datum_url, params=params, verify=False)

            # If successful get the navd adjustment
            if response:
                results = response.json()
                # pprint.pprint(results)
                # Get adjustment in meters (NGVD29 to NAVD88)
                adjustment = results['t_z']
                # convert meters to feet
                #adjustment_ft = round(float(adjustment) * 3.28084, 2)
                adjustment_ft = float(adjustment) * 3.28084
            else:
                adjustment_ft = None

            # Update the DataFrame directly using the index
            if adjustment_ft is not None:
                combined_gdf.at[index, 'datum_offset_ft'] = adjustment_ft
                combined_gdf.at[index, 'adj_elev_ft'] = row['elev_ft'] + adjustment_ft

        except KeyError:
            combined_gdf.at[index, 'adj_elev_ft'] = 9999.0
    
    # Write the combined GeoDataFrame to a new GeoPackage file
    merged_filename = os.path.join(output_directory, 'all_events.gpkg')
    combined_gdf.to_file(merged_filename, layer='combined_data', driver='GPKG')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Functionality to reformat downloaded USGS HWM CSV data into benchmark data for FIM evaluation.')
    parser.add_argument('-c', '--usgs-hwm-csv', help='The downloaded USGS HWM data.', type=str, required=True)
    parser.add_argument('-o', '--output-directory', help='The directory where geopackages for individual events will be created.',
        type=str, required=True
    )
    
    args = vars(parser.parse_args())
    subset_hwms_by_event(**args)
