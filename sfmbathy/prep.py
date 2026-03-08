import laspy
import rasterio
import pandas as pd
import numpy as np
from pyproj import Transformer
from eo_tides.model import model_tides
from rasterio.transform import rowcol

## (1) Load LAS file and extract point cloud data, median coordinates, and CRS information ##

def load_las(file_path):
    """
    Load the LAS file and extract point cloud data, median coordinates, and CRS information.
    Parameters:
    file_path (str): The file path to the input LAS file.
    
    Returns:
    tuple: A tuple containing the point cloud as a numpy array, the LAS object, and the median longitude and latitude.  
    """
    #input las file, return point cloud as numpy array and las object
    las = laspy.read(file_path)
    xyz = np.asarray([las.x, las.y, las.z], dtype=np.float32).T
    rgb = np.asarray([las.red, las.green, las.blue], dtype=np.uint8).T
    pc = np.hstack((xyz, rgb))
    
    #Find median of x and y for tide calculation
    x_med = np.median(pc[:,0])
    y_med = np.median(pc[:,1])

    #Get EPSG code from LAS file and transform median x,y to lat,lon
    epsg_src = las.header.parse_crs().to_epsg()
    transformer = Transformer.from_crs(epsg_src, 4326, always_xy=True)
    lon_med, lat_med = transformer.transform(x_med, y_med)
    print(f"CRS EPSG:{epsg_src} → EPSG:4326" f" | Median Lon: {lon_med:.6f}, Lat: {lat_med:.6f}")
  
    return pc,las, lon_med, lat_med


## (2) Tide calculation to define water level##
def tide_calc(tide_dir, model, x, y, start_time, end_time, freq):
    """
    Calculate the representative water level (WL) for the location and time range of the point cloud data.
    This function uses the eo-tides library to model tides and rasterio to read the MSL to Ellipsoid height from a GeoTIFF file.    
    Parameters:
    tide_dir (str): Directory path where the tide model data and DATUM GeoTIFF are stored.
    model (str): The name of the tide model to use (e.g., "GOT", "FES", "TPXO").
    x (float): Longitude of the location.     
    y (float): Latitude of the location.
    start_time (str): Start time for tide calculation in ISO format UTC Time (e.g., "2024-01-01T00:00:00Z").
    end_time (str): End time for tide calculation in ISO format UTC Time (e.g., "2024-01-01T00:00:00Z").
    freq (str): Frequency of the time range for tide calculation (e.g., "1H" for hourly).
    
    eturns:
    float: The representative water level (WL) in meters relative to the ellipsoid reference.

    """
    #Calculate tides using the specified model and parameters, return tide results as a DataFrame
    tide_results = model_tides(
        x=x,
        y=y,
        time= pd.date_range( start_time, end_time, freq=freq),
        model=model,
        directory=tide_dir
    )

    # Calculate mean tide height from the results
    tide_mean = np.mean(tide_results['tide_height'])

    
    # Identify Ellipsoid height to MSL datum from the provided DATUM GeoTIFF file
    tif_path = tide_dir + "/DATUM/IS_MSL_ELLIPSOID.tif"
    with rasterio.open(tif_path) as src:
        # Convert lat/lon to row/col pixel coordinates
        row, col = rowcol(src.transform, x, y)
        
        # Read the value at that pixel
        msl_ellips = src.read(1)[row, col]

    WL = msl_ellips + tide_mean

    print(f"Location Lon: {x:.6f}, Lat: {y:.6f} | time range: {start_time} to {end_time}")
    print(f"Representative Water Level: {tide_mean:.3f} m , refer to MSL")    
    print(f"Representative Water Level: {WL:.3f} m , refer to Ellipsoid Reference")

    return WL