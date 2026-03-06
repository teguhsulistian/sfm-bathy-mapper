import laspy
import pandas as pd
import numpy as np
from pyproj import Transformer
from eo_tides.model import model_tides


def load_las(file_path):
    #input las file, return point cloud as numpy array and las object
    las = laspy.read(file_path)
    pc = np.vstack([las.x, las.y, las.z]).T.astype("float32")
    
    #Find median of x and y for tide calculation
    x_med = np.median(pc[:,0])
    y_med = np.median(pc[:,1])

    #Get EPSG code from LAS file and transform median x,y to lat,lon
    epsg_src = las.header.parse_crs().to_epsg()
    transformer = Transformer.from_crs(epsg_src, 4326, always_xy=True)
    lon_med, lat_med = transformer.transform(x_med, y_med)
    print(f"CRS EPSG:{epsg_src} → EPSG:4326" f" | Median Lon: {lon_med:.6f}, Lat: {lat_med:.6f}")
  
    return pc,las, lon_med, lat_med

def tide_calc(tide_dir, model, x, y, start_time, end_time, freq):
    tide_results = model_tides(
        x=x,
        y=y,
        time= pd.date_range( start_time, end_time, freq=freq),
        model=model,
        directory=tide_dir
    )
    return tide_results