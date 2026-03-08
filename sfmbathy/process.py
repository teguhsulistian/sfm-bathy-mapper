import laspy
import numpy as np

def process_pc(pc, WL, n_water):
    """
    Process the point cloud to correct for refraction based on the water level and refractive index.

    Parameters:
    pc (numpy.ndarray): The input point cloud as a Nx6 array (x, y, z, red, green, blue).
    WL (float): The water level (tide height) at the time of data capture.
    n_water (float): The refractive index of water.

    Returns:
    numpy.ndarray: The corrected point cloud.
    """
    # Defining the refractive index of water, default is 1.33 for visible light in water
    if n_water == "default":
        n_water = 1.33
    else:
        n_water = float(n_water)

    # Processing the point cloud to correct for refraction (small angel approach)
    pc_filtered = pc[pc[:,2] < WL]
    pc_filtered[:,2] = ((pc_filtered[:,2]-WL) * n_water) + WL

    pc_land = pc[pc[:,2] >= WL]

    pc_corrected = np.vstack((pc_filtered, pc_land))

    print(f"Number of points below water level: {len(pc_filtered)}, percentage: {len(pc_filtered)/len(pc)*100:.2f}%")
    print(f"Number of points above water level: {len(pc_land)}, percentage: {len(pc_land)/len(pc)*100:.2f}%")   
    print(f"Original point cloud size: {len(pc)}, Corrected point cloud size: {len(pc_corrected)}")
    
    return pc_corrected

def export_pc(pc_corrected, las, output_path):
    """
    Save the corrected point cloud to a new LAS file.

    Parameters:
    pc_corrected (numpy.ndarray): The corrected point cloud as a Nx6 array (x, y, z, red, green, blue).
    las (laspy.LasData): The original LAS data object to copy header information from.
    output_path (str): The file path to save the corrected LAS file.
    """
    # Create a new LAS object with the same header as the original
    las_corrected = laspy.LasData(las.header)

    # Update the point data with the corrected point cloud
    las_corrected.x = pc_corrected[:, 0]
    las_corrected.y = pc_corrected[:, 1]
    las_corrected.z = pc_corrected[:, 2]
    las_corrected.red = pc_corrected[:, 3].astype(np.uint16)
    las_corrected.green = pc_corrected[:, 4].astype(np.uint16)
    las_corrected.blue = pc_corrected[:, 5].astype(np.uint16)

    # Save the corrected LAS file
    las_corrected.write(output_path)
    print(f"Corrected LAS file saved to: {output_path}")