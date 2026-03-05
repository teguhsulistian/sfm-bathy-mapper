import laspy
import numpy as np

file_path = "D:\PROGRAM\PROGRAM_MATLAB\GUI_SFM\DATA\LAS_KELAPA_DUA.las"

def load_las(file_path):
    las = laspy.read(file_path)
    pc = np.vstack([las.x, las.y, las.z]).T.astype("float32")
    return pc,las

pc,las = load_las(file_path)
print(f"Total points : {len(pc):,}")
print(f"X range      : {pc[:,0].min():.2f} → {pc[:,0].max():.2f}")
print(f"Y range      : {pc[:,1].min():.2f} → {pc[:,1].max():.2f}")
print(f"Z range      : {pc[:,2].min():.2f} → {pc[:,2].max():.2f}")

crs = las.header.parse_crs()
if crs is not None:
    print(f"CRS Name     : {crs.name}")
    print(f"EPSG Code    : {crs.to_epsg()}")
else:
    print("⚠️ No CRS found in file!")