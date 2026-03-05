import laspy
import numpy as np

def load_las(file_path):
    las = laspy.read(file_path)
    pc = np.vstack([las.x, las.y, las.z]).T.astype("float32")
    return pc,las