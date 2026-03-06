import laspy
import numpy as np
from eo_tides.utils import list_models
from pyTMD.io.model import load_database

def load_las(file_path):
    las = laspy.read(file_path)
    pc = np.vstack([las.x, las.y, las.z]).T.astype("float32")
    return pc,las

