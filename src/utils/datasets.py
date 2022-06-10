from copy import deepcopy
from typing import  List
import numpy as np

from model.voxel_grid import VoxelGrid
from utils.io import *



def simulate_scan(voxel_grid: VoxelGrid, trajectory: List[np.array], scan_range: float):
    visible_voxels = set()
    
    for p in trajectory:
        visibility = voxel_grid.visibility(p, scan_range)
        visible_voxels = visible_voxels.union(set(visibility.voxels.keys()))
        
    return voxel_grid.subset(lambda v: v in visible_voxels)

