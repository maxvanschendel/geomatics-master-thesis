from __future__ import annotations

import numpy as np
from model.topometric_map import *
from model.point_cloud import *
from model.voxel_grid import *
from processing.parameters import PreProcessingParameters


def pre_process(partial_map: PointCloud, params: PreProcessingParameters) -> VoxelGrid:
    partial_map_reduced = partial_map.random_reduce(params.reduce)
    partial_map_scaled = partial_map_reduced.scale(np.array(params.scale))
    partial_map_rot = partial_map_scaled.rotate(random()*360, [0,1,0])
    
    return partial_map_rot
