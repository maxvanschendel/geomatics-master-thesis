from __future__ import annotations

import numpy as np
from model.topometric_map import *
from model.point_cloud import *
from model.voxel_grid import *
from processing.configuration import PreProcessingParameters
from random import random


def pre_process(partial_map: PointCloud, params: PreProcessingParameters) -> VoxelGrid:
    # Add Gaussian noise to input point cloud
    partial_map_noise = partial_map.add_noise(params.noise_scale)

    # Random reduce points
    partial_map_reduced = partial_map_noise.random_reduce(params.reduce)

    # Apply a scaling factor to point cloud
    partial_map_scaled = partial_map_reduced.scale(np.array(params.scale))

    # Randomly rotate input point cloud to remove any prior alignment
    partial_map_rot = partial_map_scaled.rotate(random()*360, [0,1,0])
    
    return partial_map_rot
