from __future__ import annotationsC

import numpy as np
from model.topometric_map import *
from model.point_cloud import *
from model.voxel_grid import *
from utils.array import flatten_list
from processing.parameters import PreProcessingParameters
from random import random, randint


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

def degrade_point_cloud(pcd: PointCloud, n_holes: int, max_pt_dist: float, size_range: Tuple[float, float]) -> PointCloud:
    # Random initial hole points
    holes: List[int] = [randint(0, len(pcd.points)) for _ in range(n_holes)]
    
    # Grow hole from initial hole points with random maximum size within given size_range
    grown_holes: List[List[int]] = [pcd.region_grow(start=hole, 
                                                    max_pt_dist=max_pt_dist, 
                                                    max_region_size=size_range[0]+random()*(size_range[1]-size_range[0])) 
                                    for hole in holes]
    
    points_in_holes: List[int] = flatten_list(grown_holes)
    unique_points_in_holes = set(points_in_holes)
    
    return pcd.subset(list(unique_points_in_holes))
    
