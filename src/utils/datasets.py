from typing import  List
import numpy as np

from model.point_cloud import Trajectory
from model.voxel_grid import VoxelGrid
from utils.io import *
from utils.linalg import random_transform

def simulate_partial_maps(pcd, trajectories, vis_range, voxel_size):
    # Apply a random transformation to the ground truth map for each voxel grid
    transforms = [random_transform() for _ in trajectories]
    
    transformed_ground_truth = [pcd.transform(t) for t in transforms]
    transformed_trajectories = [trajectories[i].transform(t) for i, t in enumerate(transforms)]
    
    transformed_voxel_grids = [pcd.voxelize(voxel_size) for pcd in transformed_ground_truth]
    simulated_scans = [simulate_scan(transformed_voxel_grids[i], transformed_trajectories[i], vis_range) for i, _ in enumerate(transforms)]
    
    return simulated_scans, transforms

def simulate_scan(voxel_grid: VoxelGrid, trajectory: Trajectory, scan_range: float):
    visible_voxels = set()
    
    for p in trajectory.points:
        visibility = voxel_grid.visibility(p, scan_range)
        visible_voxels = visible_voxels.union(set(visibility.voxels.keys()))
        
    return voxel_grid.subset(lambda v: v in visible_voxels)

def read_trajectory(fns: List[str]) -> List[np.array]:
    return list(map(lambda fn: Trajectory.read_xyz(fn), fns))