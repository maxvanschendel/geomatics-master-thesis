import unittest
from parameterized import parameterized, parameterized_class


import numpy as np
from model.point_cloud import PointCloud
from model.voxel_grid import VoxelGrid
from model.sparse_voxel_octree import SVO
from utils.array import flatten_list
from utils.linalg import *
from utils.datasets import simulate_scan
from utils.visualization import visualize_visibility

@parameterized_class([
   { "ply_file": "../data/cslam/flat/flat.ply", "trajectory_file": "../data/cslam/flat/flat_trajectory_01.csv", "voxel_size": 0.2, "interior_point": np.array([-1.665, 0.1014, 4.934]), "visualize": False},
   { "ply_file": "../data/cslam/flat/flat.ply", "trajectory_file": "../data/cslam/flat/flat_trajectory_02.csv", "voxel_size": 0.2, "interior_point": np.array([-1.665, 0.1014, 4.934]), "visualize": False},
   { "ply_file": "../data/s3dis/area_3/area_3.ply", "trajectory_file": "../data/s3dis/area_3/area_3_trajectory_01.csv", "voxel_size": 0.1, "interior_point": np.array([5.68, -4.43, 3.016]), "visualize": True},
])
class VoxelGridTest(unittest.TestCase):
    @classmethod
    def load_grid(self):
        return PointCloud.read_ply(self.ply_file).voxelize(self.voxel_size)
    
    @classmethod
    def load_trajectory(self):
        return PointCloud.read_xyz(self.trajectory_file)

    def test_visibility(self):
        pass
        
    @parameterized.expand([
        [5],
        [1],
    ])
    def test_radius_search(self, radius):       
        grid = self.load_grid()

        radius_voxels = grid.radius_search(self.interior_point, radius)
        voxels_distance = [euclidean_distance(grid.voxel_centroid(v), self.interior_point) for v in radius_voxels]

        voxels_outside_radius = [d >= radius + SVO.max_centroid_in_radius_distance(self.voxel_size) for d in voxels_distance]
        n_outside_radius = sum(voxels_outside_radius)
        
        self.assertFalse(n_outside_radius, f"Found {n_outside_radius} voxels outside search radius")

    @parameterized.expand([
        [5],
        [1],
    ])
    def test_simulate_scan(self, radius):
        grid = self.load_grid()
        trajectory = self.load_trajectory()

        scan = simulate_scan(grid, trajectory, radius)
        scan_radius = set(flatten_list([grid.radius_search(p, radius) for p in trajectory.points]))

        self.assertTrue(type(scan) == VoxelGrid, "Scan output must be voxel grid.")
        self.assertTrue(scan.get_voxels().issubset(grid.get_voxels()), 
                        "All voxels in simulated scan must also be in voxel grid")
        self.assertTrue(scan.get_voxels().issubset(scan_radius), 
                        "Some voxels in simulated scan are outside trajectory radius")
        
        if self.visualize:
            visualize_visibility(scan, trajectory.points)