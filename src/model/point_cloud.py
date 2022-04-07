from __future__ import annotations

from random import random

import numpy as np
import open3d as o3d
from plyfile import PlyData
from sklearn.decomposition import PCA


class PointCloud:
    def __init__(self, points: np.array, colors: np.array = None):
        self.points = points
        self.colors = colors
        
        # Point cloud shape and bounds
        self.size = np.shape(points)[0]
        self.aabb = np.array([
            [np.min(points[:, 0]), np.max(points[:, 0])],
            [np.min(points[:, 1]), np.max(points[:, 1])],
            [np.min(points[:, 2]), np.max(points[:, 2])]
        ])

    def __str__(self) -> str:
        '''Get point cloud in human-readable format.'''

        return f"Point cloud with {self.size} points\n"

    def to_o3d(self) -> o3d.geometry.PointCloud:
        '''Creates Open3D point cloud for visualisation purposes.'''

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        if self.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.colors)
        return pcd
    
    def rotate(self, angle: float, axis: np.array) -> PointCloud:
        from scipy.spatial.transform import Rotation as R
        
        r = R.from_rotvec(angle * np.array(axis), degrees=True)
        rotated = r.apply(self.points)
        
        return PointCloud(rotated)

    def scale(self, scale: np.array):
        return PointCloud(self.points*scale)
    
    def pca(self):
        nbs = self.points.squeeze()
        pca = PCA()
        pca.fit(nbs)
        
        return pca.components_           

    def voxelize(self, cell_size: float) -> VoxelGrid:
        '''Convert point cloud to discretized voxel representation.'''
        
        from model.voxel_grid import VoxelGrid

        aabb_min, aabb_max = self.aabb[:, 0], self.aabb[:, 1]
        result_voxels = ((self.points - aabb_min) // cell_size).astype(int)
        
        voxels = {tuple(cell): {} for cell in result_voxels}
        voxel_model = VoxelGrid(cell_size, aabb_min, voxels)
        
        return voxel_model

    def random_reduce(self, keep_fraction: float) -> PointCloud:
        keep_points = [p for p in self.points if random() < keep_fraction]

        return PointCloud(np.array(keep_points))

    @ staticmethod
    def read_ply(fn: str) -> PointCloud:
        '''Reads .ply file to point cloud. Discards all mesh data.'''

        with open(fn, 'rb') as f:
            plydata = PlyData.read(f)

        num_points = plydata['vertex'].count

        points = np.zeros(shape=[num_points, 3], dtype=np.float32)
        points[:, 0] = plydata['vertex'].data['x']
        points[:, 1] = plydata['vertex'].data['y']
        points[:, 2] = plydata['vertex'].data['z']

        try:
            colors = np.zeros(shape=[num_points, 3], dtype=np.float32)
            colors[:, 0] = plydata['vertex'].data['red']
            colors[:, 1] = plydata['vertex'].data['green']
            colors[:, 2] = plydata['vertex'].data['blue']
        except ValueError:
            colors = np.zeros(shape=[num_points, 3], dtype=np.float32)

        return PointCloud(points, colors/255)
