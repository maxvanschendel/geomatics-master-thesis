from __future__ import annotations

from argparse import ArgumentError
from os.path import exists
from random import random

import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R

from model.voxel_grid import *


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

    def translate(self, translation: np.array) -> PointCloud:
        return PointCloud(self.points + translation)

    def rotate(self, angle: float, axis: np.array) -> PointCloud:
        r = R.from_rotvec(angle * np.array(axis), degrees=True)
        rotated = r.apply(self.points)

        return PointCloud(rotated)

    def scale(self, scale: np.array):
        return PointCloud(self.points*scale)

    def transform(self, transformation: np.array) -> PointCloud:
        # Add column of 1s to allow for multiplication with 4x4 transformation matrix
        pts = np.hstack((self.points, np.ones((self.points.shape[0], 1))))

        # Multiply every point with the transformation matrix
        # Remove added column of 1s
        pts_t = np.array([transformation.reshape(4, 4).dot(pt) for pt in pts])
        pts_t = pts_t[:, :3]
        
        return PointCloud(pts_t, self.colors)

    def eigenvectors(self) -> np.array:
        return np.linalg.eig(self.points)

    def voxelize(self, cell_size: float) -> VoxelGrid:
        aabb_min = self.aabb[:, 0]
        result_voxels = ((self.points - aabb_min) // cell_size).astype(int)

        voxels = {tuple(cell): {} for cell in result_voxels}
        voxel_model = VoxelGrid(cell_size, aabb_min, voxels)

        return voxel_model

    def random_reduce(self, keep_fraction: float) -> PointCloud:
        reduced_points = self.points[random() < keep_fraction]

        return PointCloud(reduced_points)

    @staticmethod
    def read_ply(fn: str) -> PointCloud:
        '''Reads .ply file to point cloud. Discards all mesh data.'''

        if not exists(fn):
            raise ArgumentError(fn, f'File {fn} does not exist.')

        # Read ply file from disk
        with open(fn, 'rb') as f:
            plydata = PlyData.read(f)

        # Create an empty matrix which we will fill with points
        num_points = plydata['vertex'].count
        points = np.zeros(shape=[num_points, 3], dtype=np.float32)

        # Fill matrix with point coordinates
        points[:, 0] = plydata['vertex'].data['x']
        points[:, 1] = plydata['vertex'].data['y']
        points[:, 2] = plydata['vertex'].data['z']

        # Try to read colors from ply file, if they don't exist then colors are set to zero
        try:
            colors = np.zeros(shape=[num_points, 3], dtype=np.float32)
            colors[:, 0] = plydata['vertex'].data['red']
            colors[:, 1] = plydata['vertex'].data['green']
            colors[:, 2] = plydata['vertex'].data['blue']
            colors = colors/255
        except ValueError:
            colors = np.zeros(shape=[num_points, 3], dtype=np.float32)

        return PointCloud(points, colors)
