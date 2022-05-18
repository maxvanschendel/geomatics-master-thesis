from __future__ import annotations

from argparse import ArgumentError
from os import sep
from os.path import exists
from random import random

import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R

from model.voxel_grid import *


class PointCloud:
    def __init__(self, points: np.array = None, colors: np.array = None, attributes: Dict[str, np.array] = None):
        if points is None:
            self.points = np.empty((0, 3))
            self.size = 0
            self.aabb = None
        else:
            self.points = points

            # Point cloud shape and bounds
            self.size = np.shape(points)[0]
            self.aabb = np.array([
                [np.min(points[:, 0]), np.max(points[:, 0])],
                [np.min(points[:, 1]), np.max(points[:, 1])],
                [np.min(points[:, 2]), np.max(points[:, 2])]
            ])

        if colors is None:
            self.colors = np.empty((0,3))
        else:
            self.colors = colors

        self.colors = attributes if attributes else np.empty((0,3))
        self.attributes = attributes if attributes else {}

    def __str__(self) -> str:
        '''Get point cloud in human-readable format.'''

        return f"Point cloud with {self.size} points\n"

    def to_o3d(self) -> o3d.geometry.PointCloud:
        """Creates Open3D point cloud for visualisation purposes.

        Returns:
            o3d.geometry.PointCloud: Open3D point cloud.
        """

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        if self.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.colors)
        return pcd

    def translate(self, translation: np.array) -> PointCloud:
        return PointCloud(self.points + translation, self.colors, self.attributes)

    def rotate(self, angle: float, axis: np.array) -> PointCloud:
        r = R.from_rotvec(angle * np.array(axis), degrees=True)
        rotated = r.apply(self.points)

        return PointCloud(rotated, self.colors, self.attributes)

    def scale(self, scale: np.array):
        return PointCloud(self.points*scale, self.colors, self.attributes)

    def transform(self, transformation: np.array) -> PointCloud:
        if transformation.shape() != (4, 4):
            # TODO: IMPLEMENT
            pass

        # Add column of 1s to allow for multiplication with 4x4 transformation matrix
        pts = np.hstack((self.points, np.ones((self.points.shape[0], 1))))

        # Multiply every point with the transformation matrix
        # Remove added column of 1s
        pts_t = np.array([transformation.reshape(4, 4).dot(pt) for pt in pts])
        pts_t = pts_t[:, :3]

        return PointCloud(pts_t, self.colors)

    def merge(self, other: PointCloud) -> PointCloud:
        """Takes two point clouds and creates a new one containing the points of both.
        Args:
            other (PointCloud): The point cloud to merge this point cloud with.

        Returns:
            PointCloud: Resultant merged point cloud containing both input point clouds.
        """

        merged_points = np.concatenate((self.points, other.points), axis=0)

        # Vertically stack point arrays and initialize new point cloud
        if not len(self.colors) and len(other.colors):
            merged_colors = other.colors
        elif len(self.colors) and not len(other.colors):
            merged_colors = self.colors
        elif not len(other.colors) and not len(self.colors):
            merged_colors = None
        else:
            merged_colors = np.concatenate((self.colors, other.colors), axis=0)

        merged_attributes = {}
        for attr in list(self.attributes.keys()) + list(other.attributes.keys()):
            if attr in self.attributes and attr in other.attributes:
                attr_tuple = self.attributes[attr], other.attributes[attr]
            elif attr in self.attributes and attr not in other.attributes:
                attr_tuple = self.attributes[attr], np.empty((other.size, self.attributes[attr].shape[1]))
            else:
                attr_tuple = np.empty((self.size, other.attributes[attr].shape[1])), other.attributes[attr]
                
            merged_attributes[attr] = np.concatenate(attr_tuple, axis=0)

        merged_point_cloud = PointCloud(merged_points, merged_colors, merged_attributes)
        return merged_point_cloud

    def voxelize(self, cell_size: float) -> VoxelGrid:
        from model.voxel_grid import VoxelGrid

        aabb_min = self.aabb[:, 0]

        result_voxels = ((self.points - aabb_min) // cell_size).astype(int)

        voxels = {tuple(cell): {attr: self.attributes[attr][index][0] for attr in self.attributes} for index, cell in enumerate(result_voxels)}
        voxel_model = VoxelGrid(cell_size, aabb_min, voxels)

        return voxel_model

    def random_reduce(self, keep_fraction: float) -> PointCloud:
        """Randomly remove a fraction of the point cloud's points.

        Args:
            keep_fraction (float): Fraction of points to keep. 

        Returns:
            PointCloud: Randomly reduced point cloud.
        """

        reduced_points = self.points[random() < keep_fraction]
        return PointCloud(reduced_points)

    def add_noise(self, scale: float, center: float = 0) -> PointCloud:
        """Adds Gaussian noise to each point in the point cloud.

        Args:
            scale (float): Standard deviation of noise distribution.
            center (float, optional): Mean value of noise. Defaults to 0.

        Returns:
            PointCloud: Original point cloud with added Gaussian noise.
        """

        noise = np.random.normal(center, scale, self.size())
        noisy_points = self.points + noise

        return PointCloud(noisy_points)

    @staticmethod
    def read_ply(fn: str) -> PointCloud:
        """Reads .ply file to point cloud. Discards all mesh data.

        Args:
            fn (str): Filename of .ply file.

        Raises:
            ArgumentError: Specified file does not exist.

        Returns:
            PointCloud: Point cloud geometry read from .ply file.
        """

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

    def read_xyz(fn: str, separator: str) -> PointCloud:
        """Read XYZ file from disk.

        Args:
            fn (str): Filename of XYZ file, extension is not necessarily .xyz.
            separator (str): Symbol separating values on each line of the XYZ file.

        Raises:
            ArgumentError: Specified file does not exist.

        Returns:
            PointCloud: Point cloud geometry read from XYZ file.
        """

        if not exists(fn):
            raise ArgumentError(fn, f'File {fn} does not exist.')

        # Read file from disk and extract point cloud.
        with open(fn, 'r') as f:
            lines = f.readlines()

        # Separate each line using the provided separator symbol and cast each value to float
        # Put result in numpy matrix where each row represents a point.
        xyz = [[float(i) for i in line.split(separator)] for line in lines]
        pt_matrix = np.array(xyz)

        # Split matrix into point positions and colors
        pt_color = pt_matrix[:, 3:] / 255
        pt_pos = pt_matrix[:, :3]

        # Create point cloud object from point matrix.
        pcd = PointCloud(pt_pos, pt_color)
        return pcd
