from __future__ import annotations

from os.path import exists
from random import random

import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree

from model.voxel_grid import *
from sklearn.decomposition import PCA


def vector_element_product(vector):
    return reduce(lambda x, y: x * y, vector)


def column_extrema(matrix: np.array):
    column_range = range(matrix.shape[1])

    minimum = np.array([min(matrix[:, i]) for i in column_range])
    maximum = np.array([max(matrix[:, i]) for i in column_range])

    return minimum, maximum


def minkowski_distance(p_a, p_b, p):
    return sum(np.power(p_a - p_b, p)) ** (1/p)


def batch_read_xyz(xyz_dir: str) -> List[PointCloud]:
    xyz_files = os.listdir(xyz_dir)
    point_clouds = [PointCloud.from_xyz(xyz_dir+file) for file in xyz_files]

    return point_clouds


@dataclass
class Plane:
    point: float
    normal: np.array

    def distance(self, p):
        '''https://mathworld.wolfram.com/Point-PlaneDistance.html'''

        return np.linalg.norm(self.normal * self.point + p)


class PointCloud:
    leaf_size: float = 16

    def __init__(self, points: np.array = None, colors: np.array = None, attributes: Dict[str, np.array] = None):
        if points is None or len(points) == 0:
            self.points = np.empty((0, 3))
            self.size = 0
            self.aabb = None
            self.kdt = None
        else:
            self.points = points

            # Point cloud shape and bounds
            self.size = np.shape(points)[0]
            self.aabb = np.array([
                [np.min(points[:, 0]), np.max(points[:, 0])],
                [np.min(points[:, 1]), np.max(points[:, 1])],
                [np.min(points[:, 2]), np.max(points[:, 2])]
            ])

            # Used for fast nearest neighbour operations
            self.kdt = KDTree(self.points, PointCloud.leaf_size)

        if colors is None or len(colors) == 0:
            self.colors = np.empty((0, 3))
        else:
            self.colors = colors

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

    def transform(self, transformation: np.array) -> PointCloud:
        if transformation.shape != (4, 4):
            raise ValueError("Transformation must be 4x4 matrix")

        # Add column of 1s to allow for multiplication with 4x4 transformation matrix
        pts = np.hstack((self.points, np.ones((self.points.shape[0], 1))))

        # Multiply every point with the transformation matrix
        # Remove added column of 1s
        pts_t = np.array([transformation.reshape(4, 4).dot(pt) for pt in pts])
        pts_t = pts_t[:, :3]

        return PointCloud(pts_t, self.colors, self.attributes)

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

        # TODO: CLEAN THIS UP, ADD COMMENTS
        merged_attributes = {}
        for attr in list(self.attributes.keys()) + list(other.attributes.keys()):
            if attr in self.attributes and attr in other.attributes:
                attr_tuple = self.attributes[attr], other.attributes[attr]
            elif attr in self.attributes and attr not in other.attributes:
                attr_tuple = self.attributes[attr], np.empty(
                    (other.size, self.attributes[attr].shape[1]))
            else:
                attr_tuple = np.empty(
                    (self.size, other.attributes[attr].shape[1])), other.attributes[attr]

            merged_attributes[attr] = np.concatenate(attr_tuple, axis=0)

        merged_point_cloud = PointCloud(
            merged_points, merged_colors, merged_attributes)
        return merged_point_cloud

    def voxelize(self, cell_size: float) -> VoxelGrid:
        from model.voxel_grid import VoxelGrid

        aabb_min = self.aabb[:, 0]

        result_voxels = ((self.points - aabb_min) // cell_size).astype(int)

        voxels = {tuple(cell): {attr: self.attributes[attr][index] for attr in self.attributes}
                  if len(self.attributes) else {}
                  for index, cell in enumerate(result_voxels)}

        voxel_model = VoxelGrid(cell_size, aabb_min, voxels)

        return voxel_model

    def radius_search(self, p, r) -> List[int]:
        '''Get all points within r radius of point p.'''

        return self.kdt.query_radius(p.reshape(1, -1), r)

    def region_grow(self, start: int, max_pt_dist: float, max_region_size: float = math.inf) -> List[int]:
        start_pt = self.points[start]
        unvisited, visited = set([start]), set()

        while len(unvisited):
            cur_idx = unvisited.pop()
            cur_pt = self.points[cur_idx]
            visited.add(cur_idx)

            cur_nbs_idx = self.radius_search(cur_pt, max_pt_dist)
            for nb_idx in cur_nbs_idx:
                nb_pt = self.points[nb_idx]

                if nb_idx not in visited and \
                        np.linalg.norm(nb_pt - start_pt) <= max_region_size:
                    unvisited.add(nb_idx)

        return list(visited)

    def random_reduce(self, keep_fraction: float) -> PointCloud:
        """Randomly remove a fraction of the point cloud's points.

        Args:
            keep_fraction (float): Fraction of points to keep. 

        Returns:
            PointCloud: Randomly reduced point cloud.
        """

        reduced_points = self.points[random() < keep_fraction].squeeze()

        return PointCloud(reduced_points)

    def add_noise(self, scale: float, center: float = 0) -> PointCloud:
        """Adds Gaussian noise to each point in the point cloud.

        Args:
            scale (float): Standard deviation of noise distribution.
            center (float, optional): Mean value of noise. Defaults to 0.

        Returns:
            PointCloud: Original point cloud with added Gaussian noise.
        """

        noise = np.array([np.random.normal(center, scale, self.size)]*3)
        noisy_points = self.points + noise.T

        return PointCloud(noisy_points)

    @staticmethod
    def read_ply(fn: str, y_up: bool = True) -> PointCloud:
        """Reads .ply file to point cloud. Discards all mesh data.

        Args:
            fn (str): Filename of .ply file.

        Raises:
            ArgumentError: Specified file does not exist.

        Returns:
            PointCloud: Point cloud geometry read from .ply file.
        """

        if not exists(fn):
            raise ValueError(f'File {fn} does not exist.')

        # Read ply file from disk
        with open(fn, 'rb') as f:
            plydata = PlyData.read(f)

        # Create an empty matrix which we will fill with points
        num_points = plydata['vertex'].count
        points = np.zeros(shape=[num_points, 3], dtype=np.float32)

        # Fill matrix with point coordinates
        points[:, 0] = plydata['vertex'].data['x']

        if y_up:
            points[:, 1] = plydata['vertex'].data['y']
            points[:, 2] = plydata['vertex'].data['z']
        else:
            points[:, 2] = plydata['vertex'].data['y']
            points[:, 1] = plydata['vertex'].data['z']

        # Try to read colors from ply file, if they don't exist then colors are set to zero
        try:
            colors = np.zeros(shape=[num_points, 3], dtype=np.float32)
            colors[:, 0] = plydata['vertex'].data['red']
            colors[:, 1] = plydata['vertex'].data['green']
            colors[:, 2] = plydata['vertex'].data['blue']
            colors = colors/255
        except ValueError:
            colors = np.zeros(shape=[num_points, 3], dtype=np.float32)

        rooms = plydata['vertex'].data['scalar_Original_cloud_index'].astype(
            np.int32)

        return PointCloud(points, colors, {'room': rooms})

    def read_xyz(fn: str, separator: str = ' ', y_up: bool = True) -> PointCloud:
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
            raise ValueError(f'File {fn} does not exist.')

        # Read file from disk and extract point cloud.
        with open(fn, 'r') as f:
            lines = f.readlines()

        # Separate each line using the provided separator symbol and cast each value to float
        # Put result in numpy matrix where each row represents a point.
        xyz = [[float(i) for i in line.split(separator)] for line in lines]
        pt_matrix = np.array(xyz)

        if not y_up:
            pt_matrix[:, [1, 2]] = pt_matrix[:, [2, 1]]

        # Split matrix into point positions and colors
        pt_color = pt_matrix[:, 3:] / 255
        pt_pos = pt_matrix[:, :3]

        # Create point cloud object from point matrix.
        pcd = PointCloud(pt_pos, pt_color)
        return pcd

    def to_tensor(self, bcn: bool = False, batch: bool = True):
        from torch import as_tensor

        if batch:
            if bcn:
                return as_tensor(np.swapaxes(self.points[:, :][np.newaxis, ...], 1, 2)).float()
            return as_tensor(self.points[:, :][np.newaxis, ...]).float()
        else:
            return as_tensor(self.points).float()

    def centroid(self):
        return np.mean(self.points, axis=0)

    def center(self):
        return PointCloud(self.points - self.centroid())

    def normalize(self, target_min, target_max):
        min, max = self.aabb[:, 0], self.aabb[:, 1]
        points = (self.points - min) / (max - min).reshape((1, 3))
        points = points * (target_max - target_min) + target_min

        return PointCloud(self.points)

    def filter(self, func):
        return PointCloud(np.array([p for p in self.points if func(p)]))

    def intersect(self, voxel_grid: VoxelGrid):
        return self.filter(lambda p: voxel_grid.contains_point(p))

    def svd(self):
        pca = PCA(n_components=3)
        components = pca.fit(self.points)

        return components.singular_values_

    def pca(self):
        pca = PCA(n_components=3)
        components = pca.fit(self.points)

        return components.components_

    def explained_variance_ratio(self):
        pca = PCA(n_components=3)
        components = pca.fit(self.points)

        return components.explained_variance_ratio_

    def fit_plane(self):
        nbs_eig = self.pca()
        nbs_normal = nbs_eig[2]
        best_fit_plane = Plane(self.centroid(), nbs_normal)

        return best_fit_plane

    def mean_dist_to_point(self, p_other):
        sum_dist = 0
        for p_self in self.points:
            sum_dist += minkowski_distance(p_self, p_other, 2)
        return sum_dist / len(self.points)

    def planarity(self):
        best_fit_plane = self.fit_plane()

        sum_plane_distance = 0
        for p in self.points:
            plane_dist = best_fit_plane.distance(p)
            sum_plane_distance += abs(plane_dist)

        return sum_plane_distance / len(self.points)

    def roughness(self, k):
        sum_plane_distance = 0
        for p in self.points:
            nbs_i = self.sindex.query(p, k)[1]

            if len(nbs_i) >= 3:
                nbs = self.points[nbs_i]
                nbs_pcd = PointCloud(nbs + p)
                best_fit_plane = nbs_pcd.fit_plane()
                plane_dist = best_fit_plane.distance(p)
                sum_plane_distance += abs(plane_dist)

        return sum_plane_distance / len(self.points)

    def centroid(self):
        return np.mean(self.points, axis=0)

    def quotient_of_eigenvalues(self):
        variance = self.explained_variance_ratio()
        return variance[0] / variance[1] / variance[2]

    def volume(self) -> float:
        min_corner, max_corner = column_extrema(self.points)
        axis_lengths = max_corner - min_corner
        volume = vector_element_product(axis_lengths)

        return volume

    def height(self) -> float:
        return max(self.points[:, 2]) - min(self.points[:, 2])

    def projected_area(self) -> float:
        projected_points = self.points[:, :2]
        min_corner, max_corner = column_extrema(projected_points)
        horizontal_axis_lengths = max_corner - min_corner
        area = vector_element_product(horizontal_axis_lengths)

        return area

    def mean_distance_to_centroid(self):
        return self.mean_dist_to_point(self.centroid())


class Trajectory(PointCloud):
    pass
