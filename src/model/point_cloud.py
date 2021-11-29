from __future__ import annotations

from typing import Tuple, Dict

import numpy as np
import open3d as o3d
from plyfile import PlyData
from seaborn.distributions import kdeplot
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA


class VoxelModel:
    def __init__(self, shape: np.array, cell_size: np.array, origin: np.array, voxels = None):
        self.shape = shape
        self.cell_size = cell_size
        self.origin = origin

        if voxels is None:
            self.voxels: Dict[Tuple[int, int, int], float] = {}
        else:
            self.voxels = voxels

    def __str__(self):
        return str(self.voxels)

    def __getitem__(self, key: Tuple[int, int, int]) -> float:
        return self.voxels[key]
    
    def __setitem__(self, key: Tuple[int, int, int], value: float) -> None:
        self.voxels[key] = value

    def add_voxel(self, cell: Tuple[int, int, int], value: float) -> None:
        self.voxels[cell] = value

    def remove_voxel(self, cell: Tuple[int, int, int]) -> None:
        self.voxels.pop(cell)

    def is_occupied(self, cell: Tuple[int, int, int]) -> bool:
        return cell in self.voxels

    def convolve(self, kernel) -> VoxelModel:
        pass

    def to_pcd(self) -> PointCloud:
        points = []
        for voxel in self.voxels.keys():
            points.append(self.origin + (self.cell_size * voxel))

        return PointCloud(np.array(points), source = self)



class PointCloud:
    leaf_size = 20  # Number of leaf nodes in KD-Tree spatial index

    def __init__(self, points: np.array, colors: np.array = None, source: object = None):
        self.points = points
        self.colors = colors   
        self.source = source

        # Point cloud shape and bounds
        self.size = np.shape(points)[0]
        self.aabb = np.array([
            [np.min(points[:, 0]), np.max(points[:, 0])],
            [np.min(points[:, 1]), np.max(points[:, 1])],
            [np.min(points[:, 2]), np.max(points[:, 2])]
        ])

        # Used for fast nearest neighbour operations
        self.kdt = KDTree(self.points, PointCloud.leaf_size)

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

    def k_nearest_neighbour(self, p: np.array, k: int) -> Tuple[np.array, np.array]:
        '''Get the first k neighbours that are closest to a given point.'''

        return self.kdt.query(p.reshape(1,-1), k)

    def ball_search(self, p, r):
        '''Get all points within r radius of point p.'''

        return self.kdt.query_radius(p.reshape(1,-1), r)
        
    def estimate_normals(self, k) -> np.array:
        ''' Estimate normalized point cloud normals using principal component analysis. 
            Component with smallest magnitude gives the normal vector.'''

        pca = PCA()

        normals = []
        for p in self.points:
            knn = self.k_nearest_neighbour(p, k)
            nbs = self.points[knn[1]].squeeze()

            pca.fit(nbs)
            normals.append(pca.components_[2] / np.linalg.norm(pca.components_[2]))
        
        return np.array(normals)

    def filter(self, property, func):
        '''Functional programming filter for point clouds'''

        out_points = []

        for i, p in enumerate(self.points):
            if func(property[i]):
                out_points.append(p)

        return PointCloud(np.array(out_points), source=self)

    def voxel_filter(self, cell_size) -> PointCloud:
        '''Divides point cloud into grid and returns new point cloud with only the centers of occupied cells.'''

        edge_sizes = self.aabb[:, 1] - self.aabb[:, 0]
        cell_sizes = edge_sizes / (edge_sizes // cell_size)

        occupied_cells, points = set(), []
        for p in self.points:
            # Find voxel cell that the given point is in
            cell = (p - self.aabb[:, 0]) // cell_sizes

            # Check if cell already contains point, otherwise add one
            if cell.tobytes() not in occupied_cells:
                occupied_cells.add(cell.tobytes())
                points.append(self.aabb[:, 0] + (cell + 0.5) * cell_sizes)

        return PointCloud(np.array(points), colors=None, source=self)

    def voxelize(self, cell_size) -> VoxelModel:
        '''Convert point cloud to discretized voxel representation.'''

        edge_sizes = self.aabb[:, 1] - self.aabb[:, 0]
        shape = edge_sizes // cell_size
        cell_sizes = edge_sizes / shape

        voxel_model = VoxelModel(shape, cell_sizes, self.aabb[:, 0])

        for p in self.points:
            # Find voxel cell that the given point is in
            cell = tuple(((p - self.aabb[:, 0]) // cell_sizes).astype(int))

            if voxel_model.is_occupied(cell):
                voxel_model[cell] += 1
            else:
                voxel_model[cell] = 1
                       
        return voxel_model

    @staticmethod
    def read_ply(fn: str) -> PointCloud:
        '''Reads .ply file to point cloud. Discards all mesh data.'''

        with open(fn, 'rb') as f:
            plydata = PlyData.read(f)

        num_points = plydata['vertex'].count

        points = np.zeros(shape=[num_points, 3], dtype=np.float32)
        points[:, 0] = plydata['vertex'].data['x']
        points[:, 1] = plydata['vertex'].data['y']
        points[:, 2] = plydata['vertex'].data['z']

        colors = np.zeros(shape=[num_points, 3], dtype=np.float32)
        colors[:, 0] = plydata['vertex'].data['red']
        colors[:, 1] = plydata['vertex'].data['green']
        colors[:, 2] = plydata['vertex'].data['blue']

        return PointCloud(points, colors/255, source=fn)
