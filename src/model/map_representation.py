from __future__ import annotations
import enum

from typing import Tuple, Dict

import numpy as np
import open3d as o3d
from plyfile import PlyData
from seaborn.distributions import kdeplot
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from itertools import product
import copy
import networkx


class SpatialGraphRepresentation:
    def __init__(self,
                 scale: np.array = np.array([1, 1, 1]),
                 origin: np.array = np.array([0, 0, 0]),
                 graph: networkx.Graph = None):

        self.scale = scale
        self.origin = origin

        if graph is None:
            self.graph = networkx.Graph()
        else:
            self.graph = graph

        for i, node in enumerate(self.graph.nodes()):
            self.graph.nodes[node]['index'] = i

    def node_index(self, node):
        return self.graph.nodes[node]['index']

    def to_o3d(self) -> o3d.geometry.LineSet:
        '''Creates Open3D point cloud for visualisation purposes.'''

        points = self.graph.nodes()
        lines = [(self.node_index(n[0]), self.node_index(n[1]))
                 for n in self.graph.edges()]
        color = [self.graph.nodes[p]['color'] for p in self.graph.nodes]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(color)

        return line_set

    def to_voxel(self):
        vox = VoxelRepresentation(np.array([0, 0, 0]), self.scale, self.origin, {
                                  node: {'population': 1} for node in self.graph.nodes()})
        vox.shape = vox.extents()

        for node in self.graph.nodes:
            vox[node]['color'] = self.graph.nodes[node]['color']

        return vox

    def connected_components(self):
        return sorted(networkx.connected_components(self.graph), key=len, reverse=True)


class VoxelRepresentation:
    leaf_size = 20

    def __init__(self,
                 shape: np.array = None,
                 cell_size: np.array = np.array([1, 1, 1]),
                 origin: np.array = np.array([0, 0, 0]),
                 voxels=None):

        self.shape = shape
        self.cell_size = cell_size
        self.origin = origin

        if voxels is None:
            self.voxels: Dict[Tuple[int, int, int], Dict[str, object]] = {}
        else:
            self.voxels = voxels

    def __str__(self):
        return str(self.voxels)

    def __getitem__(self, key: Tuple[int, int, int]) -> Dict[str, object]:
        return self.voxels[key]

    def __setitem__(self, key: Tuple[int, int, int], value: Dict[str, object]) -> None:
        self.voxels[key] = value

    def __add__(self, other: VoxelRepresentation):
        new_model = self.clone()
        for v in other.voxels:
            if (new_model.is_occupied(v)):
                new_model.voxels[v] += other.voxels[v]
            else:
                new_model.voxels[v] = other.voxels[v]

        # Adapt size of voxel model to new extents after addition
        new_model.shape = new_model.extents()
        return new_model

    def __sub__(self, other):
        new_model = self.clone()

        for v in other.voxels:
            if (new_model.is_occupied(v)):
                if new_model[v] < other.voxels[v]:
                    new_model.remove_voxel(v)
                else:
                    new_model[v] -= other.voxels[v]

        return new_model

    def set_attribute(self, vals, attr):
        for i, val in enum(vals):
            self.voxels.keys()[i][attr] = val

    def map(self, func, *args):
        return map(lambda v: func(v, *args), self.voxels)

    @staticmethod
    def pca(vals):
        try:
            pca = PCA()
            pca.fit(vals)
        except Exception as e:
            print(e, vals)

        return pca
        
    def estimate_normals(self, kernel: VoxelRepresentation):
        nbs = self.map(self.get_kernel, kernel)
        pca_components = map(lambda nb: VoxelRepresentation.pca(nb).components_, nbs)
        normals = map(lambda c: c[2] / np.linalg.norm(c[2]), pca_components)

        return list(normals)
            
    def extents(self):
        new_voxels = np.array(list(self.voxels.keys()))
        new_shape = list(self.shape)

        for v in new_voxels:
            for i, c in enumerate(self.shape):
                new_shape[i] = max([v[i].max(), c])
        return new_shape

    def clone(self):
        return VoxelRepresentation(copy.deepcopy(self.shape),
                                   copy.deepcopy(self.cell_size),
                                   copy.deepcopy(self.origin),
                                   copy.deepcopy(self.voxels))

    def add_voxel(self, cell: Tuple[int, int, int], value: Dict[str, object] = {}) -> None:
        self.voxels[cell] = value

    def remove_voxel(self, cell: Tuple[int, int, int]) -> None:
        self.voxels.pop(cell)

    def is_occupied(self, cell: Tuple[int, int, int]) -> bool:
        return cell in self.voxels

    def get_kernel(self, cell: Tuple[int, int, int], kernel: VoxelRepresentation):
        kernel_cells = []
        for k in kernel.voxels:
            nb = tuple(cell + (k - kernel.origin))
            if self.is_occupied(nb):
                kernel_cells.append(nb)

        return kernel_cells

    def kernel_contains_neighbours(self, cell: Tuple[int, int, int], kernel: VoxelRepresentation) -> bool:
        for k in kernel.voxels:
            nb = tuple(cell + (k - kernel.origin))
            if self.is_occupied(nb):
                return True

        return False

    def to_o3d(self, has_color=False):
        return self.to_pcd(has_color).to_o3d()

    def to_pcd(self, has_color=False) -> PointCloudRepresentation:
        points = []
        color = []
        for voxel in self.voxels.keys():
            points.append(self.origin + (self.cell_size * voxel))

            if has_color:
                color.append(self.voxels[voxel]['color'])

        return PointCloudRepresentation(np.array(points), colors=color, source=self)

    def to_graph(self, nb26=False) -> SpatialGraphRepresentation:
        if nb26:
            kernel = VoxelRepresentation.nb26()
        else:
            kernel = VoxelRepresentation.nb6()

        graph = networkx.Graph()

        for v in self.voxels:
            nbs = self.get_kernel(v, kernel)
            for nb in nbs:
                graph.add_edge(*(v, nb))

        return SpatialGraphRepresentation(self.cell_size, self.origin, graph)

    ### Basic mathematical morphology operations ###
    def erode(self, kernel) -> VoxelRepresentation:
        '''Erode model with kernel.'''

        pass

    def dilate(self, kernel) -> VoxelRepresentation:
        '''Dilate model with kernel'''

        dilated_model = VoxelRepresentation(
            self.shape, self.cell_size, self.origin, {})

        for v in self.voxels:
            dilated_model.add_voxel(v, 1)

            for k in kernel.voxels:
                nb = tuple(v + (k - kernel.origin))
                dilated_model.add_voxel(nb, 1)

        return dilated_model

    def translate(self, translation: np.array):
        translated_cells = {}
        new_shape = list(self.shape)

        for voxel in self.voxels:
            voxel_t = translation + voxel
            out_of_bounds_voxels = np.where(voxel_t - new_shape > -1)[0]

            if len(out_of_bounds_voxels) > 0:
                for v in out_of_bounds_voxels:
                    new_shape[int(v)] = voxel_t[int(v)]

            translated_cells[tuple(voxel_t)] = self.voxels[voxel]

        return VoxelRepresentation(new_shape, self.cell_size, self.origin, translated_cells)

    @ staticmethod
    def cylinder(d, h, origin=np.array([0, 0, 0]), cell_size=np.array([1, 1, 1])):
        r = d/2
        voxel_model = VoxelRepresentation((d, h, d), cell_size, origin)

        for x, y, z in product(range(d), range(h), range(d)):
            dist = np.linalg.norm([x+0.5-r, z+0.5-r])
            if dist <= r:
                voxel_model.add_voxel((x, y, z), 1)

        return voxel_model

    @ staticmethod
    def sphere(d, origin, cell_size):
        r = d/2
        voxel_model = VoxelRepresentation((d, d, d), cell_size, origin)

        for x, y, z in product(range(d), range(d), range(d)):
            dist = np.linalg.norm([x+0.5-r, y+0.5-r, z+0.5-r])
            if dist <= r:
                voxel_model.add_voxel((x, y, z), 1)

        return voxel_model

    @staticmethod
    def nb6():
        return VoxelRepresentation((3, 3, 3), np.array([1, 1, 1]), np.array([1, 1, 1]),
                                   {(1, 2, 1): None,
                                    (1, 1, 0): None, (0, 1, 1): None, (2, 1, 1): None, (1, 1, 2): None,
                                    (1, 0, 1): None, })

    def nb26():
        return VoxelRepresentation((3, 3, 3), np.array([1, 1, 1]), np.array([1, 1, 1]),
                                   {(x,y,z): None for x, y, z in product(range(3), range(3), range(3))}.pop((1, 1, 1)))

    @ staticmethod
    def box():
        pass


class PointCloudRepresentation:
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
        self.kdt = KDTree(self.points, PointCloudRepresentation.leaf_size)

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

    def scale(self, scale: np.array):
        return PointCloudRepresentation(np.array(self.points*scale), source=self)

    def k_nearest_neighbour(self, p: np.array, k: int) -> Tuple[np.array, np.array]:
        '''Get the first k neighbours that are closest to a given point.'''

        return self.kdt.query(p.reshape(1, -1), k)

    def ball_search(self, p, r):
        '''Get all points within r radius of point p.'''

        return self.kdt.query_radius(p.reshape(1, -1), r)

    def estimate_normals(self, k) -> np.array:
        ''' Estimate normalized point cloud normals using principal component analysis.
            Component with smallest magnitude gives the normal vector.'''

        pca = PCA()

        normals = []
        for p in self.points:
            knn = self.k_nearest_neighbour(p, k)
            nbs = self.points[knn[1]].squeeze()

            pca.fit(nbs)
            normals.append(pca.components_[2] /
                           np.linalg.norm(pca.components_[2]))

        return np.array(normals)

    def filter(self, property, func):
        '''Functional programming filter for point clouds'''

        out_points = []

        for i, p in enumerate(self.points):
            if func(property[i]):
                out_points.append(p)

        return PointCloudRepresentation(np.array(out_points), source=self)

    def voxel_filter(self, cell_size) -> PointCloudRepresentation:
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

        return PointCloudRepresentation(np.array(points), colors=None, source=self)

    def voxelize(self, cell_size) -> VoxelRepresentation:
        '''Convert point cloud to discretized voxel representation.'''

        edge_sizes = self.aabb[:, 1] - self.aabb[:, 0]
        shape = edge_sizes // cell_size
        cell_sizes = edge_sizes / shape

        voxel_model = VoxelRepresentation(shape, cell_sizes, self.aabb[:, 0])

        for p in self.points:
            # Find voxel cell that the given point is in
            cell = tuple(((p - self.aabb[:, 0]) // cell_sizes).astype(int))

            if voxel_model.is_occupied(cell):
                voxel_model[cell]['population'] += 1
            else:
                voxel_model.add_voxel(cell)
                voxel_model[cell]['population'] = 1

        return voxel_model

    @ staticmethod
    def read_ply(fn: str) -> PointCloudRepresentation:
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

        return PointCloudRepresentation(points, colors/255, source=fn)
