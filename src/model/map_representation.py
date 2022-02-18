from __future__ import annotations
from ast import Lambda
from fnmatch import translate
from numba import jit, cuda
import numba as nb

import copy
import enum
from itertools import product
from random import random
from typing import Dict, Tuple

import networkx
import numpy as np
import open3d as o3d
from plyfile import PlyData
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import math


class SpatialGraphRepresentation:
    leaf_size = 20

    def __init__(self,
                 scale: np.array = np.array([1, 1, 1]),
                 origin: np.array = np.array([0, 0, 0]),
                 graph: networkx.Graph = networkx.Graph()):

        self.scale = scale
        self.origin = origin

        self.graph = graph

        self.nodes = self.graph.nodes
        self.edges = self.graph.edges
        self.sindex: KDTree = KDTree(self.list_attr(
            'pos'), SpatialGraphRepresentation.leaf_size)

        for i, node in enumerate(self.nodes):
            self.nodes[node]['index'] = i

    def node_index(self, node):
        return self.nodes[node]['index']

    def get_by_index(self, index):
        for node in self.nodes:
            if self.nodes[node]['index'] == index:
                return node

    def list_attr(self, attr):
        return [self.nodes[n][attr] for n in self.nodes]

    def nearest_neighbour(self, p):
        nn_index = self.sindex.query(p, k=1)[1][0][0]
        return self.get_by_index(nn_index)

    def to_o3d(self, has_color=False) -> o3d.geometry.LineSet:
        points = self.nodes
        lines = [(self.node_index(n[0]), self.node_index(n[1]))
                 for n in self.edges]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(
            (np.array(points)*self.scale) + self.origin)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        color = [self.nodes[p]['color'] if has_color else [0, 0, 0]
                 for p in self.nodes]
        line_set.colors = o3d.utility.Vector3dVector(color)

        return line_set

    # Get cartesian coordinates of voxel at index based on grid origin and voxel size
    def local_to_world_coordinates(self, pos):
        return (self.origin - pos) * self.scale

    def to_voxel(self):
        vox = VoxelRepresentation(np.array([0, 0, 0]), self.scale, self.origin, {
                                  node: {'count': 1} for node in self.nodes})
        vox.shape = vox.extents()

        # transfer voxel attributes to nodes
        for node in self.nodes:
            for attr in self.nodes[node].keys():
                vox[node][attr] = self.nodes[node][attr]

        return vox

    def minimum_spanning_tree(self) -> SpatialGraphRepresentation:
        mst = networkx.minimum_spanning_tree(self.graph)

        return SpatialGraphRepresentation(self.scale, self.origin, mst)

    def betweenness_centrality(self, n_target_points: int):
        return networkx.betweenness_centrality(self.graph, n_target_points)

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

    def map(self, func, **kwargs):
        return map(lambda v: func(v, **kwargs), self.voxels)

    def filter(self, func, **kwargs):
        return filter(lambda v: func(v, **kwargs), self.voxels)

    def intersect(self, other: VoxelRepresentation) -> Set[Tuple[int, int, int]]:
        return self.voxels.keys() & other.voxels.keys()

    def disjoint(self, other: VoxelRepresentation) -> bool:
        return self.voxels.keys().isdisjoint(other.voxels.keys())

    def subset(self, func: Lambda, **kwargs) -> VoxelRepresentation:
        filtered_voxels = self.filter(func, **kwargs)
        voxel_dict = {voxel: self.voxels[voxel] for voxel in filtered_voxels}

        return VoxelRepresentation(self.shape, self.cell_size, self.origin, voxel_dict)

    def for_each(self, func, **kwargs):
        for voxel in self.voxels:
            func(voxel, **kwargs)

    def set_attribute(self, voxel, attr, val):
        self.voxels[voxel][attr] = val

    def get_by_attribute(self, attr, val):
        return self.filter(lambda vox: attr in self[vox] and self[vox][attr] == val)

    def list_attribute(self, attr):
        return self.map(lambda vox: self[vox][attr])

    def colorize(self, color):
        self.for_each(self.set_attribute, attr='color', val=color)

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
        pca_components = map(
            lambda nb: VoxelRepresentation.pca(nb).components_, nbs)
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
        return self.intersect(kernel.translate(np.array(cell)))

    def kernel_contains_neighbours(self, cell: Tuple[int, int, int], kernel: VoxelRepresentation) -> bool:
        return not self.disjoint(kernel.translate(np.array(cell - kernel.origin)))

    def to_o3d(self, has_color=False):
        return self.to_pcd(has_color).to_o3d()

    def to_pcd(self, has_color=False) -> PointCloudRepresentation:
        points, colors = [], []

        for voxel in self.voxels:
            points.append(self.origin + (self.cell_size * voxel))
            if has_color:
                colors.append(self.voxels[voxel]['color'])

        return PointCloudRepresentation(np.array(points), colors=colors, source=self)

    def to_graph(self) -> SpatialGraphRepresentation:
        kernel = VoxelRepresentation.nb6()
        graph = networkx.Graph()

        for v in self.voxels:
            nbs = self.get_kernel(v, kernel)
            graph.add_node(v)

            for nb in nbs:
                graph.add_edge(*(v, nb))

            for attr in self.voxels[v]:
                graph.nodes[v][attr] = self.voxels[v][attr]
            graph.nodes[v]['pos'] = self.voxel_coordinates(v)

        return SpatialGraphRepresentation(self.cell_size, self.origin, graph)

    def dilate(self, kernel) -> VoxelRepresentation:
        '''Dilate model with kernel'''

        dilated_model = VoxelRepresentation(
            self.shape, self.cell_size, self.origin, {})

        for v in self.voxels:
            dilated_model.add_voxel(v, {})

            for k in kernel.voxels:
                nb = tuple(v + (k - kernel.origin))
                dilated_model.add_voxel(nb, {})

        return dilated_model

    def translate(self, translation: np.array):
        translated_cells = {}

        for voxel in self.voxels:
            voxel_t = translation + voxel
            translated_cells[tuple(voxel_t)] = self.voxels[voxel]

        translated_map = VoxelRepresentation(
            self.shape, self.cell_size, self.origin, translated_cells)
        translated_map.shape = translated_map.extents()

        return translated_map

    def distance_to_unoccupied(self):
        hipf = []

        for v in self.voxels:
            kernel = VoxelRepresentation.nb4()
            v_nbs = self.get_kernel(v, kernel)

            i = 0
            while len(v_nbs) == len(kernel.voxels):
                kernel = kernel.dilate(VoxelRepresentation.nb4())
                v_nbs = self.get_kernel(v, kernel)
                i += 1

            hipf.append(i)
        return hipf



    # Get cartesian coordinates of voxel at index based on grid origin and voxel size
    def voxel_coordinates(self, current_voxel):
        return self.origin + (current_voxel * self.cell_size) + (0.5*self.cell_size)

    def get_voxel(self, p):
        return tuple(((p - self.origin) // self.cell_size).astype(int))

    def project(self, voxel, axis, step):
        current_voxel = list(voxel)

        while tuple(current_voxel) not in self.voxels:
            if 0 <= current_voxel[0] <= self.shape[0] and \
                    0 <= current_voxel[1] <= self.shape[1] and \
                    0 <= current_voxel[2] <= self.shape[2]:

                current_voxel[axis] += step
            else:
                return None

        return tuple(current_voxel)

    def isovist(self, origin):
        print(origin)

        directions = np.array([self.voxel_coordinates(v) - origin for v in self.voxels], dtype=np.float)
        bbox = np.array([self.origin, self.origin + (self.cell_size * self.shape)], dtype=np.float)

        voxel_matrix = np.full(self.shape.astype(int)+1, 0, dtype=np.int8)
        for v in self.voxels:
            voxel_matrix[v] = 1

        # Allocate memory on the device for the result
        intersections = np.zeros(shape=(len(directions),3), dtype=np.int32)

        threadsperblock = 256
        blockspergrid = (len(directions) + (threadsperblock - 1)) // threadsperblock
        VoxelRepresentation.dda[blockspergrid, threadsperblock](
            origin, directions, bbox, self.cell_size, voxel_matrix, intersections)

        from copy import deepcopy

        isovist = {tuple(i) for i in intersections}
        if (0,0,0) in isovist:
            isovist.remove((0,0,0))

        return isovist

    # "Fast" voxel traversal, based on Amanitides & Woo, 1987
    # Terminates when first voxel has been found
    @cuda.jit
    def dda(point, directions, bbox, cell_size, voxels, intersections):
        pos = cuda.grid(1)

        min_bbox, max_bbox = bbox[0], bbox[1]

        direction = directions[pos]
        # direction = direction / np.linalg.norm(direction)
        # line equation
        x1, y1, z1, l, m, n = point[0], point[1], point[2], direction[0], direction[1], direction[2]

        if (l == 0.0):
            l = 0.00000001
        if (m == 0.0):
            m = 0.00000001
        if (n == 0.0):
            n = 0.00000001

        x_sign = int(l / abs(l))
        y_sign = int(m / abs(m))
        z_sign = int(n / abs(n))

        border_distances = cuda.local.array(shape=3, dtype=nb.float32) 
        border_distances[0] = cell_size[0] * x_sign
        border_distances[1] = cell_size[1] * y_sign
        border_distances[2] = cell_size[2] * z_sign

        current_position = point
        current_voxel_x = int((current_position[0] - min_bbox[0]) // cell_size[0])
        current_voxel_y = int((current_position[1] - min_bbox[1]) // cell_size[1])
        current_voxel_z = int((current_position[2] - min_bbox[2]) // cell_size[2])

        while ( min_bbox[0] <= current_position[0] <= max_bbox[0]   and
                min_bbox[1] <= current_position[1] <= max_bbox[1]   and
                min_bbox[2] <= current_position[2] <= max_bbox[2]):

            current_voxel_center_x = min_bbox[0] + (current_voxel_x * cell_size[0]) + (0.5 * cell_size[0])
            current_voxel_center_y = min_bbox[1] + (current_voxel_y * cell_size[1]) + (0.5 * cell_size[1])
            current_voxel_center_z = min_bbox[2] + (current_voxel_z * cell_size[2]) + (0.5 * cell_size[2])

            # get coordinates of axis-aligned planes that border cell
            x_edge = current_voxel_center_x + border_distances[0]
            y_edge = current_voxel_center_y + border_distances[1]
            z_edge = current_voxel_center_z + border_distances[2]

            # find intersection of line with cell borders and its distance
            x_vec_x = x_edge                          - current_position[0]
            x_vec_y = (((x_edge - x1) / l) * m) + y1  - current_position[1]
            x_vec_z = (((x_edge - x1) / l) * n) + z1  - current_position[2]
            x_magnitude = (x_vec_x**2 + x_vec_y**2 + x_vec_z**2)**(1/2)

            y_vec_x = (((y_edge - y1) / m) * l) + x1  - current_position[0]
            y_vec_y = y_edge                          - current_position[1]
            y_vec_z = (((y_edge - y1) / m) * n) + z1  - current_position[2]
            y_magnitude = (y_vec_x**2 + y_vec_y**2 + y_vec_z**2)**(1/2)

            z_vec_x = (((z_edge - z1) / n) * l) + x1  - current_position[0]
            z_vec_y = (((z_edge - z1) / n) * m) + y1  - current_position[1]
            z_vec_z = z_edge                          - current_position[2]
            z_magnitude = (z_vec_x**2 + z_vec_y**2 + z_vec_z**2)**(1/2)

            if x_magnitude <= y_magnitude and x_magnitude <= z_magnitude:
                current_voxel_x += x_sign
                current_position[0] += x_vec_x
                current_position[1] += x_vec_y
                current_position[2] += x_vec_z
            elif y_magnitude <= x_magnitude and y_magnitude <= z_magnitude:
                current_voxel_y += y_sign
                current_position[0] += y_vec_x
                current_position[1] += y_vec_y
                current_position[2] += y_vec_z
            elif z_magnitude <= x_magnitude and z_magnitude <= y_magnitude:
                current_voxel_z += z_sign
                current_position[0] += z_vec_x
                current_position[1] += z_vec_y
                current_position[2] += z_vec_z

            if voxels[current_voxel_x, current_voxel_y, current_voxel_z] == 1:
                intersections[pos][0] += current_voxel_x
                intersections[pos][1] += current_voxel_y
                intersections[pos][2] += current_voxel_z

                cuda.syncthreads()
                break

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
                voxel_model.add_voxel((x, y, z), {})

        return voxel_model

    @staticmethod
    def nb6():
        return VoxelRepresentation((3, 3, 3), np.array([1, 1, 1]), np.array([1, 1, 1]),
                                   {(1, 2, 1): None,
                                    (1, 1, 0): None, (0, 1, 1): None, (2, 1, 1): None, (1, 1, 2): None,
                                    (1, 0, 1): None, })

    @staticmethod
    def nb4():
        '''Horizontal neighbours'''

        return VoxelRepresentation((3, 3, 1), np.array([1, 1, 1]), np.array([1, 1, 1]),
                                   {(1, 1, 0): None, (0, 1, 1): None, (2, 1, 1): None, (1, 1, 2): None})

    @staticmethod
    def stick_kernel(scale):
        a_height = int(15 // scale)
        a_width = 1 + int(5 // scale)

        b_height = int(5 // scale)
        b_width = 1

        kernel_a = VoxelRepresentation.cylinder(
            a_width, a_height).translate(np.array([0, b_height, 0]))
        kernel_b = VoxelRepresentation.cylinder(b_width, b_height).translate(
            np.array([a_width//2, 0, a_width//2]))
        kernel_c = kernel_a + kernel_b

        kernel_c.origin = np.array([a_width//2, 0, a_width//2])
        kernel_c.remove_voxel((a_width//2, 0, a_width//2))

        return kernel_c


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
        '''Functional filter for point cloud'''

        out_points = []
        for i, p in enumerate(self.points):
            if func(property[i]):
                out_points.append(p)

        return PointCloudRepresentation(np.array(out_points), source=self)

    def voxelize(self, cell_size) -> VoxelRepresentation:
        '''Convert point cloud to discretized voxel representation.'''

        edge_sizes = self.aabb[:, 1] - self.aabb[:, 0]
        shape = edge_sizes // cell_size
        cell_sizes = edge_sizes / shape

        voxel_model = VoxelRepresentation(
            shape, cell_sizes, self.aabb[:, 0], {})

        for p in self.points:
            # Find voxel cell that the given point is in
            cell = tuple(((p - self.aabb[:, 0]) // cell_sizes).astype(int))

            if not voxel_model.is_occupied(cell):
                voxel_model.add_voxel(cell, {'pos': p})

        return voxel_model

    def random_reduce(self, keep_fraction: float) -> PointCloudRepresentation:
        keep_points = []

        for p in self.points:
            if random() < keep_fraction:
                keep_points.append(p)

        return PointCloudRepresentation(np.array(keep_points), source=self)

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


class TopoMetricRepresentation():
    pass
