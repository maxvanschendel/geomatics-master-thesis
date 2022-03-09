from __future__ import annotations

import math
from ast import Lambda
from collections import Counter
from copy import deepcopy
from itertools import product
from typing import Dict, Tuple
from warnings import filterwarnings

import networkx
import numpy as np
from misc.helpers import most_common
from numba import cuda

import model.point_cloud
from model.sparse_voxel_octree import *
from model.spatial_graph import *

filterwarnings('ignore')

class VoxelGrid:
    cluster_attr: str = 'cluster'
    color_attr: str = 'color'

    def __init__(self,
                 shape: np.array = None,
                 cell_size: np.array = np.array([1, 1, 1]),
                 origin: np.array = np.array([0, 0, 0]),
                 voxels: Dict[Tuple[int, int, int], Dict[str, object]] = None):

        self.shape = shape
        self.cell_size = cell_size
        self.origin = origin

        if voxels is None:
            voxels = {}
        self.voxels = voxels
        
    def clone(self) -> VoxelGrid:
        return VoxelGrid(deepcopy(self.shape),
                         deepcopy(self.cell_size),
                         deepcopy(self.origin),
                         deepcopy(self.voxels))
        
    def __getitem__(self, key: Tuple[int, int, int]) -> Dict[str, object]:
        return self.voxels[key]

    def __setitem__(self, key: Tuple[int, int, int], value: Dict[str, object]) -> None:
        self.voxels[key] = value

    def __add__(self, other: VoxelGrid) -> VoxelGrid:
        new_model = self.clone()
        for v in other.voxels:
            new_model.voxels[v] = other.voxels[v]

        # Adapt size of voxel model to new extents after addition
        new_model.shape = new_model.extents()
        return new_model

    def __contains__(self, val):
        return val in self.voxels

    def map(self, func, **kwargs):
        return map(lambda v: func(v, **kwargs), self.voxels)

    def filter(self, func, **kwargs):
        return filter(lambda v: func(v, **kwargs), self.voxels)

    def add(self, cell: Tuple[int, int, int], value: Dict[str, object] = {}) -> None:
        self.voxels[cell] = value

    def remove(self, cell: Tuple[int, int, int]) -> None:
        self.voxels.pop(cell)

    def voxel_centroids(self) -> np.array:
        return np.array(list(self.map(self.voxel_coordinates)))
    
    def get_voxel(self, p: np.array) -> Tuple[int, int, int]:
        return tuple(((p-self.origin) // self.cell_size).astype(int))

    def contains_point(self, point: np.array) -> bool:
        return self.get_voxel(point) in self

    def intersect(self, other: VoxelGrid) -> Set[Tuple[int, int, int]]:
        return self.voxels.keys() & other.voxels.keys()

    def subset(self, func: Lambda, **kwargs) -> VoxelGrid:
        filtered_voxels = self.filter(func, **kwargs)
        voxel_dict = {voxel: self.voxels[voxel] for voxel in filtered_voxels}

        return VoxelGrid(deepcopy(self.shape), deepcopy(self.cell_size), deepcopy(self.origin), voxel_dict)

    def for_each(self, func: Lambda, **kwargs) -> None:
        for voxel in self.voxels:
            func(voxel, **kwargs)

    def set_attr(self, voxel, attr, val):
        self.voxels[voxel][attr] = val

    def set_attr_uniform(self, attr, val):
        self.for_each(self.set_attr, attr=attr, val=val)
        
    def colorize(self, color):
        self.set_attr_uniform(attr=VoxelGrid.color_attr, val=color)

    def get_attr(self, attr, val):
        return self.filter(lambda vox: attr in self[vox] and self[vox][attr] == val)

    def list_attr(self, attr):
        voxels_with_attr = self.filter(lambda vox: attr in self.voxels[vox])
        return map(lambda vox: self.voxels[vox][attr], voxels_with_attr)

    def kernel_attr(self, voxel: Tuple[int, int, int], kernel: VoxelGrid, attr: str):
        voxel_nbs = self.get_kernel(voxel, kernel)
        voxel_nbs_labels = [self[v][attr] for v in voxel_nbs]

        return voxel_nbs_labels

    def most_common_kernel_attr(self, voxel: Tuple[int, int, int], kernel: VoxelGrid, attr: str):
        kernel_attrs = self.kernel_attr(voxel, kernel, attr)
        most_common_attr = most_common(kernel_attrs)

        return most_common_attr

    def propagate_attr(self, attr: str, max_iterations: int, prop_kernel: VoxelGrid) -> VoxelGrid:
        prop_map = self.clone()
        attributes = list(self.list_attr(attr))
        unique_attributes = np.unique(attributes)

        # Get most common label of 6-neighbourhood of voxel
        # Assign most common label to voxel in propagated map
        for i in range(max_iterations):
            prev_map = prop_map.clone()

            # Allocate memory on the device for the result
            max_nb = np.zeros(shape=(len(self.voxels), 1), dtype=np.int32)

            voxel_matrix = np.full(self.shape.astype(int)+1, -1, dtype=np.int8)
            for v in self.voxels:
                voxel_matrix[v] = int(prev_map.voxels[v][attr])

            kernel_voxels = np.zeros(
                shape=(len(prop_kernel.voxels), 3), dtype=np.int32)
            for i, k in enumerate(prop_kernel.voxels):
                kernel_voxels[i][0] = k[0]
                kernel_voxels[i][1] = k[1]
                kernel_voxels[i][2] = k[2]

            origin_voxels = np.zeros(
                shape=(len(self.voxels), 3), dtype=np.int32)
            for i, k in enumerate(self.voxels):
                origin_voxels[i][0] = k[0]
                origin_voxels[i][1] = k[1]
                origin_voxels[i][2] = k[2]

            nbs_occurences = np.zeros(
                shape=(len(self.voxels), len(unique_attributes)), dtype=np.int32)
            n_vals = len(unique_attributes)

            threadsperblock = 64
            blockspergrid = (len(self.voxels)) // threadsperblock
            VoxelGrid.convolve_most_common_nb_gpu[blockspergrid, threadsperblock](
                voxel_matrix, origin_voxels, kernel_voxels, nbs_occurences, n_vals, max_nb)

            for i, label in enumerate(max_nb):
                prop_map[tuple(origin_voxels[i])][attr] = label

            # Stop propagation once it fails to make a change
            if prev_map.voxels == prop_map.voxels:
                break

        return prop_map

    def attr_borders(self, attr, kernel) -> VoxelGrid:
        border_voxels = self.clone()

        for voxel in self.voxels:
            voxel_nbs = self.get_kernel(voxel, kernel)
            voxel_nbs_labels = [self[v][attr] for v in voxel_nbs]

            if voxel_nbs_labels == [self[voxel][attr]]*len(voxel_nbs_labels):
                border_voxels.remove(voxel)

        return border_voxels

    def split_by_attr(self, attr) -> List[VoxelGrid]:
        attributes = list(self.list_attr(attr))
        unique_attributes = np.unique(attributes)

        split = []
        for unique_attr in unique_attributes:
            attr_subset = self.subset(
                lambda v: attr in self[v] and self[v][attr] == unique_attr)
            split.append(attr_subset)

        return split
    
    def extents(self):
        new_voxels = np.array(list(self.voxels.keys()))
        new_shape = list(self.shape)

        for v in new_voxels:
            for i, c in enumerate(self.shape):
                new_shape[i] = max([v[i].max(), c])
        return new_shape

    def centroid(self) -> np.array:
        return np.mean([self.voxel_coordinates(v) for v in self.voxels], axis=0)
    
    def voxel_coordinates(self, voxel: Tuple[int, int, int]) -> np.array:
        return self.origin + voxel * self.cell_size + 0.5 * self.cell_size

    def get_kernel(self, voxel: Tuple[int, int, int], kernel: VoxelGrid):
        kernel_cells = []
        for k in kernel.voxels:
            nb = tuple(voxel + (k - kernel.origin))
            if nb in self:
                kernel_cells.append(nb)

        return kernel_cells

    def connected_components(self, kernel) -> List[VoxelGrid]:
        graph_representation = self.to_graph(kernel)
        components = graph_representation.connected_components()

        comps = []
        for c in components:
            component = SpatialGraph(
                graph_representation.scale,
                graph_representation.origin,
                graph_representation.graph.subgraph(c))
            component_voxel = component.to_voxel()
            comps.append(component_voxel)

        return comps

    @cuda.jit
    def convolve_has_nbs_gpu(voxels, occupied_voxels, kernel, conv_voxels):
        pos = cuda.grid(1)
        current_voxel = occupied_voxels[pos]

        for nb in kernel:
            if voxels[
                current_voxel[0] + nb[0],
                current_voxel[1] + nb[1],
                current_voxel[2] + nb[2]
            ] == 1:

                conv_voxels[pos] = 1
                break

    @cuda.jit
    def convolve_most_common_nb_gpu(voxels, origin_voxels, kernel, nbs_occurences, n_vals, max_occurences):
        pos = cuda.grid(1)
        cur_voxel = origin_voxels[pos]

        for nb in kernel:
            nb_val = voxels[
                cur_voxel[0] + nb[0],
                cur_voxel[1] + nb[1],
                cur_voxel[2] + nb[2]
            ]

            if nb_val != -1:
                nbs_occurences[pos][nb_val] += 1

        max_occurence, max_val = 1, voxels[cur_voxel[0], cur_voxel[1], cur_voxel[2]]
        for val in range(n_vals):
            if nbs_occurences[pos][val] > max_occurence:
                max_occurence = nbs_occurences[pos][val]
                max_val = val

        max_occurences[pos] = max_val

    def filter_gpu_kernel_nbs(self, kernel):
        # Allocate memory on the device for the result
        result_voxels = np.zeros(shape=(len(self.voxels), 1), dtype=np.int32)
        voxel_matrix = np.full(self.shape.astype(int)+1, 0, dtype=np.int8)
        for v in self.voxels:
            voxel_matrix[v] = 1

        kernel_voxels = np.zeros(shape=(len(kernel.voxels), 3), dtype=np.int32)
        for i, k in enumerate(kernel.voxels):
            kernel_voxels[i][0] = k[0]
            kernel_voxels[i][1] = k[1]
            kernel_voxels[i][2] = k[2]

        occupied_voxels = np.zeros(shape=(len(self.voxels), 3), dtype=np.int32)
        for i, k in enumerate(self.voxels):
            occupied_voxels[i][0] = k[0]
            occupied_voxels[i][1] = k[1]
            occupied_voxels[i][2] = k[2]

        threadsperblock = 512
        blockspergrid = (len(self.voxels) +
                         (threadsperblock - 1)) // threadsperblock
        VoxelGrid.convolve_has_nbs_gpu[blockspergrid, threadsperblock](
            voxel_matrix, occupied_voxels, kernel_voxels, result_voxels)

        nb_voxels = []
        for i, occ in enumerate(result_voxels):
            if not occ[0]:
                nb_voxels.append(tuple(occupied_voxels[i]))

        return VoxelGrid(self.shape, self.cell_size, self.origin, {v: self.voxels[v] for v in nb_voxels})

    def dilate(self, kernel) -> VoxelGrid:
        dilated_model = VoxelGrid(
            self.shape, self.cell_size, self.origin)

        for v in self.voxels:
            dilated_model.add(v)

            for k in kernel.voxels:
                nb = tuple(v + (k - kernel.origin))
                dilated_model.add(nb)

        return dilated_model

    def translate(self, translation: np.array):
        translated_cells = {}

        for voxel in self.voxels:
            voxel_t = translation + voxel
            translated_cells[tuple(voxel_t)] = self.voxels[voxel]

        translated_map = VoxelGrid(
            self.shape, self.cell_size, self.origin, translated_cells)
        translated_map.shape = translated_map.extents()

        return translated_map

    def visibility(self, origin: np.array, max_dist: float) -> VoxelGrid:
        # Get all voxels within range (max_dist) of the isovist
        # For each voxel within range, get a vector pointing from origin to the voxel
        candidate_voxels = np.array(
            [self.voxel_coordinates(v) for v in self.voxels])
        candidate_directions = candidate_voxels - origin
        range_filter = np.linalg.norm(candidate_directions, axis=1) <= max_dist
        directions = candidate_directions[range_filter]

        # If no voxels are within range return an empty isovist
        if len(directions) == 0:
            return VoxelGrid(shape=self.shape,
                             cell_size=self.cell_size,
                             origin=self.origin)

        # Construct a 3D matrix where occupied voxels are 1 and unoccupied are 0
        # This matrix is used by the GPU to quickly find occupied voxels at the cost of memory
        voxel_matrix = np.full(self.shape.astype(int)+1, 0, dtype=np.int8)
        for v in self.voxels:
            voxel_matrix[v] = 1

        # Allocate memory on the device for the intersection results
        intersections = np.zeros(shape=(len(directions), 3), dtype=np.int32)
        
        bbox = np.array([self.origin, self.origin + (self.cell_size * self.shape)], dtype=np.float)

        # Cast isovists on GPU
        threadsperblock = 256
        blockspergrid = (len(directions) // threadsperblock)
        VoxelGrid.fast_voxel_traversal[blockspergrid, threadsperblock](
            origin, directions, bbox, self.cell_size, voxel_matrix, intersections, max_dist)

        intersection_tuples = [tuple(i) for i in intersections]
        isovist = {i for i in intersection_tuples if i != (0,0,0)}

        return VoxelGrid(shape=self.shape,
                         cell_size=self.cell_size,
                         origin=self.origin,
                         voxels={v: self[v] for v in isovist})

    @cuda.jit
    def fast_voxel_traversal(point, directions, bbox, cell_size, voxels, intersections, max_dist):
        """
        "Fast" voxel traversal, based on Amanitides & Woo (1987)
        Terminates when first voxel has been found

        Args:
            point (_type_): _description_
            directions (_type_): _description_
            bbox (_type_): _description_
            cell_size (_type_): _description_
            voxels (_type_): _description_
            intersections (_type_): _description_
            max_dist (_type_): _description_
        """

        pos = cuda.grid(1)
        min_bbox, max_bbox = bbox[0], bbox[1]
        direction = directions[pos]

        # line equation
        x1, y1, z1, l, m, n = point[0], point[1], point[2], direction[0], direction[1], direction[2]

        if (l == 0.0):
            l = 1/math.inf
        if (m == 0.0):
            m = 1/math.inf
        if (n == 0.0):
            n = 1/math.inf

        x_sign = int(l / abs(l))
        y_sign = int(m / abs(m))
        z_sign = int(n / abs(n))

        # Distance from voxel center to plane for each axis
        bd_x = cell_size[0] * x_sign
        bd_y = cell_size[1] * y_sign
        bd_z = cell_size[2] * z_sign

        current_position = point
        current_voxel_x = int(
            (current_position[0] - min_bbox[0]) // cell_size[0])
        current_voxel_y = int(
            (current_position[1] - min_bbox[1]) // cell_size[1])
        current_voxel_z = int(
            (current_position[2] - min_bbox[2]) // cell_size[2])

        while (min_bbox[0] <= current_position[0] <= max_bbox[0] and
                min_bbox[1] <= current_position[1] <= max_bbox[1] and
                min_bbox[2] <= current_position[2] <= max_bbox[2] and
                ((x1 - current_position[0])**2 + (y1 - current_position[1])**2 + (z1 - current_position[2])**2)**(1/2) < max_dist):

            if voxels[current_voxel_x, current_voxel_y, current_voxel_z] == 1:
                intersections[pos][0] += current_voxel_x
                intersections[pos][1] += current_voxel_y
                intersections[pos][2] += current_voxel_z
                break

            current_voxel_center_x = min_bbox[0] + \
                (current_voxel_x * cell_size[0]) + (0.5 * cell_size[0])
            current_voxel_center_y = min_bbox[1] + \
                (current_voxel_y * cell_size[1]) + (0.5 * cell_size[1])
            current_voxel_center_z = min_bbox[2] + \
                (current_voxel_z * cell_size[2]) + (0.5 * cell_size[2])

            # get coordinates of axis-aligned planes that border cell
            x_edge = current_voxel_center_x + bd_x
            y_edge = current_voxel_center_y + bd_y
            z_edge = current_voxel_center_z + bd_z

            # find intersection of line with cell borders and its distance
            x_vec_x = x_edge - current_position[0]
            x_vec_y = (((x_edge - x1) / l) * m) + y1 - current_position[1]
            x_vec_z = (((x_edge - x1) / l) * n) + z1 - current_position[2]
            x_magnitude = (x_vec_x**2 + x_vec_y**2 + x_vec_z**2)**(1/2)

            y_vec_x = (((y_edge - y1) / m) * l) + x1 - current_position[0]
            y_vec_y = y_edge - current_position[1]
            y_vec_z = (((y_edge - y1) / m) * n) + z1 - current_position[2]
            y_magnitude = (y_vec_x**2 + y_vec_y**2 + y_vec_z**2)**(1/2)

            z_vec_x = (((z_edge - z1) / n) * l) + x1 - current_position[0]
            z_vec_y = (((z_edge - z1) / n) * m) + y1 - current_position[1]
            z_vec_z = z_edge - current_position[2]
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

    def detect_peaks(self, axis: int, height: int = None, width: int = None) -> Tuple[int]:
        from scipy.signal import find_peaks

        axis_values = [v[axis] for v in self.voxels]
        axis_occurences = Counter(axis_values)

        x = [0]*(max(axis_values)+1)
        for k in axis_occurences.keys():
            x[k] = axis_occurences[k]

        peaks, _ = find_peaks(np.array(x), height=height)
        return peaks

    def distance_field(self) -> np.array:
        hipf = []
        for v in self.voxels:
            i = 1
            
            kernel = Kernel.circle(r=i)
            while len(self.get_kernel(v, kernel)) == len(kernel.voxels):
                i += 1
                kernel = Kernel.circle(r=i)

            hipf.append(i)
        return np.array(hipf)

    def local_distance_field_maxima(self, kernel_radius=5) -> VoxelGrid:
        kernel = Kernel.sphere(r=kernel_radius)

        df = self.distance_field()
        distance_field = {v: d for d, v in zip(df, list(self.voxels))}

        local_maxima = set()
        for vx in distance_field:
            vx_dist = distance_field[vx]
            vx_nbs = self.get_kernel(vx, kernel)
            vx_nbs_dist = [distance_field[nb] for nb in vx_nbs]

            if vx_dist >= max(vx_nbs_dist):
                local_maxima.add(vx)

        return self.subset(lambda v: v in local_maxima)

    def to_o3d(self, has_color=False):
        return self.to_pcd(has_color).to_o3d()

    def to_pcd(self, has_color=False) -> model.point_cloud.PointCloud:
        points, colors = [], []

        for voxel in self.voxels:
            points.append(self.voxel_coordinates(voxel))
            if has_color:
                colors.append(self.voxels[voxel][VoxelGrid.color_attr])

        return model.point_cloud.PointCloud(np.array(points), colors=colors, source=self)

    def to_graph(self, kernel: Kernel = None) -> SpatialGraph:
        if not kernel:
            kernel = Kernel.nb6()
            
        graph = networkx.Graph()
        for v in self.voxels:
            nbs = self.get_kernel(v, kernel)
            graph.add_node(v)

            for nb in nbs:
                graph.add_edge(*(v, nb))

            for attr in self.voxels[v]:
                graph.nodes[v][attr] = self.voxels[v][attr]
            graph.nodes[v]['pos'] = self.voxel_coordinates(v)

        return SpatialGraph(self.cell_size, self.origin, graph)

class Kernel(VoxelGrid):
    @ staticmethod
    def cylinder(d, h, origin=np.array([0, 0, 0]), cell_size=np.array([1, 1, 1])):
        r = d/2
        voxel_model = VoxelGrid((d, h, d), cell_size, origin)

        for x, y, z in product(range(d), range(h), range(d)):
            dist = np.linalg.norm([x+0.5-r, z+0.5-r])
            if dist <= r:
                voxel_model.add((x, y, z), 1)

        return voxel_model

    @staticmethod
    def sphere(r):
        d = r*2
        voxel_model = VoxelGrid((d, d, d), np.array(
            [1, 1, 1]), np.array([0, 0, 0]))

        for x, y, z in product(range(-r, r+1), range(-r, r+1), range(-r, r+1)):
            dist = np.linalg.norm([x, y, z])
            if dist <= r:
                voxel_model.add((x, y, z), {})

        return voxel_model

    @staticmethod
    def circle(r):
        d = r*2
        voxel_model = VoxelGrid((d, 1, d), np.array(
            [1, 1, 1]), np.array([0, 0, 0]))

        for x, z in product(range(-r, r+1), range(-r, r+1)):
            dist = np.linalg.norm([x, z])
            if dist <= r:
                voxel_model.add((x, 0, z), {})

        return voxel_model

    @staticmethod
    def nb6():
        return VoxelGrid((3, 3, 3), np.array([1, 1, 1]), np.array([1, 1, 1]),
                         {(1, 2, 1): None,
                          (1, 1, 0): None, (0, 1, 1): None, (2, 1, 1): None, (1, 1, 2): None,
                          (1, 0, 1): None, })

    @staticmethod
    def nb4():
        return VoxelGrid((3, 3, 1), np.array([1, 1, 1]), np.array([1, 1, 1]), {(1, 1, 0): None, (0, 1, 1): None, (2, 1, 1): None, (1, 1, 2): None})

    @staticmethod
    def stick_kernel(scale):
        a_height = int(15 // scale)
        a_width = 1 + int(5 // scale)

        b_height = int(5 // scale)
        b_width = 1

        kernel_a = Kernel.cylinder(
            a_width, a_height).translate(np.array([0, b_height, 0]))
        kernel_b = Kernel.cylinder(b_width, b_height).translate(
            np.array([a_width//2, 0, a_width//2]))
        kernel_c = kernel_a + kernel_b

        kernel_c.origin = np.array([a_width//2, 0, a_width//2])
        kernel_c.remove((a_width//2, 0, a_width//2))

        kernel_c = kernel_c.translate(-kernel_c.origin)
        return kernel_c
