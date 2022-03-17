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
from scipy.signal import find_peaks

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
        self.voxels = voxels if voxels is not None else {}
        
        if self.voxels:
            self.size = len(voxels)
            self.svo = SVO.from_voxels(self.voxels.keys(), self.cell_size[0]/2)
        else:
            self.size = 0
            self.svo = None            

    def clone(self) -> VoxelGrid:
        return deepcopy(self)

    def level_of_detail(self, level):
        lod_voxels = self.svo.get_depth(self.svo.max_depth() - level)
        lod_geometry = VoxelGrid(self.shape, self.cell_size * (2**level),
                                 self.origin, {tuple(v): {} for v in lod_voxels})
        return lod_geometry

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
        new_model.size = len(new_model.voxels)
        return new_model

    def __contains__(self, val):
        return val in self.voxels

    def map(self, func, **kwargs):
        return map(lambda v: func(v, **kwargs), self.voxels)

    def filter(self, func, **kwargs):
        return filter(lambda v: func(v, **kwargs), self.voxels)

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

    def unique_attr(self, attr):
        attributes = list(self.list_attr(attr))
        return np.unique(attributes)

    def kernel_attr(self, voxel: Tuple[int, int, int], kernel: VoxelGrid, attr: str):
        voxel_nbs = self.get_kernel(voxel, kernel)
        voxel_nbs_labels = [self[v][attr] for v in voxel_nbs]

        return voxel_nbs_labels

    def most_common_kernel_attr(self, voxel: Tuple[int, int, int], kernel: VoxelGrid, attr: str):
        kernel_attrs = self.kernel_attr(voxel, kernel, attr)
        most_common_attr = most_common(kernel_attrs)

        return most_common_attr

    def to_2d_array(self):
        arr = np.zeros((self.size, 3), dtype=np.int32)
        for i, k in enumerate(self.voxels):
            arr[i] = k
        return arr

    def to_3d_array(self, attr=None):
        voxel_matrix = np.full(self.shape.astype(int)+1, 0, dtype=np.int8)
        for v in self.voxels:
            voxel_matrix[v] = self[v][attr] if attr else 1
        return voxel_matrix

    def propagate_attr(self, attr: str, prop_kernel: VoxelGrid) -> VoxelGrid:
        prop_map = self.clone()
        unique_attr = self.unique_attr(attr)

        kernel_voxels = cuda.to_device(prop_kernel.to_2d_array())
        origin_voxels = cuda.to_device(self.to_2d_array())

        # Get most common label of 6-neighbourhood of voxel
        # Assign most common label to voxel in propagated map
        propagating = True
        while propagating:
            prev_map = prop_map.clone()

            # Allocate memory on the device for the result
            max_nb = np.zeros((self.size, 1), dtype=np.int32)
            nbs_occurences = np.zeros(
                (self.size, len(unique_attr)), dtype=np.int32)

            voxel_matrix = np.full(
                self.shape.astype(int)+1, -1, dtype=np.int32)
            for v in self.voxels:
                voxel_matrix[v] = int(prev_map.voxels[v][attr])

            threadsperblock = 128
            blockspergrid = (self.size // threadsperblock) + 1
            VoxelGrid.convolve_most_common_nb_gpu[blockspergrid, threadsperblock](
                voxel_matrix, origin_voxels, kernel_voxels, nbs_occurences, len(unique_attr), max_nb)

            for i, label in enumerate(max_nb):
                prop_map[tuple(origin_voxels[i])][attr] = label

            # Stop propagation once it fails to make a change
            propagating = prev_map.voxels != prop_map.voxels

        return prop_map

    def attr_borders(self, attr, kernel) -> VoxelGrid:
        border_voxels = set()
        for voxel in self.voxels:
            voxel_nbs = self.get_kernel(voxel, kernel)
            voxel_nbs_labels = [self[v][attr] for v in voxel_nbs]

            # Check if neighbourhood all has the same attribute value
            if voxel_nbs_labels != [self[voxel][attr]]*len(voxel_nbs_labels):
                border_voxels.add(voxel)

        return self.subset(lambda v: v in border_voxels)

    def split_by_attr(self, attr) -> List[VoxelGrid]:
        split = []
        for unique_attr in self.unique_attr(attr):
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

    def voxel_centroid(self, voxel: Tuple[int, int, int]) -> np.array:
        return self.origin + voxel * self.cell_size + 0.5 * self.cell_size

    def voxel_centroids(self) -> np.array:
        return np.array(list(self.map(self.voxel_centroid)))

    def centroid(self) -> np.array:
        return np.mean(self.voxel_centroids(), axis=0)

    def bbox(self) -> np.array:
        centroids = self.voxel_centroids()
        min_bbox, max_bbox = centroids.min(axis=0), centroids.max(axis=0)
        return np.array([min_bbox, max_bbox])

    def get_kernel(self, voxel: Tuple[int, int, int], kernel: VoxelGrid) -> List[Tuple[int, int, int]]:
        kernel_cells = []
        for k in kernel.voxels:
            nb = tuple(voxel + (k - kernel.origin))
            if nb in self:
                kernel_cells.append(nb)

        return kernel_cells

    def radius_search(self, p: np.array, r: float) -> List[Tuple[int, int, int]]:
        radius_morton = self.svo.radius_search(p - self.origin, r)
        return [tuple(v) for v in radius_morton]

    def range_search(self, aabb: np.array) -> List[Tuple[int, int, int]]:
        range_morton = self.svo.range_search(aabb - self.origin)
        return [tuple(v) for v in range_morton]

    def connected_components(self, kernel) -> List[VoxelGrid]:
        graph_representation = self.to_graph(kernel)
        components = graph_representation.connected_components()
        components_voxel = [self.subset(lambda v: v in c) for c in components]

        return components_voxel

    @cuda.jit
    def convolve_has_nbs_gpu(voxels, occupied_voxels, kernel, conv_voxels):
        pos = cuda.grid(1)
        cur_vox = occupied_voxels[pos]

        for nb in kernel:
            if voxels[
                cur_vox[0]+nb[0],
                cur_vox[1]+nb[1],
                cur_vox[2]+nb[2]
            ] == 1:

                conv_voxels[pos] = 1
                break

    @cuda.jit
    def convolve_most_common_nb_gpu(voxels, origin_voxels, kernel, nbs_occurences, n_vals, max_occurences):
        pos = cuda.grid(1)
        cur_vox = origin_voxels[pos]

        for nb in kernel:
            nb_val = voxels[
                cur_vox[0]+nb[0],
                cur_vox[1]+nb[1], 
                cur_vox[2]+nb[2]]
            if nb_val != -1:
                nbs_occurences[pos][nb_val] += 1

        max_occurence = 1 
        max_val = voxels[cur_vox[0], cur_vox[1], cur_vox[2]]
        
        for val in range(n_vals):
            if nbs_occurences[pos][val] > max_occurence:
                max_occurence = nbs_occurences[pos][val]
                max_val = val

        max_occurences[pos] = max_val

    def mutate_voxels(self, voxels):
        return VoxelGrid(deepcopy(self.shape),
                         deepcopy(self.cell_size),
                         deepcopy(self.origin),
                         voxels)

    def filter_gpu_kernel_nbs(self, kernel):
        # Allocate memory on the device for the result
        result_voxels = np.zeros(shape=(self.size, 1), dtype=np.int32)

        voxel_matrix = cuda.to_device(self.to_3d_array())
        occupied_voxels = cuda.to_device(self.to_2d_array())
        kernel_voxels = cuda.to_device(kernel.to_2d_array())

        threadsperblock = 1024
        blockspergrid = (self.size + (threadsperblock - 1)) // threadsperblock
        VoxelGrid.convolve_has_nbs_gpu[blockspergrid, threadsperblock](
            voxel_matrix, occupied_voxels, kernel_voxels, result_voxels)

        nb_voxels = [occupied_voxels[v] for (v, occ) in enumerate(result_voxels) if not occ[0]]
        nb_voxels = [tuple(v) for v in nb_voxels]

        return self.mutate_voxels({v: self.voxels[v] for v in nb_voxels})

    def dilate(self, kernel) -> VoxelGrid:
        dilated_voxels = set(self.voxels.keys())
        for v in self.voxels:
            kernel_voxels = {tuple(v + (k - kernel.origin)) for k in kernel.voxels}
            dilated_voxels.update(kernel_voxels)

        return self.mutate_voxels({v: {} for v in dilated_voxels})

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
        radius_voxels = self.radius_search(origin, max_dist)
        if not len(radius_voxels):
            return self.mutate_voxels({})
        directions = [self.voxel_centroid(v) for v in radius_voxels] - origin

        # This matrix is used by the GPU to quickly find occupied voxels at the cost of memory
        voxel_matrix = self.to_3d_array()

        # Allocate memory on the device for the intersection results
        intersections = np.zeros(shape=(len(directions), 3), dtype=np.int32)
        bbox = np.array([self.origin, self.origin +
                        (self.cell_size * self.shape)])

        # Cast isovists on GPU
        threadsperblock = 512
        blockspergrid = (len(directions) // threadsperblock) + 1
        VoxelGrid.fast_voxel_traversal[blockspergrid, threadsperblock](
            origin, directions, bbox, self.cell_size, voxel_matrix, intersections, max_dist)

        intersection_tuples = [tuple(i) for i in intersections]
        isovist = {i for i in intersection_tuples if i != (0, 0, 0)}

        return self.subset(lambda v: v in isovist)

    @cuda.jit
    def fast_voxel_traversal(point, directions, bbox, cell_size, voxels, intersections, max_dist):
        """
        "Fast" voxel traversal, based on Amanitides & Woo (1987)
        Terminates when first voxel has been found
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

        cur_pos = point
        cur_vox_x = int((cur_pos[0] - min_bbox[0]) // cell_size[0])
        cur_vox_y = int((cur_pos[1] - min_bbox[1]) // cell_size[1])
        cur_vox_z = int((cur_pos[2] - min_bbox[2]) // cell_size[2])

        i = 0
        while (min_bbox[0] <= cur_pos[0] <= max_bbox[0] and
                min_bbox[1] <= cur_pos[1] <= max_bbox[1] and
                min_bbox[2] <= cur_pos[2] <= max_bbox[2]):

            if voxels[cur_vox_x, cur_vox_y, cur_vox_z] == 1:
                intersections[pos][0] += cur_vox_x
                intersections[pos][1] += cur_vox_y
                intersections[pos][2] += cur_vox_z
                break

            cur_vox_center_x = min_bbox[0] + \
                (cur_vox_x * cell_size[0]) + (0.5 * cell_size[0])
            cur_vox_center_y = min_bbox[1] + \
                (cur_vox_y * cell_size[1]) + (0.5 * cell_size[1])
            cur_vox_center_z = min_bbox[2] + \
                (cur_vox_z * cell_size[2]) + (0.5 * cell_size[2])

            # get coordinates of axis-aligned planes that border cell
            x_edge = cur_vox_center_x + bd_x
            y_edge = cur_vox_center_y + bd_y
            z_edge = cur_vox_center_z + bd_z

            # find intersection of line with cell borders and its distance
            x_vec_x = x_edge - cur_pos[0]
            x_vec_y = (((x_edge - x1) / l) * m) + y1 - cur_pos[1]
            x_vec_z = (((x_edge - x1) / l) * n) + z1 - cur_pos[2]
            x_magnitude = (x_vec_x**2 + x_vec_y**2 + x_vec_z**2)**(1/2)

            y_vec_x = (((y_edge - y1) / m) * l) + x1 - cur_pos[0]
            y_vec_y = y_edge - cur_pos[1]
            y_vec_z = (((y_edge - y1) / m) * n) + z1 - cur_pos[2]
            y_magnitude = (y_vec_x**2 + y_vec_y**2 + y_vec_z**2)**(1/2)

            z_vec_x = (((z_edge - z1) / n) * l) + x1 - cur_pos[0]
            z_vec_y = (((z_edge - z1) / n) * m) + y1 - cur_pos[1]
            z_vec_z = z_edge - cur_pos[2]
            z_magnitude = (z_vec_x**2 + z_vec_y**2 + z_vec_z**2)**(1/2)

            if x_magnitude <= y_magnitude and x_magnitude <= z_magnitude:
                cur_vox_x += x_sign
                cur_pos[0] += x_vec_x
                cur_pos[1] += x_vec_y
                cur_pos[2] += x_vec_z
            elif y_magnitude <= x_magnitude and y_magnitude <= z_magnitude:
                cur_vox_y += y_sign
                cur_pos[0] += y_vec_x
                cur_pos[1] += y_vec_y
                cur_pos[2] += y_vec_z
            elif z_magnitude <= x_magnitude and z_magnitude <= y_magnitude:
                cur_vox_z += z_sign
                cur_pos[0] += z_vec_x
                cur_pos[1] += z_vec_y
                cur_pos[2] += z_vec_z

            i += 1
            if i == 1000:
                break

    def detect_peaks(self, axis: int, height: int = None, width: int = None) -> Tuple[int]:
        axis_values = [v[axis] for v in self.voxels]
        axis_occurences = Counter(axis_values)

        x = [0]*(max(axis_values)+1)
        for k in axis_occurences.keys():
            x[k] = axis_occurences[k]
        peaks, _ = find_peaks(np.array(x), height=height)

        return peaks

    def distance_field(self) -> np.array:
        hipf = {}
        for v in self.voxels:
            i = 1
            kernel = Kernel.circle(r=i)
            while len(self.get_kernel(v, kernel)) == len(kernel.voxels):
                i += 1
                kernel = Kernel.circle(r=i)
            hipf[v] = i

        return hipf

    def local_distance_field_maxima(self, radius) -> VoxelGrid:
        distance_field = self.distance_field()

        local_maxima = set()
        for vx, vx_dist in distance_field.items():
            vx_nbs = self.radius_search(self.voxel_centroid(vx), radius)
            vx_nbs_dist = [distance_field[nb] for nb in vx_nbs]

            if vx_dist >= max(vx_nbs_dist):
                local_maxima.add(vx)

        return self.subset(lambda v: v in local_maxima)

    def to_o3d(self, has_color=False):
        pcd = self.to_pcd(has_color).to_o3d()
        vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, self.cell_size[0])
        return vg

    def to_pcd(self, color=False) -> model.point_cloud.PointCloud:
        points = [self.voxel_centroid(v) for v in self.voxels]
        colors = list(self.list_attr(VoxelGrid.color_attr)) if color else []

        return model.point_cloud.PointCloud(np.array(points), colors=colors, source=self)

    def to_graph(self, kernel: Kernel = None) -> SpatialGraph:
        graph = networkx.Graph()
        for v in self.voxels:
            nbs = self.get_kernel(v, kernel)
            graph.add_node(v)

            for nb in nbs:
                graph.add_edge(*(v, nb))
            for attr in self.voxels[v]:
                graph.nodes[v][attr] = self.voxels[v][attr]
            graph.nodes[v]['pos'] = self.voxel_centroid(v)

        return SpatialGraph(self.cell_size, self.origin, graph)


class Kernel(VoxelGrid):
    def __init__(self, shape, origin=np.array([0, 0, 0]), voxels={}):
        self.shape = shape
        self.cell_size = np.array([1, 1, 1])
        self.origin = origin
        self.voxels = voxels
        self.size = len(voxels)

    @staticmethod
    def cylinder(d, h):
        r = d/2
        cylinder_voxels = set()
        for x, y, z in product(range(d), range(h), range(d)):
            dist = np.linalg.norm([x+0.5-r, z+0.5-r])
            if dist <= r:
                cylinder_voxels.add((x, y, z))

        return Kernel(shape=(d, h, d), voxels={v: {} for v in cylinder_voxels})

    @staticmethod
    def sphere(r):
        sphere_voxels = set()
        for x, y, z in product(range(-r, r+1), repeat=3):
            dist = np.linalg.norm([x, y, z])
            if dist <= r:
                sphere_voxels.add((x, y, z))

        return Kernel(shape=(r*2, r*2, r*2), voxels={v: None for v in sphere_voxels})

    @staticmethod
    def circle(r):
        circle_voxels = set()
        for x, z in product(range(-r, r+1), repeat=2):
            dist = np.linalg.norm([x, z])
            if dist <= r:
                circle_voxels.add((x, 0, z))

        return Kernel(shape=(r*2, 1, r*2), voxels={v: None for v in circle_voxels})

    @staticmethod
    def nb6():
        return Kernel(shape=(3, 3, 3), origin=np.array([1, 1, 1]),
                      voxels={(1, 2, 1): None,
                              (1, 1, 0): None, (0, 1, 1): None, (2, 1, 1): None, (1, 1, 2): None,
                              (1, 0, 1): None, })

    @staticmethod
    def nb4():
        return VoxelGrid(shape=(3, 3, 1), origin=np.array([1, 1, 1]), voxels={(1, 1, 0): None, (0, 1, 1): None, (2, 1, 1): None, (1, 1, 2): None})

    @staticmethod
    def stick_kernel(scale):
        a_height, a_width = int(15 // scale), 1 + int(5 // scale)
        b_height, b_width = int(5 // scale), 1

        kernel_a = Kernel.cylinder(a_width, a_height).translate(
            np.array([0, b_height, 0]))
        kernel_b = Kernel.cylinder(b_width, b_height).translate(
            np.array([a_width//2, 0, a_width//2]))
        
        kernel_c = kernel_a + kernel_b
        kernel_c.origin = np.array([a_width//2, 0, a_width//2])
        kernel_c = kernel_c.translate(np.array([0, 1, 0]))
        kernel_c = kernel_c.translate(-kernel_c.origin)

        return kernel_c
