from __future__ import annotations
import logging
from numba import cuda

import math
import os
from ast import Lambda
from copy import deepcopy
from itertools import product
from typing import Callable, Dict, Iterable, Iterator, Tuple
from warnings import filterwarnings

import networkx
import numpy as np

from utils.array import all_same, most_common

import model.point_cloud
from model.sparse_voxel_octree import *
from model.spatial_graph import *
from utils.linalg import normalize

enable_debug = False
os.environ["NUMBA_ENABLE_CUDASIM"] = str(int(enable_debug))
os.environ["NUMBA_CUDA_DEBUGINFO"] = str(int(enable_debug))
filterwarnings('ignore')


class VoxelGrid:
    cluster_attr: str = 'cluster'
    color_attr: str = 'color'
    ground_truth_attr: str = 'room'
    nav_attr: str = 'navigable'

    def __init__(self,
                 cell_size: float = 1,
                 origin: np.array = np.array([0, 0, 0]),
                 voxels: Dict[Tuple[int, int, int], Dict[str, object]] = None):

        self.cell_size: float = cell_size
        self.origin: np.array = origin
        
        # Store voxel indices (x, y, z) as key in dictionary, with attributes as {'attr_name': attr_value} values
        self.voxels: Dict[Tuple[int, int, int], Dict[str, object]] = voxels if voxels else {}

        # Set size of voxel grid to 0 if it contains no voxels (is empty)
        self.size = len(voxels) if self.voxels else 0

        # Generate sparse voxel octree if voxel grid is not empty
        self.svo = self.generate_svo() if self.voxels else None

        # Easy lookup from Morton code to voxel index
        self.morton_index = {morton_code(
            v): i for i, v in enumerate(self.voxels)}

    def clone(self) -> VoxelGrid:
        return deepcopy(self)

    def generate_svo(self) -> SVO:
        return SVO.from_voxels(self.voxels.keys(), self.cell_size/2)

    @staticmethod
    def merge(voxel_grids: List[VoxelGrid]) -> VoxelGrid:
        """
        Takes a list of voxels and merges them into a single voxel grid containing all voxels
        """

        if len(voxel_grids) == 0:
            raise ValueError("Can't merge 0 voxel grids.")

        # check if voxel grids can be merged, voxel grids with differing voxel sizes or origins cannot be merged
        cell_sizes = [vg.cell_size for vg in voxel_grids]
        origins = [tuple(vg.origin) for vg in voxel_grids]

        # sum up the voxels of each voxel grid
        voxels = {}
        for vg in voxel_grids:
            voxels.update(vg.voxels)
        voxel_grid = VoxelGrid(cell_sizes[0], origins[0], voxels)

        return voxel_grid

    def level_of_detail(self, level: int) -> VoxelGrid:
        """
        Generate level of detail using octree, where level 0 represents the leaf-level voxel grid 
        and each level above represents a non-leaf or root level in the tree. 
        The maximum level is equal to the octree's depth.
        """

        lod_voxels = self.svo.get_depth(self.svo.max_depth() - level)
        lod_voxel_grid = VoxelGrid(
            self.cell_size*(2**level), self.origin, {tuple(v): {} for v in lod_voxels})

        # TODO: clean this part
        attributes = self.attributes()
        lod_voxel_attributes = {
            voxel: {attr: [] for attr in attributes} for voxel in lod_voxel_grid.voxels}
        for v in self.voxels:
            parent_voxel = lod_voxel_grid.get_voxel(self.voxel_centroid(v))
            for attr, value in self.voxels[v].items():
                lod_voxel_attributes[parent_voxel][attr].append(value)

        for voxel, attributes in lod_voxel_attributes.items():
            for attr, values in attributes.items():
                if len(values):
                    try:
                        lod_voxel_grid[voxel][attr] = most_common(values)
                    except:
                        pass

        return lod_voxel_grid

    def __getitem__(self, key: Tuple[int, int, int]) -> Dict[str, object]:
        return self.voxels[key]

    def __setitem__(self, key: Tuple[int, int, int], value: Dict[str, object]) -> None:
        self.voxels[key] = value

    def __add__(self, other: VoxelGrid) -> VoxelGrid:
        new_model = self.clone()
        for v in other.voxels:
            new_model.voxels[v] = other.voxels[v]

        # Adapt size of voxel model to new extents after addition
        new_model.size = len(new_model.voxels)
        return new_model

    def __contains__(self, val: Tuple[int, int, int]) -> bool:
        return val in self.voxels

    def get_voxels(self) -> Set[Tuple[int, int, int]]:
        return set(self.voxels.keys())

    def map(self, func: Callable, **kwargs) -> Iterator:
        return map(lambda v: func(v, **kwargs), self.voxels)

    def filter(self, func: Callable, **kwargs) -> Iterator:
        return filter(lambda v: func(v, **kwargs), self.voxels)

    def get_voxel(self, p: np.array) -> Tuple[int, int, int]:
        return tuple(((p-self.origin) // self.cell_size).astype(int))

    def contains_point(self, point: np.array) -> bool:
        return self.get_voxel(point) in self

    def transform(self, transformation: np.array) -> VoxelGrid:
        pcd = self.to_pcd()
        pcd_t = pcd.transform(transformation)
        voxel_grid_t = pcd_t.voxelize(self.cell_size)

        return voxel_grid_t

    def intersect(self, other: VoxelGrid) -> Set[Tuple[int, int, int]]:
        return self.voxels.keys() & other.voxels.keys()

    def union(self, other: VoxelGrid) -> Set[Tuple[int, int, int]]:
        return self.voxels.keys() | other.voxels.keys()

    def subset(self, func: Lambda, **kwargs) -> VoxelGrid:
        filtered_voxels = self.filter(func, **kwargs)
        voxel_dict = {voxel: self.voxels[voxel] for voxel in filtered_voxels}

        return VoxelGrid(deepcopy(self.cell_size), deepcopy(self.origin), voxel_dict)

    def voxel_subset(self, voxels: Iterable[Tuple[int, int, int]]) -> VoxelGrid:
        return VoxelGrid(deepcopy(self.cell_size),
                         deepcopy(self.origin),
                         {v: self.voxels[v] for v in voxels})

    def for_each(self, func: Lambda, **kwargs) -> None:
        for voxel in self.voxels:
            func(voxel, **kwargs)

    def set_attr(self, voxel: Tuple[int, int, int], attr: str, val: object) -> None:
        self.voxels[voxel][attr] = val

    def set_attr_uniform(self, attr: str, val: object) -> None:
        self.for_each(self.set_attr, attr=attr, val=val)

    def colorize(self, color: np.array) -> None:
        self.set_attr_uniform(attr=VoxelGrid.color_attr, val=color)

    def get_attr(self, attr: str, val: object) -> Iterator:
        return self.filter(lambda vox: attr in self[vox] and self[vox][attr] == val)

    def list_attr(self, attr: str) -> Iterator:
        voxels_with_attr = self.filter(lambda v: attr in self.voxels[v])
        voxel_attributes = map(
            lambda v: self.voxels[v][attr], voxels_with_attr)

        return voxel_attributes

    def unique_attr(self, attr: str) -> np.array:
        attributes = list(self.list_attr(attr))
        unique_attributes = np.unique(attributes)

        return unique_attributes

    def kernel_attr(self, voxel: Tuple[int, int, int], kernel: VoxelGrid, attr: str) -> List:
        voxel_nbs = self.get_kernel(voxel, kernel)
        voxel_nbs_labels = [self[v][attr] for v in voxel_nbs]

        return voxel_nbs_labels

    def most_common_kernel_attr(self, voxel: Tuple[int, int, int], kernel: VoxelGrid, attr: str) -> np.array:
        kernel_attrs = self.kernel_attr(voxel, kernel, attr)
        most_common_attr = most_common(kernel_attrs)

        return most_common_attr

    def attributes(self) -> np.array:
        from utils.array import flatten_list

        voxel_attributes = [list(self.voxels[v].keys()) for v in self.voxels]
        return np.unique(flatten_list(voxel_attributes))

    def clear_attributes(self) -> None:
        for voxel in self.voxels:
            self.voxels[voxel] = {}

    def to_2d_array(self) -> np.array:
        arr = np.zeros((self.size, 3), dtype=np.int32)
        for i, k in enumerate(self.voxels):
            arr[i] = k
        return arr

    def to_3d_array(self, attr=None) -> np.array:
        voxel_matrix = np.full(self.extents().astype(int)+1, 0, dtype=np.int8)
        for v in self.voxels:
            voxel_matrix[v] = self[v][attr] if attr else 1
        return voxel_matrix

    def propagate_attr(self, attr: str, prop_kernel: VoxelGrid, max_its: int = 1000) -> VoxelGrid:
        prop_map = self.clone()
        unique_attr = self.unique_attr(attr)

        kernel_voxels = cuda.to_device(prop_kernel.to_2d_array())
        origin_voxels = cuda.to_device(self.to_2d_array())

        # Get most common label of 6-neighbourhood of voxel
        # Assign most common label to voxel in propagated map
        for i in range(max_its):
            prev_map = prop_map.clone()

            # Allocate memory on the device for the result
            max_nb = cuda.to_device(np.zeros((self.size, 1), dtype=np.int32))
            nbs_occurences = cuda.to_device(
                np.zeros((self.size, len(unique_attr)), dtype=np.int32))

            # Map unique attributes to integers in range 0 to number of unique attributes
            # Also store inverse for easy lookup
            attr_to_i = {u_attr: i for (i, u_attr) in enumerate(unique_attr)}
            i_to_attr = {value: key for key, value in attr_to_i.items()}

            # Create dense matrix where each cell contains the mapped attribute
            voxel_matrix = np.full(prev_map.extents() + 1, -1, dtype=np.int32)
            for v in self.voxels:
                voxel_matrix[v] = attr_to_i[int(prev_map.voxels[v][attr])]

            # For each cell, find the most common neighbour
            threadsperblock = 256
            blockspergrid = (self.size + (threadsperblock - 1)
                             ) // threadsperblock
            VoxelGrid.most_common_nb_gpu[blockspergrid, threadsperblock](
                voxel_matrix, origin_voxels, kernel_voxels, nbs_occurences, len(unique_attr), max_nb)
            cuda.synchronize()

            # Map integer attributes back to original attributes
            for i, max_i in enumerate(max_nb):
                prop_map[tuple(origin_voxels[i])][attr] = i_to_attr[max_i[0]]

            # Stop propagation once it fails to make a change
            if prev_map.voxels == prop_map.voxels:
                break

        return prop_map

    def attr_borders(self, attr: str, kernel: Kernel) -> VoxelGrid:
        border_voxels = set()
        for voxel in self.voxels:
            voxel_nbs_labels = self.kernel_attr(voxel, kernel, attr)

            # Check if neighbourhood all has the same attribute value
            if voxel_nbs_labels != [self[voxel][attr]]*len(voxel_nbs_labels):
                border_voxels.add(voxel)

        # Return a new voxel grid containing only the voxels that lie on the border
        # between two attributes
        border_voxel_grid = self.subset(lambda v: v in border_voxels)
        return border_voxel_grid

    def split_by_attr(self, attr: str, get_attr: bool = False) -> List[VoxelGrid]:
        split = []
        for unique_attr in self.unique_attr(attr):
            attr_subset = self.subset(
                lambda v: attr in self[v] and self[v][attr] == unique_attr)

            split.append(
                (attr_subset, unique_attr) if get_attr else attr_subset)

        return split

    def extents(self) -> np.array:
        if len(self.voxels) == 0:
            raise ValueError("No voxels in voxel grid")
        return np.max(self.to_2d_array(), axis=0)

    def voxel_centroid(self, voxel: Tuple[int, int, int]) -> np.array:
        return self.origin + (np.array(voxel) * self.cell_size) + (0.5 * self.cell_size)

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
        """ Returns all voxels within a sphere surrounding a point. Voxels centroids may
            lie outside the sphere by a maximum of half times the diagonal of a voxel.

        Args:
            p (np.array): sphere center global coordinates
            r (float): search radius

        Returns:
            List[Tuple[int, int, int]]: voxels within sphere
        """

        radius_morton = self.svo.radius_search(p - self.origin, r)
        return [tuple(v) for v in radius_morton]

    def range_search(self, aabb: np.array) -> List[Tuple[int, int, int]]:
        """Returns all voxels within an axis-aligned bounding box.

        Args:
            aabb (np.array): bounding box 2x3 matrix in global coordinates

        Returns:
            List[Tuple[int, int, int]]: voxels within bounding box
        """

        range_morton = self.svo.range_search(aabb - self.origin)
        return [tuple(v) for v in range_morton]

    def connected_components(self, kernel: Kernel) -> List[VoxelGrid]:
        """ Splits voxel grid into one or more connected components. For each connected
            component there is a path between every voxel within, moving from adjacent
            voxel to adjacent voxel. 

        Args:
            kernel (_type_): kernel defining voxel adjacency

        Returns:
            List[VoxelGrid]: connected components
        """

        graph_representation = self.to_graph(kernel)
        components = graph_representation.connected_components()
        components_voxel = [self.subset(lambda v: v in c) for c in components]

        return components_voxel

    @cuda.jit
    def has_nb_gpu(voxels, occupied_voxels, kernel, conv_voxels):
        pos = cuda.grid(1)
        if pos < occupied_voxels.shape[0]:
            cur_vox = occupied_voxels[pos]

            for nb in kernel:
                nb_x = cur_vox[0]+nb[0]
                nb_y = cur_vox[1]+nb[1]
                nb_z = cur_vox[2]+nb[2]

                if 0 < nb_x < voxels.shape[0] and \
                        0 < nb_y < voxels.shape[1] and \
                        0 < nb_z < voxels.shape[2] and \
                        voxels[nb_x, nb_y, nb_z] == 1:

                    conv_voxels[pos] = 1
                    break

    @cuda.jit
    def most_common_nb_gpu(voxels, occupied_voxels, kernel, attribute_count, n_vals, max_occurences):
        pos = cuda.grid(1)

        if pos < occupied_voxels.shape[0]:
            current_voxel = occupied_voxels[pos]

            for nb in kernel:
                nb_x = current_voxel[0] + nb[0]
                nb_y = current_voxel[1] + nb[1]
                nb_z = current_voxel[2] + nb[2]

                # Check that neighbour index is not out of bounds
                if 0 <= nb_x < voxels.shape[0] and \
                        0 <= nb_y < voxels.shape[1] and \
                        0 <= nb_z < voxels.shape[2]:

                    # Get attribute of neighbour
                    nb_attr = voxels[nb_x, nb_y, nb_z]

                    # Increment the counter for the voxel's attribute unless its value is -1
                    # -1 is reserved for null attribute
                    if nb_attr != -1:
                        attribute_count[pos][nb_attr] += 1

            # The most common neighbour attribute is assumed to be the current
            # voxel's attribute first. Another attribute needs to occur more than once
            # to be the most common neighbour attribute.
            occurences = voxels[current_voxel[0],
                                current_voxel[1], current_voxel[2]]
            most_common_attribute = 1

            for val in range(n_vals):
                if attribute_count[pos][val] > most_common_attribute:
                    most_common_attribute = attribute_count[pos][val]
                    occurences = val

            max_occurences[pos] = occurences

    def jaccard_index(self, other: VoxelGrid) -> float:
        intersect = self.intersect(other)
        union = self.union(other)

        intersect_size = len(intersect)
        union_size = len(union)

        return intersect_size / union_size if union_size else 0

    def symmetric_overlap(self, other):
        if len(self.voxels) and len(other.voxels):
            return len(self.intersect(other)) / min([len(other.voxels), len(self.voxels)])
        else:
            return 0

    def mutate(self, voxels: Dict) -> VoxelGrid:
        return VoxelGrid(deepcopy(self.cell_size),
                         deepcopy(self.origin),
                         voxels)

    def filter_gpu_kernel_nbs(self, kernel: Kernel) -> VoxelGrid:
        # Allocate memory on the device for the result
        result_voxels = np.zeros(shape=(self.size, 1), dtype=np.int32)

        voxel_matrix = cuda.to_device(self.to_3d_array())
        occupied_voxels = cuda.to_device(self.to_2d_array())
        kernel_voxels = cuda.to_device(kernel.to_2d_array())

        threadsperblock = 1024
        blockspergrid = (self.size + (threadsperblock - 1)) // threadsperblock
        VoxelGrid.has_nb_gpu[blockspergrid, threadsperblock](
            voxel_matrix, occupied_voxels, kernel_voxels, result_voxels)

        nb_voxels = [tuple(occupied_voxels[v])
                     for (v, occ) in enumerate(result_voxels) if not occ[0]]
        return self.mutate({v: self.voxels[v] for v in nb_voxels})

    def dilate(self, kernel: Kernel) -> VoxelGrid:
        dilated_voxels = set(self.voxels.keys())
        for v in self.voxels:
            kernel_voxels = {tuple(v + (k - kernel.origin))
                             for k in kernel.voxels}
            dilated_voxels.update(kernel_voxels)

        return self.mutate({v: {} for v in dilated_voxels})

    def translate(self, translation: np.array) -> VoxelGrid:
        translated_cells = {}
        for voxel in self.voxels:
            voxel_t = translation + voxel
            translated_cells[tuple(voxel_t)] = self.voxels[voxel]

        return self.mutate(translated_cells)

    def visibility(self, origin: np.array, max_dist: float) -> VoxelGrid:
        # Get all voxels within range (max_dist) of the isovist
        # For each voxel within range, get a vector pointing from origin to the voxel
        radius_voxels = set(self.radius_search(origin, max_dist))
        directions = np.array(
            [normalize(self.voxel_centroid(v) - origin) for v in radius_voxels])

        if len(directions) == 0:
            logging.warning(f"No voxels within radius {max_dist}")
            return self.mutate({})

        # This matrix is used by the GPU to quickly find occupied voxels at the cost of memory
        voxel_matrix = self.voxel_subset(radius_voxels).to_3d_array()

        # Allocate matrix for raycasting results
        intersections = np.zeros(shape=(len(directions), 3), dtype=np.int32)

        # Perform fast voxel traversal on GPU
        # This is done in the coordinate system of the voxel grid, so ray origin must
        # be transformed first.
        threadsperblock = 512
        blockspergrid = (len(directions) +
                         (threadsperblock - 1)) // threadsperblock
        VoxelGrid.fast_voxel_traversal[blockspergrid, threadsperblock](
            origin - self.origin, directions, self.cell_size, voxel_matrix, intersections)

        visible_voxels = set([tuple(i) for i in intersections])
        visible_voxel_grid = self.subset(
            lambda v: v in visible_voxels and v != (0, 0, 0))
        return visible_voxel_grid

    @cuda.jit
    def fast_voxel_traversal(point, directions, cell_size, voxels, intersections):
        """
        Fast voxel traversal (Amanitides & Woo, 1987)
        Terminates when first voxel has been found.

        Based on: https://github.com/francisengelmann/fast_voxel_traversal
        """

        pos = cuda.grid(1)
        if pos < directions.shape[0]:
            direction = directions[pos]

            x_size = voxels.shape[0]
            y_size = voxels.shape[1]
            z_size = voxels.shape[2]

            # line equation
            x1, y1, z1, l, m, n = point[0], point[1], point[2], direction[0], direction[1], direction[2]

            l = l if l else 1/math.inf
            m = m if m else 1/math.inf
            n = n if n else 1/math.inf

            # is line moving in positive or negative direction for each axis
            step_x = int(l / abs(l))
            step_y = int(m / abs(m))
            step_z = int(n / abs(n))

            # get starting voxel
            cur_vox_x = int(math.floor((x1) / cell_size))
            cur_vox_y = int(math.floor((y1) / cell_size))
            cur_vox_z = int(math.floor((z1) / cell_size))

            next_voxel_boundary_x = (cur_vox_x + step_x) * cell_size
            next_voxel_boundary_y = (cur_vox_y + step_y) * cell_size
            next_voxel_boundary_z = (cur_vox_z + step_z) * cell_size

            t_max_x = (next_voxel_boundary_x - x1)/l
            t_max_y = (next_voxel_boundary_y - y1)/m
            t_max_z = (next_voxel_boundary_z - z1)/n

            t_delta_x = cell_size/l*step_x
            t_delta_y = cell_size/m*step_y
            t_delta_z = cell_size/n*step_z

            while x_size > cur_vox_x >= 0 and \
                    y_size > cur_vox_y >= 0 and \
                    z_size > cur_vox_z >= 0:

                if voxels[cur_vox_x, cur_vox_y, cur_vox_z] == 1:
                    intersections[pos][0] += cur_vox_x
                    intersections[pos][1] += cur_vox_y
                    intersections[pos][2] += cur_vox_z
                    break

                if t_max_x < t_max_y:
                    if t_max_x < t_max_z:
                        cur_vox_x += step_x
                        t_max_x += t_delta_x
                    else:
                        cur_vox_z += step_z
                        t_max_z += t_delta_z
                else:
                    if t_max_y < t_max_z:
                        cur_vox_y += step_y
                        t_max_y += t_delta_y
                    else:
                        cur_vox_z += step_z
                        t_max_z += t_delta_z

    def distance_field(self) -> np.array:
        df = {}
        for v in self.voxels:
            i = 1
            kernel = Kernel.circle(r=i)
            while len(self.get_kernel(v, kernel)) == len(kernel.voxels):
                i += 1
                kernel = Kernel.circle(r=i)
            df[v] = i

        return df

    def to_o3d(self, has_color: bool = False) -> o3d.geometry.VoxelGrid:
        pcd = self.to_pcd(has_color).to_o3d()
        vg = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, self.cell_size)
        return vg

    def to_pcd(self, has_color: bool = False) -> model.point_cloud.PointCloud:
        points = [self.voxel_centroid(v) for v in self.voxels]
        attributes = {attr: np.array(list(self.list_attr(attr)))
                      for attr in self.attributes()}
        colors = list(self.list_attr(VoxelGrid.color_attr)
                      ) if has_color else []

        return model.point_cloud.PointCloud(np.array(points),
                                            colors=colors,
                                            attributes=attributes)

    def to_graph(self, kernel: Kernel = None) -> SpatialGraph:
        from model.spatial_graph import SpatialGraph

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

    def write(self, fn: str) -> None:
        import pickle as pickle
        with open(fn, 'wb') as write_file:
            pickle.dump(self, write_file)

    @staticmethod
    def read(fn: str) -> VoxelGrid:
        import pickle as pickle
        with open(fn, 'rb') as read_file:
            vg = pickle.load(read_file)
        return vg

    def shape_dna(self, kernel, k):
        laplacian = self.laplacian_matrix(kernel)
        laplacian_eigenvalues = np.linalg.eigvals(laplacian / np.max(laplacian))
        sorted_eigenvalues = np.real(np.sort(laplacian_eigenvalues)[::-1])

        if len(sorted_eigenvalues) > k:
            first_k_eigenvalues = sorted_eigenvalues[:k]
        else:
            first_k_eigenvalues = np.hstack(
                (sorted_eigenvalues[:k], np.zeros(k - len(sorted_eigenvalues))))

        # Fit line through eigenvalues (y = ax + b)
        a, b = np.polyfit(np.arange(len(sorted_eigenvalues)),
                          sorted_eigenvalues, 1)
        return first_k_eigenvalues / a

    def laplacian_matrix(self, kernel):
        return self.degree_matrix(kernel) - self.adjacency_matrix(kernel)

    def degree_matrix(self, kernel):
        degree_matrix = np.zeros((self.size, self.size))

        for v in self.voxels:
            v_index = self.morton_index[morton_code(v)]
            v_nbs = self.get_kernel(v, kernel)

            degree_matrix[v_index][v_index] = len(v_nbs)

        return degree_matrix

    def adjacency_matrix(self, kernel):
        adjacency_matrix = np.zeros((self.size, self.size))

        for v in self.voxels:
            v_index = self.morton_index[morton_code(v)]
            v_nbs = self.get_kernel(v, kernel)
            nbs_index = [self.morton_index[morton_code(v)] for v in v_nbs]

            adjacency_matrix[v_index][nbs_index] = 1

        return adjacency_matrix


class Kernel(VoxelGrid):
    def __init__(self, origin=np.array([0, 0, 0]), voxels={}):
        self.cell_size = 1.
        self.origin = origin
        self.voxels = voxels
        self.size = len(voxels)

    @staticmethod
    def cylinder(d: float, h: float) -> Kernel:
        r = d/2
        cylinder_voxels = set()
        for x, y, z in product(range(d), range(h), range(d)):

            dist = np.linalg.norm([x+0.5-r, z+0.5-r])
            if dist <= r:
                cylinder_voxels.add((x, y, z))

        return Kernel(voxels={v: {} for v in cylinder_voxels})

    @staticmethod
    def sphere(r: float) -> Kernel:
        sphere_voxels = set()
        for x, y, z in product(range(-r, r+1), repeat=3):
            dist = np.linalg.norm([x, y, z])
            if dist <= r:
                sphere_voxels.add((x, y, z))

        return Kernel(voxels={v: None for v in sphere_voxels})

    @staticmethod
    def circle(r: float) -> Kernel:
        circle_voxels = set()
        for x, z in product(range(-r, r+1), repeat=2):
            dist = np.linalg.norm([x, z])
            if dist <= r:
                circle_voxels.add((x, 0, z))

        return Kernel(voxels={v: None for v in circle_voxels})

    @staticmethod
    def nb6() -> Kernel:
        return Kernel(origin=np.array([1, 1, 1]),
                      voxels={(1, 2, 1): None,
                              (1, 1, 0): None, (0, 1, 1): None, (2, 1, 1): None, (1, 1, 2): None,
                              (1, 0, 1): None, })

    @staticmethod
    def nb26() -> Kernel:
        return Kernel(origin=np.array([1, 1, 1]),
                      voxels={v: None for v in product(range(-1, 2), range(-1, 2), range(-1, 2))})

    @staticmethod
    def nb4() -> Kernel:
        return Kernel(origin=np.array([1, 1, 1]), voxels={(1, 1, 0): None, (0, 1, 1): None, (2, 1, 1): None, (1, 1, 2): None})

    @staticmethod
    def stick_kernel(scale: float) -> Kernel:
        a_height, a_width = int(15 // scale), 1 + int(5 // scale)
        b_height, b_width = int(5 // scale), 1

        kernel_a = Kernel.cylinder(a_width, a_height).translate(
            np.array([0, b_height, 0]))
        kernel_b = Kernel.cylinder(b_width, b_height).translate(
            np.array([a_width//2, 0, a_width//2]))

        kernel_c = kernel_a + kernel_b
        kernel_c.origin = np.array([a_width//2, 0, a_width//2])
        kernel_c = kernel_c.translate(np.array([0, 1, 0]))
        kernel_c = kernel_c.translate(kernel_c.origin*-1)

        return kernel_c
