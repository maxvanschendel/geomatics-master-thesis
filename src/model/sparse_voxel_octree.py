from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Set
import numpy as np


from numba import jit


def get_nbs(voxel_index, connectivity):
    nbs_i = connectivity[voxel_index]
    nbs_i = nbs_i[nbs_i != -1]
    # nbs_i = nbs_i[np.where(nbs_i != -1)] # This option is 1.5x slower

    return nbs_i

# Largely based on https://stackoverflow.com/a/18528775


@jit(nopython=True, parallel=True)
def part_64(x: int) -> int:
    x &= 0x1fffff
    x = (x | x << 32) & 0x1f00000000ffff
    x = (x | x << 16) & 0x1f0000ff0000ff
    x = (x | x << 8) & 0x100f00f00f00f00f
    x = (x | x << 4) & 0x10c30c30c30c30c3
    x = (x | x << 2) & 0x1249249249249249

    return x


@jit(nopython=True, parallel=True)
def unpart_64(x: int) -> int:
    x &= 0x1249249249249249
    x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3
    x = (x ^ (x >> 4)) & 0x100f00f00f00f00f
    x = (x ^ (x >> 8)) & 0x1f0000ff0000ff
    x = (x ^ (x >> 16)) & 0x1f00000000ffff
    x = (x ^ (x >> 32)) & 0x1fffff

    return x


@jit(nopython=True, parallel=True)
def interleave(x: int, y: int, z: int) -> int:
    return x | (y << 1) | (z << 2)


@jit(nopython=True, parallel=True)
def decode_morton(x):
    return np.array([unpart_64(x), unpart_64(x >> 1), unpart_64(x >> 2)])


@jit(nopython=True, parallel=True)
def morton_code(p) -> int:
    return interleave(part_64(p[0]), part_64(p[1]), part_64(p[2]))


def z_order_curve(points: np.array) -> np.array:
    return sorted([morton_code(p) for p in points])

# https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection
@jit(nopython=True)
def aabb_inside_aabb(a, b):
    return (b[0][0] <= a[0][0] <= b[1][0] and b[1][0] >= a[1][0] >= b[0][0]) and \
        (b[0][1] <= a[0][1] <= b[1][1] and b[1][1] >= a[1][1] >= b[0][1]) and \
        (b[0][2] <= a[0][2] <= b[1][2] and b[1][2] >= a[1][2] >= b[0][2])

# https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection
@jit(nopython=True)
def aabb_intersects(a, b):
    return  (a[0][0] <= b[1][0] and a[1][0] >= b[0][0]) and \
            (a[0][1] <= b[1][1] and a[1][1] >= b[0][1]) and \
            (a[0][2] <= b[1][2] and a[1][2] >= b[0][2])

# https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection
@jit(nopython=True)
def aabb_sphere_intersects(p, r, aabb):
    # Clamp sphere center to closest point on box
    x = max(aabb[0][0], min(p[0], aabb[1][0]))
    y = max(aabb[0][1], min(p[1], aabb[1][1]))
    z = max(aabb[0][2], min(p[2], aabb[1][2]))

    # Check if closest point on box is within sphere
    dist = np.linalg.norm(p - np.array((x, y, z)))
    return dist < r

# https://geidav.wordpress.com/2014/08/18/advanced-octrees-2-node-representations/
@dataclass
class OctreeNode:
    morton: int                        # Morton code of node center
    half_width: float
    children: Tuple[OctreeNode]
    leaf: bool = False

    def __post_init__(self):
        self.center = self.center()
        self.aabb = np.vstack([self.center - self.half_width, self.center + self.half_width])

    def center(self):
        return (decode_morton(self.morton) * (self.half_width*2)) + self.half_width

    def occupied_children(self):
        return [c for c in self.children if c is not None]

    def octant(self, p):
        oct = 0
        center = self.center()

        if p[0] >= center[0]:
            oct |= 4
        if p[1] >= center[1]:
            oct |= 2
        if p[2] >= center[2]:
            oct |= 1
        return oct
    
    def depth(self):
        cur = self
        i = 0
        while not cur.leaf:
            cur = cur.occupied_children()[0]
            i+=1
            
        return i

    def get_children_recursive(n: OctreeNode, leaf: Set[int]):
        if n.leaf:
            return leaf.add(n.morton)
        else:
            for child in n.occupied_children():
                OctreeNode.get_children_recursive(child, leaf)
        return leaf

    def leaf_nodes(self):
        return OctreeNode.get_children_recursive(self, set())


@dataclass
class SVO:
    root: OctreeNode
    nodes: List[OctreeNode]
    
    def max_depth(self):
        return self.root.depth()

    @staticmethod
    def merge_octants(nodes: List[OctreeNode]):
        parent_morton = nodes[0].morton // 8
        parent_half_width = nodes[0].half_width * 2
        parent_nodes = []

        accumulator = [None] * 8
        for z in nodes:
            # Morton code of parent is morton code of child bit but with first 3 bits removed (//8)
            new_parent_morton = z.morton >> 3
            if parent_morton != new_parent_morton:
                parent = OctreeNode(
                    parent_morton, parent_half_width, tuple(accumulator))
                parent_nodes.append(parent)

                parent_morton = new_parent_morton
                accumulator = [None] * 8
            accumulator[z.morton % 8] = z

        if accumulator:
            parent = OctreeNode(
                parent_morton, z.half_width * 2, tuple(accumulator))
            parent_nodes.append(parent)

        return parent_nodes

    @staticmethod
    def from_voxels(voxels, half_width):
        if not voxels:
            raise ValueError("Can't create octree from 0 voxels.")
        if half_width == 0:
            raise ValueError("Voxel half width must be greater than 0.")
        
        z_order = z_order_curve(voxels)
        nodes = [OctreeNode(z, half_width, tuple([None]*8), leaf=True)
                 for z in z_order]

        parent_nodes = SVO.merge_octants(nodes)
        while len(parent_nodes) > 1:
            parent_nodes = SVO.merge_octants(parent_nodes)
            nodes += parent_nodes

        octree = SVO(root=parent_nodes[0], nodes=nodes)
        return octree

    @staticmethod
    def recursive_range_search(aabb, o, voxels):
        if o.leaf:
            return voxels.add(o.morton)

        for child in o.occupied_children():
            if aabb_intersects(child.aabb, aabb):
                SVO.recursive_range_search(aabb, child, voxels)
            elif aabb_inside_aabb(child.aabb, aabb):
                leaf_nodes = child.leaf_nodes()
                voxels.update(leaf_nodes)
                    

        return voxels

    @staticmethod
    def recursive_radius_search(p, r, o, voxels):
        if o.leaf:
            return voxels.add(o.morton)

        for child in o.occupied_children():
            if aabb_sphere_intersects(p, r, child.aabb):
                SVO.recursive_radius_search(p, r, child, voxels)

        return voxels

    def range_search(self, aabb):
        return SVO.recursive_range_search(aabb, self.root, set())

    def radius_search(self, p, r):
        return SVO.recursive_radius_search(p, r, self.root, set())

    def get_depth(self, depth: int) -> List[Tuple[int, int, int]]:
        current_level = [self.root]
        for _ in range(0, depth):
            current_level = [
                n_c for node in current_level for n_c in node.children if n_c is not None]
            if not current_level:
                raise ValueError(
                    "Depth argument must be less deep than leaf depth")

        return {z.morton: i for i, z in enumerate(current_level)}


def benchmark():
    from time import perf_counter
    from sys import getsizeof
    from itertools import product
    from model.topometric_map import PointCloud

    cell_size = 0.05
    depth = 9

    # Prepare input data for benchmark
    map_cloud = PointCloud.read_ply(
        "./data/meshes/diningroom2kitchen - low.ply")
    leaf_voxels = map_cloud.voxelize(cell_size)

    print("Starting benchmark...")

    # Construct a sparse voxel octree from leaf voxels
    construct_time_start = perf_counter()
    octree = SVO.from_voxels(leaf_voxels.voxels, cell_size / 2)
    print(
        f"Constructed sparse voxel octree with {len(octree.nodes)} nodes in {(perf_counter() - construct_time_start):.4f} seconds, taking up {(getsizeof(octree.nodes)/(10**6)):.4f} MB of memory")

    # Get all nodes at a certain depth in the tree
    retrieve_depth_start = perf_counter()
    voxels = octree.get_depth(depth)
    voxel_indices = np.array(list(voxels.keys()))
    print(
        f"Retrieved {len(voxels)} voxels at depth {depth} in {(perf_counter() - retrieve_depth_start):.4f} seconds")

    # Perform a range search
    range_search_start = perf_counter()
    aabb = np.array([(-12.8, -12.8, -12.8), (12.8, 12.8, 12.8)])
    range_search = octree.range_search(aabb)
    print(
        f"Performed range search with {len(range_search)} resulting voxels in {(perf_counter() - range_search_start):.4f} seconds")

    # Perform a radius search
    range_search_start = perf_counter()
    p, r = np.array([0, 0, 0]), 10
    range_search = octree.radius_search(p, r)
    print(
        f"Performed radius search with {len(range_search)} resulting voxels in {(perf_counter() - range_search_start):.4f} seconds")

    # Perform a large number of inclusion queries
    inclusion_start = perf_counter()
    inclusions = [i for i in range(max(voxels)) if i in voxels]
    print(
        f"Did {max(voxels)} inclusion tests with {len(inclusions)} positives in {(perf_counter() - inclusion_start):.4f} seconds")

    # prepare 6-neighbourhood and 26-neighbourhood kernels
    nb26 = list(product(range(-1, 2), range(-1, 2), range(-1, 2)))
    nb26.remove((0, 0, 0))
    nb26 = np.array(nb26)

    v_26nbs = [decode_morton(v) + nb26 for v in voxels]
    nbs_morton = [morton_code(v) for v_nb in v_26nbs for v in v_nb]
    connectivity26 = np.full(shape=(len(voxels), 26),
                             fill_value=-1, dtype=np.int32)

    # Build 26-neighbourhood graph
    connectivity_start = perf_counter()
    for i, nbs in enumerate(nbs_morton):
        if nbs in voxels:
            connectivity26[i//26][i % 26] = voxels[nbs]
    print(
        f"Built 26-neighbourhood connectivity in {(perf_counter()- connectivity_start):.4f} seconds, taking up {getsizeof(connectivity26)/(10**6)} MB of memory")

    # Get 26-neighbourhood of every voxel in grid
    nb26_retrieval_start = perf_counter()
    nbs_i = [get_nbs(voxels[v], connectivity26) for v in voxels]
    nbs = [voxel_indices[i] for i in nbs_i]
    print(
        f"Retrieved 26-neighbours in {(perf_counter() - nb26_retrieval_start):.4f} seconds")


if __name__ == "__main__":
    max_21_bit = 2097152
    benchmark()

    inp = (5, 3, 1)
    mc = morton_code(inp)
    dmc = decode_morton(mc)
