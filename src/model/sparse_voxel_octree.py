from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Set
import numpy as np
from numba import jit

def get_nbs(voxel_index, connectivity):
    nbs_i = connectivity[voxel_index]
    nbs_i = nbs_i[nbs_i != -1]

    return nbs_i

@jit(nopython=True, parallel=True)
def part_64(x: int) -> int:
    """
    Largely based on https://stackoverflow.com/a/18528775
    """
    
    x &= 0x1fffff
    x = (x | x << 32) & 0x1f00000000ffff
    x = (x | x << 16) & 0x1f0000ff0000ff
    x = (x | x << 8) & 0x100f00f00f00f00f
    x = (x | x << 4) & 0x10c30c30c30c30c3
    x = (x | x << 2) & 0x1249249249249249

    return x

@jit(nopython=True, parallel=True)
def unpart_64(x: int) -> int:
    """
    Largely based on https://stackoverflow.com/a/18528775
    """
    
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

def decode_z_order_curve(z_order: List[int]):
    return np.array([decode_morton(z) for z in z_order])

# https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection
@jit(nopython=True)
def aabb_inside_aabb(a, b):
    return  (b[0][0] <= a[0][0] <= b[1][0] and b[1][0] >= a[1][0] >= b[0][0]) and \
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

    def __str__(self):
        return str(self.morton)
    
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
    
    def leaf_nodes(self):
        return self.root.leaf_nodes()

    @staticmethod
    def merge_octants(nodes: List[OctreeNode]) -> List[OctreeNode]:
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
    def from_voxels(voxels: np.array, half_width: float) -> SVO:
        if not voxels:
            raise ValueError("Can't create octree from 0 voxels.")
        if half_width == 0:
            raise ValueError("Voxel half width must be greater than 0.")
        
        z_order = z_order_curve(voxels)
        nodes = [OctreeNode(z, half_width, tuple([None]*8), leaf=True) for z in z_order]

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

    def range_search(self, aabb: np.array) -> np.array:
        range_morton = SVO.recursive_range_search(aabb, self.root, set())
        range_voxel = decode_z_order_curve(range_morton)
        
        return range_voxel

    def radius_search(self, p: np.array, r: float) -> np.array:
        radius_morton = SVO.recursive_radius_search(p, r, self.root, set())
        radius_voxel = decode_z_order_curve(radius_morton)
        
        return radius_voxel

    def get_depth(self, depth: int) -> np.array:
        if depth > self.max_depth():
            raise ValueError("Depth argument must be smaller than or equal to the max depth.")
            
        cur_level = [self.root]
        for _ in range(depth):
            # Get all children of all nodes in current level that are not none
            cur_level = [child for node in cur_level 
                         for child in node.children 
                         if child is not None]

        return np.array([decode_morton(z.morton) for z in cur_level])
    
    @staticmethod
    def max_centroid_in_radius_distance(cell_size):
        # Voxel centroids can be at most half the voxel's diagonal outside the radius
        return (cell_size**2 + (2*(cell_size**2)**(1/2))**(1/2))/2