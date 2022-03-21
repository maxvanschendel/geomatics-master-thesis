from __future__ import annotations

import networkx
import numpy as np
import open3d as o3d

from model.sparse_voxel_octree import *
import model.voxel_grid


class SpatialGraph:
    def __init__(self,
                 scale: np.array = np.array([1, 1, 1]),
                 origin: np.array = np.array([0, 0, 0]),
                 graph: networkx.Graph = networkx.Graph()):

        self.scale = scale
        self.origin = origin
        self.graph = graph

        self.nodes = self.graph.nodes
        self.edges = self.graph.edges

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

    def to_voxel(self):
        vox = model.voxel_grid.VoxelGrid(np.array([0, 0, 0]), self.scale, self.origin, {node: {} for node in self.nodes})
        vox.shape = np.array(vox.extents())

        # transfer voxel attributes to nodes
        for node in self.nodes:
            for attr in self.nodes[node].keys():
                vox[node][attr] = self.nodes[node][attr]

        return vox

    def connected_components(self):
        return sorted(networkx.connected_components(self.graph), key=len, reverse=True)