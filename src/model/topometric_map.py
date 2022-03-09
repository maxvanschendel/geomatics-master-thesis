from __future__ import annotations

from copy import deepcopy
import math
import warnings
from ast import Lambda
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from itertools import product
from random import random
from typing import Dict, Tuple

import networkx
import numpy as np
import open3d as o3d
from misc.helpers import random_color, most_common
from numba import cuda
from model.sparse_voxel_octree import *
from model.voxel_grid import *

warnings.filterwarnings('ignore')

class Hierarchy(Enum):
    BUILDING = 0
    STOREY = 1
    ROOM = 2
    ISOVIST = 3


@dataclass(eq=True, frozen=True)
class TopometricNode:
    level: Hierarchy
    geometry: VoxelGrid


class EdgeType(Enum):
    TRAVERSABILITY = 0
    HIERARCHY = 1


class HierarchicalTopometricMap():
    def __init__(self):
        self.graph = networkx.DiGraph()

    def add_node(self, node: TopometricNode):
        self.graph.add_node(node, node_level=node.level)

    def add_nodes(self, nodes: List[TopometricNode]):
        for n in nodes:
            self.add_node(n)

    def add_edge(self, node_a: TopometricNode, node_b: TopometricNode, edge_type: EdgeType):
        self.graph.add_edge(node_a, node_b, edge_type=edge_type)
        
    def add_edges(self, edges, edge_type):
        for node_a, node_b in edges:
            self.add_edge(node_a, node_b, edge_type)

    def traversability_edges(self) -> List[Tuple[int, int]]:
        return self.get_edge_type(EdgeType.TRAVERSABILITY)

    def hierarchy_edges(self) -> List[Tuple[int, int]]:
        return self.get_edge_type(EdgeType.HIERARCHY)

    def get_edge_type(self, edge_type) -> List[Tuple[int, int]]:
        return [(u, v) for u, v, e in self.graph.edges(data=True) if e['edge_type'] == edge_type]

    def get_node_level(self, level):
        return [n for n, data in self.graph.nodes(data=True) if data['node_level'] == level]

    def to_o3d(self, level):
        nodes = self.get_node_level(level)
        nodes_geometry = [node.geometry for node in nodes]
        nodes_o3d = [geometry.to_o3d() for geometry in nodes_geometry]

        for n in nodes_o3d:
            n.paint_uniform_color(random_color())

        points, lines = [node.geometry.centroid() for node in nodes], []
        for i, n in enumerate(nodes):
            edges_traversability = [(u, v) for u, v, e in self.graph.edges(
                n, data=True) if e['edge_type'] == EdgeType.TRAVERSABILITY]

            for a, b in edges_traversability:
                lines.append((i, nodes.index(b)))

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        spheres = []
        for p in points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=0.2, resolution=10)
            sphere = sphere.translate(p)
            sphere = sphere.compute_triangle_normals()

            sphere.paint_uniform_color([0, 0, 1])
            spheres.append(sphere)

        return nodes_o3d, line_set, spheres

    def draw_graph(self, fn):
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_agraph import graphviz_layout

        plt.clf()

        hier_labels = {x: y['node_level'].name for x,
                       y in self.graph.nodes(data=True)}
        colors = ['r' if e['edge_type'] == EdgeType.HIERARCHY else 'b' for u,
                  v, e in self.graph.edges(data=True)]

        pos = graphviz_layout(self.graph, "twopi", args="-Grankdir=LR")
        networkx.draw(G=self.graph,
                      pos=pos,
                      edge_color=colors,
                      labels=hier_labels,
                      node_size=1)
        plt.savefig(fn)
