from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


import networkx
import open3d as o3d
from utils.visualization import random_color
from model.sparse_voxel_octree import *
from model.voxel_grid import *


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


class TopometricMap():
    def __init__(self):
        self.graph = networkx.DiGraph()

    def nodes(self):
        return self.graph.nodes()

    def add_node(self, node: TopometricNode):
        self.graph.add_node(node, node_level=node.level)

    def add_nodes(self, nodes: List[TopometricNode]):
        for n in nodes:
            self.add_node(n)

    def add_edge(self, node_a: TopometricNode, node_b: TopometricNode, 
                 edge_type: EdgeType = EdgeType.TRAVERSABILITY, directed=True):
        self.graph.add_edge(node_a, node_b, edge_type=edge_type)
        if not directed:
            self.graph.add_edge(node_b, node_a, edge_type=edge_type)

    def add_edges(self, edges, edge_type, directed=True):
        for node_a, node_b in edges:
            self.add_edge(node_a, node_b, edge_type, directed)

    def traversability_edges(self) -> List[Tuple[int, int]]:
        return self.get_edge_type(EdgeType.TRAVERSABILITY)

    def hierarchy_edges(self) -> List[Tuple[int, int]]:
        return self.get_edge_type(EdgeType.HIERARCHY)

    def get_edge_type(self, edge_type) -> List[Tuple[int, int]]:
        return [(u, v) for u, v, e in self.graph.edges(data=True) if e['edge_type'] == edge_type]

    def get_node_level(self, level):
        return [n for n, data in self.graph.nodes(data=True) if data['node_level'] == level]

    def incident_edges(self, node):
        return self.graph.edges(node, data=True)

    def to_o3d(self, level=Hierarchy.ROOM):
        nodes = self.get_node_level(level)
        nodes_geometry = [node.geometry for node in nodes]
        nodes_o3d = [geometry.to_pcd(color=True).to_o3d()
                     for geometry in nodes_geometry]

        for n in nodes_o3d:
            n.paint_uniform_color(random_color())
        nodes_o3d = [o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, nodes_geometry[i].cell_size) for i, pcd in enumerate(nodes_o3d)]

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
    
    def to_voxel_grid(self):
        return VoxelGrid.merge([node.geometry for node in self.nodes()])

    def transform(self, transformation):
        # The topometric map after applying the transformation
        map_transformed = TopometricMap()

        # Apply coordinate transformation to the geometry of every node in the map
        # and add them to the new, transformed map
        nodes_t = {n: TopometricNode(n.level, n.geometry.transform(
            transformation)) for n in self.nodes()}
        map_transformed.add_nodes(nodes_t.values())

        # Get incident edges for every node in the map and add them to the
        # corresponding nodes in the transformed map
        incident_edges = [self.incident_edges(n) for n in self.nodes()]
        for edges in incident_edges:
            for n_a, n_b, data in edges:
                map_transformed.add_edge(
                    nodes_t[n_a], nodes_t[n_b], data['edge_type'])

        return map_transformed

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
        
    @staticmethod
    def from_segmented_point_cloud(geometry_fn: str, topology_fn: str, room_attr: str, voxel_size: float) -> TopometricMap:
        from model.point_cloud import PointCloud
        
        # Output topometric map that file data will be added to
        topometric_map = TopometricMap()
        
        # Load point cloud from ply file and voxelize with given voxel size
        point_cloud = PointCloud.read_ply(geometry_fn)
        voxel_grid = point_cloud.voxelize(voxel_size)
        
        # Split voxel grid by room attribute and create nodes from them, store a mapping from attribute value to node index
        # to create the edges later
        split_voxel_grid = voxel_grid.split_by_attr(room_attr)
        index_to_node = {list(vg.list_attr(room_attr))[0]: TopometricNode(Hierarchy.ROOM, vg) for vg in split_voxel_grid}
        
        topometric_map.add_nodes(index_to_node.values())            

        # Read topological graph stored as edge list in CSV file
        with open(topology_fn) as topology_file:
            edges_str = [line.split(' ') for line in topology_file.readlines()]
        edges_int = [[int(n) for n in edge_str] for edge_str in edges_str]
        
        # Add an edge between room nodes that have a traversable relationship
        for a, b in edges_int:
            topometric_map.add_edge(index_to_node[a], index_to_node[b], 
                                    EdgeType.TRAVERSABILITY,
                                    directed=False)
        return topometric_map
            
        
    def write(self, fn):
        import pickle as pickle
        with open(fn, 'wb') as write_file:
            pickle.dump(self, write_file)

    @staticmethod
    def read(fn):
        import pickle as pickle
        with open(fn, 'rb') as read_file:
            topo_map = pickle.load(read_file)
        return topo_map
