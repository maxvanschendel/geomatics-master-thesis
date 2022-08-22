from __future__ import annotations
import csv

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


import networkx
import open3d as o3d
from utils.array import one_to_one
from utils.visualization import random_color
from model.sparse_voxel_octree import *
from model.voxel_grid import *


class Hierarchy(Enum):
    BUILDING = 0
    STOREY = 1
    ROOM = 2


@dataclass(eq=True, frozen=True)
class TopometricNode:
    level: Hierarchy = Hierarchy.ROOM
    geometry: VoxelGrid = None
    
    def __gt__(self, other):
        return self.geometry.size > other.geometry.size
    
    def __lt__(self, other):
        return other.geometry.size > self.geometry.size


class EdgeType(Enum):
    TRAVERSABILITY = 0
    HIERARCHY = 1


class TopometricMap():
    def __init__(self):
        self.graph = networkx.Graph()
        
    def node_index(self, node):
        return list(self.nodes(data=False)).index(node)

    def nodes(self, data: bool = True):
        return self.graph.nodes(data)
    
    def edges(self, node, data: bool = True):
        return self.graph.edges(node, data=data)

    def add_node(self, node: TopometricNode):
        self.graph.add_node(node, node_level=node.level)

    def add_nodes(self, nodes: List[TopometricNode]):
        for n in nodes:
            self.add_node(n)
            
    def remove_node(self, node: TopometricNode):
        self.graph.remove_node(node)
        
    def remove_nodes(self, nodes: Iterable[TopometricNode]):
        for node in nodes:
            self.remove_node(node) 

    def add_edge(self, node_a: TopometricNode, node_b: TopometricNode,
                 edge_type: EdgeType = EdgeType.TRAVERSABILITY, directed=True):
        self.graph.add_edge(node_a, node_b, edge_type=edge_type)
        if not directed:
            self.graph.add_edge(node_b, node_a, edge_type=edge_type)

    def add_edges(self, edges, edge_type, directed=True):
        for node_a, node_b in edges:
            self.add_edge(node_a, node_b, edge_type, directed)

    def get_edge_type(self, edge_type, data=True) -> List[Tuple[int, int]]:
        return [(u, v) for u, v, e in self.edges(data) if e['edge_type'] == edge_type]

    def get_node_level(self, level=Hierarchy.ROOM):
        return [n for n, data in self.nodes() if data['node_level'] == level]
    
    def incident_edges(self, node, data):
        return self.edges(node, data=data)
    
    def neighbours(self, node):       
        return self.graph.neighbors(node)
    
    def to_voxel_grid(self):
        return VoxelGrid.merge(self.geometry())

    def geometry(self):
        return [n.geometry for n in self.nodes(False)]
    
    def filter_nodes(self, func: Callable):
        return [n for n in self.nodes(data=False) if func(n)]  
        
    def to_o3d(self, level=Hierarchy.ROOM, randomize_color: bool = True, voxel: bool = True):
        nodes = self.get_node_level(level)
        nodes_geometry = [node.geometry for node in nodes]
        nodes_o3d = [geometry.to_pcd(has_color=True).to_o3d() for geometry in nodes_geometry]

        if randomize_color:
            for n in nodes_o3d:
                random_node_color = random_color()
                n.paint_uniform_color(random_node_color)
            
        if voxel:
            nodes_o3d = [o3d.geometry.VoxelGrid.create_from_point_cloud(
                pcd, nodes_geometry[i].cell_size) for i, pcd in enumerate(nodes_o3d)]

        points, lines = [node.geometry.centroid() for node in nodes], []
        for i, n in enumerate(nodes):
            edges_traversability = [(u, v) for u, v, e in self.graph.edges(
                n, data=True) if e['edge_type'] == EdgeType.TRAVERSABILITY]

            for a, b in edges_traversability:
                lines.append((i, nodes.index(b)))

        if points:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
        else:
            line_set = None
        spheres = []
        for p in points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=0.2, resolution=10)
            sphere = sphere.translate(p)
            sphere = sphere.compute_triangle_normals()

            sphere.paint_uniform_color([0, 0, 1])
            spheres.append(sphere)

        return nodes_o3d, line_set, spheres

    def transform(self, transformation):
        # The topometric map after applying the transformation
        map_transformed = TopometricMap()

        # Apply coordinate transformation to the geometry of every node in the map
        # and add them to the new, transformed map
        nodes_t = {n: TopometricNode(n.level, n.geometry.transform(
            transformation)) for n in self.nodes(False)}
        map_transformed.add_nodes(nodes_t.values())

        # Get incident edges for every node in the map and add them to the
        # corresponding nodes in the transformed map
        incident_edges = [self.incident_edges(n) for n in self.nodes(False)]
        for edges in incident_edges:
            for n_a, n_b, data in edges:
                map_transformed.add_edge(
                    nodes_t[n_a], nodes_t[n_b], data['edge_type'])

        return map_transformed

    def draw_graph(self, fn: str, edge_color: str = 'b', node_size: int = 1) -> None:
        from matplotlib.pyplot import clf, savefig
        from networkx.drawing.nx_agraph import graphviz_layout

        clf()

        hier_labels = {x: y['node_level'].name for x, y in self.graph.nodes(data=True)}
        colors = [edge_color for _ in self.graph.edges()]

        pos = graphviz_layout(self.graph, "twopi", args="-Grankdir=LR")
        networkx.draw(G=self.graph,
                      pos=pos,
                      edge_color=colors,
                      labels=hier_labels,
                      node_size=1)
        savefig(fn)

    @staticmethod
    def from_segmented_point_cloud(point_cloud, topology_fn: str, voxel_size: float) -> TopometricMap:
        # Output topometric map that file data will be added to
        topometric_map = TopometricMap()

        # voxelize with given voxel size
        voxel_grid = point_cloud.voxelize(voxel_size)

        # Split voxel grid by room attribute and create nodes from them, store a mapping from attribute value to node index
        # to create the edges later
        split_voxel_grid = voxel_grid.split_by_attr(VoxelGrid.ground_truth_attr, True)
        attr_to_node = {attr: TopometricNode(geometry=vg) for vg, attr in split_voxel_grid}

        nodes = attr_to_node.values()
        topometric_map.add_nodes(nodes)

        # Read topological graph stored as edge list in CSV file
        with open(topology_fn) as topology_file:
            csvreader = csv.reader(topology_file, delimiter=' ')
            edges = [[int(n) for n in edge_str if n != ''] for edge_str in csvreader]

        # Add an edge between room nodes that have a traversable relationship
        for attr_a, attr_b in edges:
            node_a, node_b = attr_to_node[attr_a], attr_to_node[attr_b]
            topometric_map.add_edge(node_a, node_b, directed=False)

        return topometric_map
    
    def match_nodes(self, other: TopometricMap) -> Dict[Tuple[int, int], float]:
        from scipy.optimize import linear_sum_assignment
        
        # Compute the overlap between the voxels of every segmented room
        # and every ground truth label.
        self_nodes = self.get_node_level()
        other_nodes = other.get_node_level()
        
        n_self, n_other = len(self_nodes), len(other_nodes)
        
        similarity = np.zeros((n_self, n_other))
        for i, j in product(range(n_self), range(n_other)):
            self_node, other_node = self_nodes[i], other_nodes[j]
            
            # Find overlap of extracted room voxels and ground truth subset
            jaccard = self_node.geometry.symmetric_overlap(other_node.geometry)
            similarity[i, j] = jaccard

        i, j = linear_sum_assignment(similarity, maximize=True) 
        bipartite_matching = list(zip(list(i), list(j)))
            
        return {(a, b): similarity[a, b] for a, b in bipartite_matching}

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
    
    def breadth_first_traversal(self, origin: TopometricNode, k: int = 1):
        from queue import Queue
        
        unvisited = Queue(0)
        unvisited.put((0, origin))
        
        visited = set()
        
        while unvisited.qsize():
            cur_k, cur = unvisited.get()
            
            if cur_k == k:
                visited.add(cur)
            
            if cur_k <= k:
                cur_nbs = self.neighbours(cur)
                for nb in cur_nbs:
                    if nb not in visited:
                        unvisited.put((cur_k+1,nb))
                    
                yield cur
            else:
                continue
            
    def knbr(self, origin: TopometricNode, k: int):
        from queue import Queue
        
        unvisited = Queue(0)
        unvisited.put((0, origin))
        
        visited = set()
        
        while unvisited.qsize():
            cur_k, cur = unvisited.get()
            
            if cur_k == k:
                visited.add(cur)
                yield cur
            
            if cur_k <= k:
                cur_nbs = self.neighbours(cur)
                for nb in cur_nbs:
                    if nb not in visited:
                        unvisited.put((cur_k+1,nb))
            else:
                continue
