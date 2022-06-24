import unittest
from model.topometric_map import TopometricMap, EdgeType
from parameterized import parameterized

from model.voxel_grid import VoxelGrid
from utils.datasets import read_point_cloud


class TopometricMapTest(unittest.TestCase):
    @parameterized.expand([
        ["../data/cslam/flat/flat.ply",
            "../data/cslam/flat/flat_graph.csv", 8, 7, 0.1],
    ])
    def test_read_from_file(self, map_fn, graph_fn, n_nodes, n_edges, voxel_size):
        pcd = read_point_cloud(map_fn)
        map = TopometricMap.from_segmented_point_cloud(
            pcd, graph_fn, voxel_size)

        self.assertEqual(len(map.nodes()), n_nodes)
        self.assertEqual(len(map.get_edge_type(
            EdgeType.TRAVERSABILITY, data=True)), n_edges)

    @parameterized.expand([
        ["../data/cslam/flat/flat.ply", "../data/cslam/flat/flat_graph.csv", 0.1],
    ])
    def test_to_voxel_grid(self, map_fn, graph_fn, voxel_size):
        pcd = read_point_cloud(map_fn)
        tmap = TopometricMap.from_segmented_point_cloud(
            pcd, graph_fn, voxel_size)
        voxel_grid = tmap.to_voxel_grid()

        self.assertEqual(type(voxel_grid), VoxelGrid)

        for n in tmap.nodes(data=False):
            node_voxels = n.geometry.get_voxels()
            self.assertTrue(node_voxels.issubset(voxel_grid.get_voxels()))

        self.assertTrue(sum([len(n.geometry.voxels) for n in tmap.nodes(data=False)]) == len(voxel_grid.voxels))
        self.assertTrue(voxel_grid.cell_size == voxel_size)

    @parameterized.expand([
        ["../data/cslam/flat/flat.ply", "../data/cslam/flat/flat_graph.csv", 0.1],
    ])
    def test_breadth_first_traversal(self, map_fn, graph_fn, voxel_size):
        pcd = read_point_cloud(map_fn)
        tmap = TopometricMap.from_segmented_point_cloud(pcd, graph_fn, voxel_size)
        
        start_node = list(tmap.graph.nodes(data=False))[0]
        
        visited_nodes = set()
        for n in tmap.breadth_first_traversal(start_node):
            visited_nodes.add(n)
            
        self.assertTrue(len(visited_nodes) == len(tmap.nodes()))
