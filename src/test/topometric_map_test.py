import unittest
from model.topometric_map import TopometricMap, Hierarchy, EdgeType
from parameterized import parameterized

from model.voxel_grid import VoxelGrid


class TopometricMapTest(unittest.TestCase):
    TESTDATA_MAP = "../data/cslam/flat/flat.ply"
    TESTDATA_GRAPH = "../data/cslam/flat/flat_graph.csv"
    N_NODES = 8
    N_EDGES = 14
    
    @parameterized.expand([
        ["../data/cslam/flat/flat.ply", "../data/cslam/flat/flat_graph.csv", 8, 14, 0.1],
    ])
    def test_read_from_file(self, map_fn, graph_fn, n_nodes, n_edges, voxel_size):
        map = TopometricMap.from_segmented_point_cloud(map_fn, graph_fn, 
                                                       VoxelGrid.ground_truth_attr, 
                                                       voxel_size)
        
        self.assertEqual(len(map.nodes()), n_nodes)
        self.assertEqual(len(map.get_edge_type(EdgeType.TRAVERSABILITY)), n_edges)
        
    @parameterized.expand([
        ["../data/cslam/flat/flat.ply", "../data/cslam/flat/flat_graph.csv", 0.1],
    ])
    def test_to_voxel_grid(self, map_fn, graph_fn, voxel_size):
        tmap = TopometricMap.from_segmented_point_cloud( map_fn, graph_fn, 
                                                        VoxelGrid.ground_truth_attr, 
                                                        voxel_size)
        voxel_grid = tmap.to_voxel_grid()
        
        self.assertEqual(type(voxel_grid), VoxelGrid)
        
        for n in tmap.nodes():
            node_voxels = n.geometry.get_voxels()
            self.assertTrue(node_voxels.issubset(voxel_grid.get_voxels()))
            
        self.assertTrue(sum([len(n.geometry.voxels) for n in tmap.nodes()]) == len(voxel_grid.voxels))
        self.assertTrue(voxel_grid == voxel_size) 