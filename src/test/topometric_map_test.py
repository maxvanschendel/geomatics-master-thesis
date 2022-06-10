import unittest
from model.topometric_map import TopometricMap, Hierarchy, EdgeType

class TopometricMapTest(unittest.TestCase):
    TESTDATA_MAP = "../data/cslam/flat/flat.ply"
    TESTDATA_GRAPH = "../data/cslam/flat/flat_graph.csv"
    N_NODES = 8
    N_EDGES = 14
    
    def test_read_from_file(self):
        map = TopometricMap.from_segmented_point_cloud(TopometricMapTest.TESTDATA_MAP, 
                                                       TopometricMapTest.TESTDATA_GRAPH, 
                                                       'room', 
                                                       0.1)
        
        self.assertEqual(len(map.nodes()), TopometricMapTest.N_NODES)
        self.assertEqual(len(map.get_edge_type(EdgeType.TRAVERSABILITY)), TopometricMapTest.N_EDGES)