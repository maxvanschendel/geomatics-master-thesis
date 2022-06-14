
from model.sparse_voxel_octree import *
from model.voxel_grid import *
import unittest


class SVOTest(unittest.TestCase):     
    def test_morton_code(self):
        test_cases = {(0,0,0): 0, (1,0,0): 1, (0,1,0): 2, (1,1,0): 3, (0,0,1): 4, 
                      (2**21-1,2**21-1,2**21-1): 2**63-1}
        
        for voxel, morton in test_cases.items():
            self.assertEqual(morton_code(voxel), morton)
            self.assertEqual(tuple(decode_morton(morton)), voxel)
            
        self.assertEqual(morton_code((-1,-1,-1)), 2**63-1)
        self.assertEqual(morton_code((-1,-1,-1)), morton_code((2**21-1,2**21-1,2**21-1)))
        self.assertNotEqual(tuple(decode_morton(morton_code((-1,-1,-1)))), (-1,-1,-1))
            
        self.assertRaises(OverflowError, decode_morton, x=2**63)
    
    