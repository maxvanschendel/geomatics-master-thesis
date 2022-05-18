
from model.sparse_voxel_octree import *
from model.voxel_grid import *
import unittest


class VoxelGridTest(unittest.TestCase):
    TESTDATA_VOXEL = "../data/test/dragon_voxel.pickle"
    
    @staticmethod
    def testdata_octree():
        return VoxelGrid.read(VoxelGridTest.TESTDATA_VOXEL)


class SVOTest(unittest.TestCase):
    TESTDATA_VOXEL = "../data/test/dragon_voxel.pickle"

    @staticmethod
    def testdata_octree():
        from model.voxel_grid import VoxelGrid

        leaf_voxels = VoxelGrid.read(SVOTest.TESTDATA_VOXEL)
        octree = SVO.from_voxels(
            leaf_voxels.voxels, leaf_voxels.cell_size[0] / 2)

        return octree

    def test_read(self):
        svo = SVOTest.testdata_octree()

        self.assertIsNotNone(svo.root)
        self.assertIsNotNone(svo)
        self.assertIsNotNone(svo.nodes)

        self.assertEqual(type(svo), SVO)
        self.assertEqual(type(svo.root), OctreeNode)
        self.assertEqual(type(svo.nodes), list)
        
        self.assertGreater(len(svo.nodes), 0)
        
        for n in svo.nodes:
            self.assertEqual(type(n), OctreeNode)

    def test_from_voxels(self):
        leaf_voxels = VoxelGrid.read(self.TESTDATA_VOXEL)
        self.assertRaises(ValueError, SVO.from_voxels,
                          voxels=leaf_voxels.voxels,
                          half_width=0)

        self.assertRaises(ValueError, SVO.from_voxels,
                          voxels=leaf_voxels.mutate_voxels({}).voxels,
                          half_width=leaf_voxels.cell_size[0] / 2)

        svo = SVO.from_voxels(leaf_voxels.voxels, leaf_voxels.cell_size[0] / 2)
        self.assertEqual(len(svo.leaf_nodes()), leaf_voxels.size)

    def test_get_depth(self):
        svo = SVOTest.testdata_octree()
        max_depth = svo.max_depth()

        svo_depths = {d: svo.get_depth(d) for d in range(max_depth)}
        svo_depth_sizes = {d: v.shape[0] for d, v in svo_depths.items()}

        for d in range(max_depth - 1):
            self.assertGreaterEqual(
                svo_depth_sizes[d+1] * 8, svo_depth_sizes[d])

        self.assertEqual(svo_depth_sizes[0], 1)
        
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
        
    def test_z_order_curve(self):
        pass
    
    def test_decode_z_order_curve(self):
        pass
    
    def test_aabb_intersect(self):
        pass
    
    def test_aabb_inside(self):
        pass
    
    def test_aabb_sphere_intersect(self):
        pass
    
    