
import unittest
from parameterized import parameterized, parameterized_class
import numpy as np
from model.point_cloud import PointCloud
import open3d as o3d

@parameterized_class([
   { "ply_file": "../data/cslam/flat/flat.ply", "n_pts": 2985365, "has_color": True, "attributes": ["room"]},
   { "ply_file": "../data/s3dis/area_3/area_3.ply", "n_pts": 18662173, "has_color": False, "attributes": ["room"]},
])
class PointCloudTest(unittest.TestCase):
    @classmethod
    def load_pcd(self):
        return PointCloud.read_ply(self.ply_file)
    
    @parameterized.expand([
        [None, None, None],
        [np.array([]), np.array([]), {}],
    ])
    def test_create_none(self, points, colors, attributes):
        pcd = PointCloud(points, colors, attributes)
        
        self.assertTrue(pcd.points.shape == (0, 3), "")
        self.assertTrue(pcd.size == 0)
        self.assertTrue(pcd.aabb == None)
        
        self.assertTrue(pcd.colors.shape == (0, 3))
        self.assertTrue(pcd.attributes == {})
        self.assertTrue(pcd.kdt is None)
        
    def test_create(self):
        pcd = self.load_pcd()
        
        self.assertTrue(pcd.points.shape == (self.n_pts, 3))
        self.assertTrue(pcd.size == self.n_pts)
        self.assertTrue(pcd.aabb is not None)
        
        if self.has_color:
            self.assertTrue(pcd.colors.shape == pcd.points.shape)
        else:
            self.assertTrue(pcd.colors.shape == (0, 3))
            
        for attr in self.attributes:
            self.assertTrue(attr in pcd.attributes.keys())
            self.assertTrue(pcd.attributes[attr].shape[0] <= pcd.points.shape[0])
            
        self.assertTrue(pcd.kdt is not None)
        
    def test_to_o3d(self):
        pcd = self.load_pcd()
        pcd_o3d = pcd.to_o3d()
        
        self.assertTrue(type(pcd_o3d) == o3d.geometry.PointCloud)
        self.assertTrue(len(pcd_o3d.points) == pcd.size)
        
        if self.has_color:
            self.assertTrue(pcd_o3d.colors is not None)
            
    @parameterized.expand([
        [np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])],
    ])
    def test_transform(self, t):
        pcd = self.load_pcd()
        
        if t.shape != (4, 4):
            with self.assertRaises(ValueError):
                pcd_t = pcd.transform(t)
        else:
            pcd_t = pcd.transform(t)
            
        for i, _ in enumerate(pcd.points):
            self.assertTrue(
                t.dot(
                    np.hstack((pcd.points[i], np.array([1])))
                    ) 
                == pcd_t.points[i][:3])
            
        self.assertTrue(pcd.size == pcd_t.size)
        self.assertTrue(pcd.colors == pcd_t.colors)
        self.assertTrue(pcd.attributes == pcd_t.attributes)