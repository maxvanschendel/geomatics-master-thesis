import unittest

from processing.registration import *


class RegistrationTest(unittest.TestCase):
    def test_align_least_squares(self):
        tolerance = 0.01
        
        # zero-translation and rotation case
        a = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        b = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])            
        t_ab = align_least_squares(a, b)
        self.assertTrue(np.linalg.norm(t_ab.x - np.array([0,0,0,0])) < tolerance)
        
        # simple translation case 1
        a = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        b = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
        t_ab = align_least_squares(a, b)   
        self.assertTrue(np.linalg.norm(t_ab.x - np.array([-1,-1,-1, 0])) < tolerance)
        
        # simple translation case 2
        a = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
        b = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        t_ab = align_least_squares(a, b)   
        self.assertTrue(np.linalg.norm(t_ab.x - np.array([1, 1, 1, 0])) < tolerance)
        
        
