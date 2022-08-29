from random import random

import numpy as np
from transforms3d.euler import euler2mat


def norm(x: np.array):
    return np.linalg.norm(x)

def euclidean_distance(a: np.array, b: np.array):
    return norm(a - b)

def normalize(x: np.array) -> np.array:
    return x / norm(x)

def random_transform(tmax, rmax=360):
    rot_mat = euler2mat(0, random() * rmax, 0)
    transform_mat = np.hstack((rot_mat, np.array([[random() * tmax , 
                                                   random() * tmax, 
                                                   random() * tmax]]).T))
    transform_mat = np.vstack((transform_mat, np.array([[0,0,0,1]])))
    return transform_mat
