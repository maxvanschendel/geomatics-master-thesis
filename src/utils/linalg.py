import numpy as np

def norm(x: np.array):
    return np.linalg.norm(x)

def euclidean_distance(a: np.array, b: np.array):
    return norm(a - b)

def normalize(x: np.array) -> np.array:
    return x / norm(x)