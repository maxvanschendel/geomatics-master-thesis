import numpy as np

class PointCloud:
    def __init__(self, points: np.array):
        self.points = points

    def __str__(self):
        return str(self.points)