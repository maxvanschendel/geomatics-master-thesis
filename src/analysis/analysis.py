from model.map_representation import PointCloudRepresentation
import numpy as np
from typing import Tuple, List

import seaborn as sns
import matplotlib.pyplot as plt
import open3d as o3d

class DataAnalysis:
    def dimensional_histogram(pcd: PointCloudRepresentation, dim: int, bins: int) -> Tuple[np.array, np.array]:
        return np.histogram(pcd.points[:,dim], bins)

class DataViz:
    def distplot(pcd: PointCloudRepresentation, dim: int):
        return sns.histplot(pcd.points[:,dim], bins=25)

    def show():
        plt.show()

    def draw_point_clouds(pcds: List[PointCloudRepresentation]):
        print("Drawing point clouds using Open3D.")

        o3d.visualization.draw_geometries([c.to_o3d() for c in pcds])
