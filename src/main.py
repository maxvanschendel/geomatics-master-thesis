from os import listdir
from typing import List

from analysis.analysis import DataAnalysis, DataViz
from model.point_cloud import PointCloud

import numpy as np
from scipy import signal, ndimage
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# HELPER FUNCTIONS
def get_files_with_extension(dir: str, ext: str):
    '''Get all files in directory that end with specified extension'''

    return filter(lambda x: x[-len(ext):] == ext, listdir(dir))

def batch_read_ply(dir: str, files: List[str]) -> List[PointCloud]:
    '''Read multiple .ply files at once to point clouds.'''

    return map(lambda fn: PointCloud.read_ply(dir + fn), files)

def read_input_clouds(dir: str) -> List[PointCloud]:
    '''Get all .ply files in a directory and read their point clouds.'''

    ply_files = get_files_with_extension(dir, '.ply')
    point_clouds = batch_read_ply(dir, ply_files)

    return point_clouds


if __name__ == "__main__":
    '''
    1. read directory containing multiply .ply point clouds                                         ✔

    2. extract topometric maps from point clouds                                                    x
    (He, Z., Sun, H., Hou, J., Ha, Y., & Schwertfeger, S. (2021). 
    Hierarchical topometric representation of 3D robotic maps. 
    Autonomous Robots, 45(5), 755–771. https://doi.org/10.1007/s10514-021-09991-8)
    2.1. Preprocess
    2.1.1   Voxel filter
    2.1.2   Denoising
    2.2. Ceiling/floor extraction
    2.3. Column generation
    2.4. Region/passage generation
    2.5. Area graph segmentation

    3. merge local topometric maps into global topometric map                                       x
    3.1 find correspondences between nodes
    3.2 find non-rigid transformation between local maps that maximizes global consistency

    4. validate results                                                                             x
    4.1. compare error to ground truth
    4.2. estimate likelihood of correctness

    5. write output .ply point cloud and visualise                                                  x
    '''

    # INPUT PARAMETERS #
    input_dir = "C:/Users/max.van.schendel/Documents/Source/geomatics-master-thesis/data/flat/"
    voxel_size = 0.05
    window = signal.windows.boxcar(50, False)

    print("Reading input maps")
    input_clouds = read_input_clouds(input_dir)
    
    print("Preprocessing maps")
    clouds_voxel_filter = list(map(lambda c: c.voxel_filter(voxel_size), input_clouds))
    for cloud in clouds_voxel_filter:
        cloud.normals = cloud.estimate_normals(12)
        cloud.colors = abs(cloud.normals)

        vertical_cells = int((cloud.aabb[1][1] - cloud.aabb[1][0]) // voxel_size)

        cloud_horizontal = cloud.filter(cloud.normals, lambda x: abs(np.dot(x, (0,1,0))) > .95)
        histo = DataAnalysis.dimensional_histogram(cloud_horizontal, 1, vertical_cells)
        peaks = ndimage.filters.maximum_filter1d(histo[0], vertical_cells)
        unique, counts = np.unique(peaks, return_counts=True)

        two_max_peaks = unique[np.argpartition(counts, -2)[-2:]]
        peak_heights = [float(histo[1][1:][histo[0] == p]) for p in two_max_peaks]
        floor, ceiling = min(peak_heights), max(peak_heights)
        cloud_floor = cloud_horizontal.filter(cloud_horizontal.points, lambda x: abs(x[1] - ceiling) < 4*voxel_size)

        cloud_erode = cloud_floor.erode(1.5*voxel_size, 9, 4)

        print(peak_heights)
        print(histo)

        DataViz.show()
        DataViz.draw_point_clouds([cloud_erode])

    # Visualisation
    

