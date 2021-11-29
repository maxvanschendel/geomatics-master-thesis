from os import listdir
from typing import List

from analysis.analysis import DataAnalysis, DataViz
from model.point_cloud import PointCloud

import numpy as np
from scipy import signal, ndimage
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import open3d as o3d
import copy 

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



class o3dviz:
    i = 0

    def __init__(self, pcd):
        self.pcd = pcd

        self.cur = o3d.geometry.PointCloud()
        self.cur.points = self.pcd[self.i].points
        self.cur.colors = self.pcd[self.i].colors

        self.custom_draw_geometry_with_key_callback()

    def change_background_to_black(self, vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(self,vis):
        self.i += 1
        # self.cur = self.pcd[self.i%2]

        self.cur.points = self.pcd[self.i%len(self.pcd)].points
        self.cur.colors = self.pcd[self.i%len(self.pcd)].colors

        vis.update_geometry(self.cur)
        vis.poll_events()
        vis.update_renderer()

        return False

    def custom_draw_geometry_with_key_callback(self):
        key_to_callback = {}
        key_to_callback[ord("K")] = self.change_background_to_black
        key_to_callback[ord("R")] = self.load_render_option

        o3d.visualization.draw_geometries_with_key_callbacks([self.cur], key_to_callback)

def normal_floor_filter(cloud: PointCloud):
    cloud.normals = cloud.estimate_normals(12)

    vertical_cells = int((cloud.aabb[1][1] - cloud.aabb[1][0]) // voxel_size)
    cloud_horizontal = cloud.filter(cloud.normals, lambda x: abs(np.dot(x, (0,1,0))) > .95)

    histo, labels = DataAnalysis.dimensional_histogram(cloud_horizontal, 1, vertical_cells)
    peaks = ndimage.filters.maximum_filter1d(histo, vertical_cells)
    unique, counts = np.unique(peaks, return_counts=True)

    two_max_peaks = unique[np.argpartition(counts, -2)[-2:]]
    peak_heights = [float(labels[1:][histo == p]) for p in two_max_peaks]
    floor, ceiling = min(peak_heights), max(peak_heights)

    cloud_floor = cloud_horizontal.filter(cloud_horizontal.points, lambda x: abs(x[1] - ceiling) < voxel_size)
    cloud_ceiling = cloud_horizontal.filter(cloud_horizontal.points, lambda x: abs(x[1] - floor) < voxel_size)

    return cloud_floor, cloud_ceiling


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
    input_dir = "C:/Users/max.van.schendel/Documents/Source/geomatics-master-thesis/data/flat/low/"
    voxel_size = 0.05

    print("Reading input maps")
    cloud = list(read_input_clouds(input_dir))[0]
    
    print("Preprocessing maps")
    # voxel_map = cloud.voxel_filter(voxel_size)
    voxel_map = cloud.voxelize(voxel_size)
    viz = o3dviz([voxel_map.to_pcd().to_o3d()])

    # Visualisation
    

