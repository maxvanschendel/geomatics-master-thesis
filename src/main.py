from itertools import repeat
from multiprocessing import Pool
from os import listdir
from typing import List

import numpy as np
import open3d as o3d
from scipy import ndimage

from analysis.analysis import DataAnalysis
from model.map_representation import PointCloudRepresentation, SpatialGraphRepresentation, VoxelRepresentation
import networkx

# HELPER FUNCTIONS
def get_files_with_extension(dir: str, ext: str):
    '''Get all files in directory that end with specified extension'''

    return filter(lambda x: x[-len(ext):] == ext, listdir(dir))


def batch_read_ply(dir: str, files: List[str]) -> List[PointCloudRepresentation]:
    '''Read multiple .ply files at once to point clouds.'''

    return map(lambda fn: PointCloudRepresentation.read_ply(dir + fn), files)


def read_input_clouds(dir: str) -> List[PointCloudRepresentation]:
    '''Get all .ply files in a directory and read their point clouds.'''

    ply_files = get_files_with_extension(dir, '.ply')
    point_clouds = batch_read_ply(dir, ply_files)

    return point_clouds


class o3dviz:
    def __init__(self, pcd):
        self.i = 0

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


def normal_floor_filter(cloud: PointCloudRepresentation):
    cloud.normals = cloud.estimate_normals(12)

    vertical_cells = int((cloud.aabb[1][1] - cloud.aabb[1][0]) // voxel_size)
    cloud_horizontal = cloud.filter(
        cloud.normals, lambda x: abs(np.dot(x, (0, 1, 0))) > .95)

    histo, labels = DataAnalysis.dimensional_histogram(
        cloud_horizontal, 1, vertical_cells)
    peaks = ndimage.filters.maximum_filter1d(histo, vertical_cells)
    unique, counts = np.unique(peaks, return_counts=True)

    two_max_peaks = unique[np.argpartition(counts, -2)[-2:]]
    peak_heights = [float(labels[1:][histo == p]) for p in two_max_peaks]
    floor, ceiling = min(peak_heights), max(peak_heights)

    cloud_floor = cloud_horizontal.filter(
        cloud_horizontal.points, lambda x: abs(x[1] - ceiling) < voxel_size)
    cloud_ceiling = cloud_horizontal.filter(
        cloud_horizontal.points, lambda x: abs(x[1] - floor) < voxel_size)

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
    input_dir = "C:/Users/max.van.schendel/Documents/Source/geomatics-master-thesis/data/house/low/"
    voxel_size = 0.2

    print("Reading input map")
    map_cloud = list(read_input_clouds(input_dir))[0]

    # Mirror y-axis to flip map right side up and then voxelize it
    print("Converting point cloud representation to voxel representation")
    map_cloud = map_cloud.scale(np.array([1, -1, 1]))
    map_cloud.colors = map_cloud.estimate_normals(8)
    map_voxel = map_cloud.voxelize(voxel_size)

    # map_voxel.set_attribute(map_voxel.estimate_normals(VoxelRepresentation.nb26()), 'color')

    # map_voxel_normals = map_voxel.estimate_normals(kernel=VoxelRepresentation.nb26())
    # map_voxel.set_attribute(map_voxel_normals, 'color')

    # Walkable surface extraction
    print("Applying convolution filter")

    # Construct kernel
    kernel_a = VoxelRepresentation.cylinder(7, 15).translate(np.array([0, 5, 0]))
    kernel_b = VoxelRepresentation.cylinder(1, 5).translate(np.array([9//2, 0, 9//2]))
    kernel_c = kernel_a + kernel_b
    kernel_c.origin = np.array([4, 0, 4])
    kernel_c.remove_voxel((4,0,4))
    
    # For all cells in input map, get neighbourhood as defined by kernel
    # Executed in parallel to reduce execution time
    with Pool(12) as p:
        kernel_points = p.starmap(
            map_voxel.kernel_contains_neighbours,
            zip(
                map_voxel.voxels.keys(),
                repeat(kernel_c),
                )
        )

    # Create new voxel map with only cells that did not have any other cells in neighbourhood
    floor_points = filter(lambda pts: pts[1] == False, zip(map_voxel.voxels, kernel_points))
    floor_voxel_map = VoxelRepresentation(
        shape=map_voxel.shape,
        cell_size=map_voxel.cell_size,
        origin=map_voxel.origin,
        voxels={pt[0]: map_voxel[pt[0]] for pt in floor_points}
    )

    print("Applying vertical dilation")
    dilated_voxel_map = floor_voxel_map.dilate(VoxelRepresentation.cylinder(1, 7))
    # dilated_voxel_map = dilated_voxel_map.dilate(VoxelRepresentation.nb6())

    print('Converting voxel representation to graph representation')
    dilated_graph_map = dilated_voxel_map.to_graph(nb26=False)

    print('Finding connected components')
    components = dilated_graph_map.connected_components()
    floor_graph = SpatialGraphRepresentation(dilated_graph_map.scale, dilated_graph_map.origin, dilated_graph_map.graph.subgraph(components[0]))

    print('Finding shortest paths')
    path = networkx.single_source_shortest_path_length(floor_graph.graph, list(floor_graph.graph.nodes)[0])
    max_length = max(path.values())

    for node in path: 
        floor_graph.graph.nodes[node]['distance'] = path[node]
        floor_graph.graph.nodes[node]['color'] = [path[node]/max_length, 0, path[node]/max_length]
    floor_voxel = floor_graph.to_voxel()

    # Visualisation
    print("Visualising map")

    viz = o3dviz([
                    map_voxel.to_o3d(),
                    floor_voxel_map.to_o3d(),
                    dilated_voxel_map.to_o3d(),
                    floor_voxel.to_o3d(),
                ])
