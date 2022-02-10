from dataclasses import dataclass
from itertools import repeat
from multiprocessing import Pool
from time import perf_counter
from typing import Tuple

import numpy as np
import open3d as o3d

from model.map_representation import (PointCloudRepresentation,
                                      SpatialGraphRepresentation,
                                      VoxelRepresentation)


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

    def load_render_option(self, vis):
        self.i += 1

        self.cur.points = self.pcd[self.i % len(self.pcd)].points
        self.cur.colors = self.pcd[self.i % len(self.pcd)].colors

        vis.update_geometry(self.cur)
        vis.poll_events()
        vis.update_renderer()

        return False

    def custom_draw_geometry_with_key_callback(self):
        key_to_callback = {}
        key_to_callback[ord("K")] = self.change_background_to_black
        key_to_callback[ord("R")] = self.load_render_option

        o3d.visualization.draw_geometries_with_key_callbacks(
            [self.cur], key_to_callback)


class PreProcessingParametersException(Exception):
    pass


@dataclass(frozen=True)
class PreProcessingParameters:
    voxel_size: float
    reduce: float
    scale: Tuple[float]

    # input validation
    def __post_init__(self):
        if not 0 <= self.reduce <= 1:
            raise PreProcessingParametersException(
                "Reduce must be a fraction between 0 and 1.")

        if not len(self.scale) == 3:
            raise PreProcessingParametersException(
                "Scale must be a 3-dimensional vector.")

    def serialize(self) -> str:
        pass


@dataclass(frozen=True)
class MapExtractionParameters:
    pass


@dataclass
class Performance:
    start: float = None
    read: float = None
    pre_process: float = None
    map_extract: float = None
    map_merge: float = None

    def __str__(self):
        return str(vars(self))


def pre_process(partial_map: PointCloudRepresentation, params: PreProcessingParameters):
    print("Preprocessing")

    partial_map_reduced = partial_map.random_reduce(params.reduce)
    partial_map_scaled = partial_map_reduced.scale(np.array(params.scale))
    partial_map_voxel = partial_map_scaled.voxelize(params.voxel_size)

    return partial_map_voxel


def extract_map():
    pass


def merge_maps():
    pass


if __name__ == "__main__":
    '''
    1. read directory containing multiply .ply point clouds                                         ✔

    2. extract topometric maps from point clouds                                                    x
    2.1. Preprocess
    2.1.1   Voxel filter                                                                            ✔
    2.1.2   Denoising
    2.2. Floor extraction                                                                           ✔
    2.3. Room segmentation
    2.4. Graph extraction


    3. merge local topometric maps into global topometric map                                       x
    3.1 find correspondences between nodes
    3.2 find non-rigid transformation between local maps that maximizes global consistency

    4. validate results                                                                             x
    4.1. compare error to ground truth
    4.2. estimate likelihood of correctness

    5. write output .ply point cloud and visualise                                                  x
    '''
    # INPUT PARAMETERS #
    input_path = "./data/meshes/hall2frontbedroom - low.ply"

    perf = Performance()
    perf.start = perf_counter()

    print("Reading input map")
    map_cloud = PointCloudRepresentation.read_ply(input_path)
    perf.read = perf_counter()

    map_voxel = pre_process(map_cloud, PreProcessingParameters(
        voxel_size=0.05, reduce=1, scale=[1, -1, 1]))
    perf.pre_process = perf_counter()

    print("Extracting topological-metric map")
    print("- Applying convolution filter")

    # For all cells in input map, get neighbourhood as defined by kernel
    # Executed in parallel to reduce execution time
    with Pool(12) as p:
        kernel_points = p.starmap(
            map_voxel.kernel_contains_neighbours,
            zip(
                map_voxel.voxels.keys(),
                repeat(VoxelRepresentation.pen_kernel()),
            ))

    # Create new voxel map with only cells that did not have any other cells in neighbourhood
    floor_points = filter(lambda pts: pts[1] == False, zip(
        map_voxel.voxels, kernel_points))

    floor_voxel_map = VoxelRepresentation(
        shape=map_voxel.shape,
        cell_size=map_voxel.cell_size,
        origin=map_voxel.origin,
        voxels={pt[0]: map_voxel[pt[0]] for pt in floor_points}
    )

    print("- Applying vertical dilation")
    dilated_voxel_map = floor_voxel_map.dilate(
        VoxelRepresentation.cylinder(1, 7))

    print('- Converting voxel representation to graph representation')
    dilated_graph_map = dilated_voxel_map.to_graph()

    print('- Finding connected components')
    components = dilated_graph_map.connected_components()
    floor_graph = SpatialGraphRepresentation(
        dilated_graph_map.scale, dilated_graph_map.origin, dilated_graph_map.graph.subgraph(components[0]))

    print('- Computing MST')
    floor_graph = floor_graph.minimum_spanning_tree()

    print('- Computing betweenness centrality')
    betweenness = floor_graph.betweenness_centrality(n_target_points=50)
    for node in betweenness:
        floor_graph.graph.nodes[node]['betweenness'] = betweenness[node]

    perf.map_extract = perf_counter()
    perf.map_merge = None
    print(perf)

    floor_voxel = floor_graph.to_voxel()
    floor_filter = floor_voxel.subset(lambda voxel, **kwargs: floor_voxel[voxel]['betweenness'] > kwargs['threshold'],
                                      threshold=0.005)

    floor_filter.for_each(lambda voxel: floor_filter.set_attribute(voxel, 'color', [floor_filter[voxel]['betweenness']]*3))

    for voxel in map_voxel.voxels:
        if floor_voxel.is_occupied(voxel):
            map_voxel[voxel]['color'] = [1, 0, 0]
        else:
            map_voxel[voxel]['color'] = [0, 1, 0]

    # Visualisation
    print("Visualising map")
    viz = o3dviz([
        map_voxel.to_o3d(has_color=True),
        floor_voxel_map.to_o3d(),
        dilated_voxel_map.to_o3d(),
        floor_filter.to_o3d(has_color=True),
    ])
