from dataclasses import dataclass
from itertools import repeat
from multiprocessing import Pool
from time import perf_counter
from typing import Tuple, List
from copy import deepcopy
import numpy as np
import open3d as o3d
from random import random
from model.map_representation import (PointCloudRepresentation,
                                      SpatialGraphRepresentation,
                                      VoxelRepresentation)


@dataclass
class MapViz:
    geometry: o3d.geometry
    material: o3d.visualization.rendering.MaterialRecord


class Visualizer:
    def default_pcd_mat():
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.point_size = 7

        return mat

    def default_graph_mat():
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 5
        mat.base_color = [1.0, 0, 0, 1.0]

        return mat

    def __init__(self, maps: List[List[MapViz]]):
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        gui = o3d.visualization.gui
        window = gui.Application.instance.create_window(
            "Two scenes", 1025, 512)
        widgets = []

        def on_mouse(m):
            if m.type == m.Type.DRAG:
                w_index = int(m.x // (window.content_rect.width / len(widgets)))
                if w_index < len(widgets):
                    w_focus = widgets[w_index]

                    for w_other in widgets:
                        if w_other != w_focus:
                            w_other.scene.camera.copy_from(w_focus.scene.camera)

            return gui.SceneWidget.EventCallbackResult.IGNORED

        def on_layout(theme):
            r = window.content_rect

            for i, widget in enumerate(widgets):
                widget.frame = gui.Rect(
                    r.x + ((r.width / len(widgets)) * i), r.y, r.width / len(widgets), r.height)

        for submap_index, submap in enumerate(maps):
            submap_widget = gui.SceneWidget()
            submap_widget.scene = o3d.visualization.rendering.Open3DScene(
                window.renderer)

            for object_i, submap_object in enumerate(submap):
                submap_widget.scene.add_geometry(
                    f"{submap_index}_{object_i}", submap_object.geometry, submap_object.material)

            submap_widget.setup_camera(
                60, submap_widget.scene.bounding_box, (0, 0, 0))
            submap_widget.set_on_mouse(on_mouse)

            window.add_child(submap_widget)
            widgets.append(submap_widget)

        window.set_on_layout(on_layout)
        app.run()


@dataclass(frozen=True)
class PreProcessingParameters:

    class PreProcessingParametersException(Exception):
        pass

    voxel_size: float
    reduce: float
    scale: Tuple[float]

    # input validation
    def __post_init__(self):
        if not 0 <= self.reduce <= 1:
            raise self.PreProcessingParametersException(
                "Reduce must be a fraction between 0 and 1.")

        if not len(self.scale) == 3:
            raise self.PreProcessingParametersException(
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


def pre_process(partial_map: PointCloudRepresentation, params: PreProcessingParameters) -> VoxelRepresentation:
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
    input_path = "./data/meshes/diningroom2kitchen - low.ply"

    perf = Performance()
    perf.start = perf_counter()

    print("Reading input map")
    map_cloud = PointCloudRepresentation.read_ply(input_path)
    perf.read = perf_counter()

    preprocess_parameters = PreProcessingParameters(
        voxel_size=0.1,
        reduce=1,
        scale=[1, -1, 1]
    )

    map_voxel = pre_process(map_cloud, preprocess_parameters)
    perf.pre_process = perf_counter()

    print("Extracting topological-metric map")
    print("- Segmenting floor area")

    # For all cells in input map, get neighbourhood as defined by kernel
    # Executed in parallel to reduce execution time
    with Pool(12) as p:
        kernel_points = p.starmap(
            map_voxel.kernel_contains_neighbours,
            zip(
                map_voxel.voxels.keys(),
                repeat(VoxelRepresentation.stick_kernel(
                    scale=preprocess_parameters.voxel_size/0.05)),
            ))

    floor_voxels = filter(lambda pts: pts[1] == False, zip(                      # Create new voxel map with only cells that
        map_voxel.voxels, kernel_points))                                        # does not have any other cells in neighbourhood
    floor_voxel_map = VoxelRepresentation(
        shape=map_voxel.shape,
        cell_size=map_voxel.cell_size,
        origin=map_voxel.origin,
        voxels={pt[0]: map_voxel[pt[0]] for pt in floor_voxels}
    )

    dilated_voxel_map = floor_voxel_map.dilate(VoxelRepresentation.cylinder(
        1, 1 + int(5 // (preprocess_parameters.voxel_size / 0.05))))            # Dilate floor area to connect stairs
    dilated_graph_map = dilated_voxel_map.to_graph()                            # Convert voxel floor area to dense graph
      
    components = dilated_graph_map.connected_components()
    floor_graph = SpatialGraphRepresentation(
        dilated_graph_map.scale,
        dilated_graph_map.origin,
        dilated_graph_map.graph.subgraph(components[0]))


    print('- Computing skeleton')
    floor_graph = floor_graph.minimum_spanning_tree()
    betweenness = floor_graph.betweenness_centrality(n_target_points=50)
    for node in betweenness:
        floor_graph.graph.nodes[node]['betweenness'] = betweenness[node]

    floor_voxel = floor_graph.to_voxel()
    floor_filter = floor_voxel.subset(
        lambda v, **kwargs: floor_voxel[v]['betweenness'] > kwargs['threshold'], threshold=0.2)

    floor_filter.colorize([1,0,0])
    floor_filter.origin = deepcopy(floor_filter.origin)
    floor_filter.origin += np.array([0, 1.5, 0.])

    for voxel in map_voxel.voxels:
        map_voxel[voxel]['color'] = [0.2, 0.2, 0.2]
        map_voxel[voxel]['floor'] = floor_voxel.is_occupied(voxel)

    print('- Computing isovists')
    map_voxel_low = map_cloud.scale([1,-1,1]).voxelize(0.2)
    with Pool(12) as p:
        isovists = p.map(map_voxel_low.isovist, [list(
            floor_filter.voxel_coordinates(v)) for v in floor_filter.voxels])

    for voxel in map_voxel_low.voxels:
        map_voxel_low[voxel]['color'] = [0.2, 0.2, 0.2]

    print("- Clustering isovists")
    distance_matrix = np.zeros((len(isovists), len(isovists)))
    for i in range(len(isovists)):
        for j in range(len(isovists)):
            distance_matrix[i][j] = 1 - (len(isovists[i] & isovists[j]) / len(isovists[i]))
    distance_matrix = distance_matrix / np.max(distance_matrix)

    from sklearn.cluster import OPTICS
    clustering = OPTICS(min_samples=10, metric='precomputed').fit(distance_matrix)



    label_colors = {}
    for label in np.unique(clustering.labels_):
        label_colors[label] = [random(), random(), random()]



    for iso_i, cluster in enumerate(clustering.labels_):
        if cluster != -1:
            iso = isovists[iso_i]
            for v in iso:
                map_voxel_low[v]['color'] = label_colors[cluster]

    print(distance_matrix)
    print(clustering.labels_)

    floor_voxel_flat = map_voxel.subset( lambda voxel, **kwargs: map_voxel[voxel]['floor'] == True)

    perf.map_extract = perf_counter()
    perf.map_merge = None
    print(perf)

    # Visualisation
    print("Visualising map")
    viz = Visualizer([
        [
            MapViz(map_voxel_low.to_o3d(has_color=True), Visualizer.default_pcd_mat())],
        [
            MapViz(floor_voxel_flat.to_o3d(has_color=True), Visualizer.default_pcd_mat()),
            MapViz(floor_filter.to_graph().to_o3d(has_color=True), Visualizer.default_graph_mat())
        ],
    ])
