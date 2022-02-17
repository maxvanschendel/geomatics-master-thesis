import numpy as np

from analysis.visualizer import MapVisualization, Visualizer
from model.map_representation import PointCloudRepresentation
from processing.map_extract import *
from processing.map_merge import *
from processing.pre_process import *

# INPUT PARAMETERS #
input_path = "./data/meshes/diningroom2kitchen - low.ply"
preprocess_parameters = PreProcessingParameters(
    voxel_size=0.1,
    reduce=1,
    scale=[1, -1, 1]
)

map_extract_parameters = MapExtractionParameters(
    voxel_size_high = preprocess_parameters.voxel_size,
    voxel_size_low = 0.3,
    n_parallel = 8
)

if __name__ == "__main__":
    print("Reading input map")

    map_cloud = PointCloudRepresentation.read_ply(input_path)


    map_voxel_high = pre_process(map_cloud, preprocess_parameters)
    map_voxel_high.colorize([0,0,0])

    print("Extracting topological-metric map")
    print("- Segmenting floor area")
    floor_graph = segment_floor_area(map_voxel_high)
    floor_voxels = map_voxel_high.intersect(floor_graph.to_voxel())
    floor_voxel = map_voxel_high.subset(lambda v: v in floor_voxels)

    print('- Computing skeleton')
    floor_filter = skeletonize_floor_area(floor_graph)

    print('- Computing isovists')
    map_voxel_low = map_cloud.scale(preprocess_parameters.scale).voxelize(map_extract_parameters.voxel_size_low)
    isovists = cast_isovists(map_voxel_low, floor_filter)
    
    print("- Clustering isovists")
    clustering = cluster_isovists(isovists)

    print("- Segmenting rooms")
    map_voxel_low = room_segmentation(isovists, map_voxel_low, clustering)
    

    cluster_colors = {label: [random(), random(), random()] for label in np.unique(clustering.labels_)}
    map_voxel_low.for_each(lambda v: map_voxel_low.set_attribute(
        v, 'color', cluster_colors[map_voxel_low[v]['cluster']]))


    # Visualisation
    print("Visualising map")
    viz = Visualizer([
        [
            MapVisualization(map_voxel_low.to_o3d(has_color=True), Visualizer.default_pcd_mat())],
        [
            MapVisualization(floor_voxel.to_o3d(has_color=True), Visualizer.default_pcd_mat()),
            MapVisualization(floor_filter.to_graph().to_o3d(has_color=False), Visualizer.default_graph_mat())
        ],
    ])
