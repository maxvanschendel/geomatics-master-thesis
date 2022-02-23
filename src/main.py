import numpy as np


from model.map_representation import PointCloudRepresentation
from processing.map_extract import *
from processing.map_merge import *
from processing.pre_process import *

# INPUT PARAMETERS #
input_path = "./data/meshes/diningroom2kitchen.ply"
preprocess_parameters = PreProcessingParameters(
    reduce=1,
    scale=[1, -1, 1]
)

map_extract_parameters = MapExtractionParameters(
    voxel_size_high = 0.1,
    voxel_size_low = 0.2,
    kernel_scale = 0.05,

    n_target = 50,
    betweenness_threshold = 0.15,

    path_height = 1.5,
    isovist_subsample = 1,
    xi = 0.01,
    min_sample = 3
)

if __name__ == "__main__":
    print("Reading input map")
    map_cloud = PointCloudRepresentation.read_ply(input_path)

    print('Pre-processing')
    map_cloud_pp = pre_process(map_cloud, preprocess_parameters)
    map_cloud_pp.colors = map_cloud.colors

    topometric_map = extract_map(map_cloud_pp, map_extract_parameters)
