from model.map_representation import PointCloud
from processing.map_extract import *
from processing.map_merge import *
from processing.pre_process import *

# INPUT PARAMETERS #
input_path = "./data/meshes/hall2frontbedroom - low.ply"
preprocess_parameters = PreProcessingParameters(
    reduce=1,
    scale=[1, -1, 1]
)

map_extract_parameters = MapExtractionParameters(
    # Voxelization
    voxel_size_high=0.1,
    voxel_size_low=0.2,

    # Traversability
    kernel_scale=0.1,

    # Isovists
    path_height=1.5,
    isovist_subsample=.3,
    isovist_range=3,

    # Room segmentation
    min_inflation=1.1,
    max_inflation=2,
    weight_threshold=0.2,       # Lower values lead to oversegmentation, higher to undersegmentation
    label_prop_max_its=250,     
)

if __name__ == "__main__":
    print("Reading input map")
    map_cloud = PointCloud.read_ply(input_path)

    print('Pre-processing')
    map_cloud_pp = pre_process(map_cloud, preprocess_parameters)
    map_cloud_pp.colors = map_cloud.colors

    topometric_map = extract_map(map_cloud_pp, map_extract_parameters)
