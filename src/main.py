from numpy import False_
from model.point_cloud import PointCloud
from processing.map_extract import *
from processing.map_merge import *
from processing.pre_process import *

# INPUT PARAMETERS #
preprocess_parameters = PreProcessingParameters(
    reduce=1,
    scale=[1, -1, 1]
)

map_extract_parameters = MapExtractionParameters(
    # Voxelization
    leaf_voxel_size=0.1,
    traversability_lod=0,
    segmentation_lod=1,

    # Traversability
    kernel_scale=0.05,

    # Isovists
    isovist_height=1.5,
    isovist_spacing=0.5,
    isovist_subsample=1,
    isovist_range=3,

    # Room segmentation
    min_inflation=1.1,
    max_inflation=2,
    
    # Lower values lead to oversegmentation, higher to undersegmentation
    weight_threshold=0.2,
)

skip_extract = False
partial_map_a = "../data/meshes/diningroom2kitchen.ply"
partial_map_b = "../data/meshes/hall2oldkitchen.ply"

write_htmap = False
htmap_a_fn = '../data/test/diningroom2kitchen_htmap.pickle'
htmap_b_fn = '../data/test/hall2oldkitchen_htmap.pickle'

if __name__ == "__main__":
    print("Reading input map")
    if not skip_extract:
        print('Extracting A')
        map_cloud_a = PointCloud.read_ply(partial_map_a)
        map_cloud_a_pp = pre_process(map_cloud_a, preprocess_parameters)
        htmap_a = extract_map(map_cloud_a_pp, map_extract_parameters)

        print('Extracting B')
        map_cloud_b = PointCloud.read_ply(partial_map_b)
        map_cloud_b_pp = pre_process(map_cloud_b, preprocess_parameters)
        htmap_b = extract_map(map_cloud_b_pp, map_extract_parameters)

        if write_htmap:
            htmap_a.write(htmap_a_fn)
            htmap_b_fn(htmap_b_fn)
    else:
        htmap_a = HierarchicalTopometricMap.read(htmap_a_fn)
        htmap_b = HierarchicalTopometricMap.read(htmap_b_fn)

    viz = Viz([
        # Topometric map A visualization at room level
        [MapViz(o, Viz.pcd_mat(pt_size=6)) for o in htmap_a.to_o3d(Hierarchy.ROOM)[0]] +
        [MapViz(htmap_a.to_o3d(Hierarchy.ROOM)[1], Viz.graph_mat())] +
        [MapViz(o, Viz.pcd_mat()) for o in htmap_a.to_o3d(Hierarchy.ROOM)[2]],

        # Topometric map B visualization at room level
        [MapViz(o, Viz.pcd_mat(pt_size=6)) for o in htmap_b.to_o3d(Hierarchy.ROOM)[0]] +
        [MapViz(htmap_b.to_o3d(Hierarchy.ROOM)[1], Viz.graph_mat())] +
        [MapViz(o, Viz.pcd_mat()) for o in htmap_b.to_o3d(Hierarchy.ROOM)[2]
         ], ])

    matches = match_maps(htmap_a, htmap_b)
