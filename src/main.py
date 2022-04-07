from numpy import False_
from model.point_cloud import PointCloud
from processing.map_extract import *
from processing.map_merge import *
from processing.pre_process import *

skip_extract: bool = True
partial_map_a: str = "../data/cslam_dataset/diningroom2kitchen.ply"
partial_map_b: str = "../data/cslam_dataset/hall2oldkitchen.ply"

write_htmap: bool = True
htmap_a_fn: str = '../data/test/diningroom2kitchen_htmap.pickle'
htmap_b_fn: str = '../data/test/hall2oldkitchen_htmap.pickle'

if __name__ == "__main__":
    preprocess_parameters = PreProcessingParameters.read('./config/preprocess.yaml')
    map_extract_parameters = MapExtractionParameters.read('./config/map_extract.yaml')
    map_merge_parameters = MapMergeParameters.read('./config/map_merge.yaml')
    
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
            htmap_b.write(htmap_b_fn)
    else:
        htmap_a = HierarchicalTopometricMap.read(htmap_a_fn)
        htmap_b = HierarchicalTopometricMap.read(htmap_b_fn)

    matches = match_maps(htmap_a, htmap_b)
    # merge_maps(htmap_a, htmap_b, matches)
