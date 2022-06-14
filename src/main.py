from processing.parameters import *
from processing.pre_process import *
from processing.map_extract import *
from processing.map_match import *
from processing.map_fuse import *
from evaluation.map_extract_performance import *
from evaluation.map_match_performance import *
from evaluation.map_fuse_performance import *
from utils.datasets import *

import logging
import multiprocessing

cpu_count = multiprocessing.cpu_count()

def run():
    """ 
    Pipeline entrypoint, executes steps in order using the provided configuration. 
    """
    
    logging.info(f'Simulating scans')
    ground_truth = TopometricMap.from_segmented_point_cloud(pipeline_config.ground_truth.point_cloud, 
                                                            pipeline_config.ground_truth.graph, 
                                                            VoxelGrid.ground_truth_attr, 
                                                            map_extract_config.leaf_voxel_size)
    
    partial_maps, ground_truth_transforms = simulate_partial_maps(PointCloud.read_ply(pipeline_config.ground_truth.point_cloud), 
                                                                  read_trajectory(pipeline_config.ground_truth.trajectories), 
                                                                  map_extract_config.isovist_range,
                                                                  map_extract_config.leaf_voxel_size)
    
    logging.info(f'Extracting {len(partial_maps)} partial topometric maps')
    p = multiprocessing.Pool(cpu_count) 
    partial_topometric_maps = p.starmap(extract_topometric_map, 
                                        zip(partial_maps, [map_extract_config]*len(partial_maps)))
    
    logging.info('Matching partial maps')
    matches = match(partial_topometric_maps, map_merge_config)
    
    logging.info('Fusing partial maps')
    global_map, result_transforms = fuse(matches, map_merge_config)
    
    if pipeline_config.analyse_performance:
        map_extract_perf = analyse_extract_performance(ground_truth, global_map)
        map_match_perf = analyse_match_performance(ground_truth, partial_topometric_maps, matches)
        map_fuse_perf = analyse_fusion_performance(global_map, ground_truth, result_transforms, ground_truth_transforms)
       
    
if __name__ == "__main__":
    # Read configuration from YAML files in config directory
    preprocess_config = PreProcessingParameters.read('./config/pre_process.yaml')
    map_extract_config = MapExtractionParameters.read('./config/map_extract.yaml')
    map_merge_config = MapMergeParameters.read('./config/map_merge.yaml')
    pipeline_config = PipelineParameters(s3dis_area_3_dataset, True)

    run()
