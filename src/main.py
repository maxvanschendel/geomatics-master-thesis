from processing.parameters import *
from processing.map_extract import extract_topometric_map
from processing.map_match import match
from processing.map_fuse import fuse
from evaluation.map_extract_performance import *
from evaluation.map_match_performance import *
from evaluation.map_fuse_performance import *
from utils.datasets import *

import logging
import multiprocessing

cpu_count = multiprocessing.cpu_count()
logging.getLogger().setLevel(logging.INFO)

def run(**kwargs):
    """ 
    Pipeline entrypoint, executes steps in order using the provided configuration. 
    """
    
    # Either simulate partial maps or load them from storage
    if kwargs["simulate_partial_maps"]:
        logging.info(f'Simulating partial maps')
        
        partial_maps = simulate_partial_maps(
            read_point_cloud(kwargs["point_cloud"]), 
            read_trajectory(kwargs["trajectories"]), 
            map_extract_config.isovist_range,
            map_extract_config.leaf_voxel_size)
        
        # Optionally write simulation output to be used later
        if kwargs["write_partial_maps"]:
            logging.info(f'Writing partial maps')
            
            for i, p in enumerate(partial_maps):
                out_fn = kwargs["partial_maps"][i]
                p.write(out_fn)
    else:
        logging.info(f'Loading partial maps')
        partial_maps = [PartialMap.read(fn) for fn in kwargs["partial_maps"]]
    
    partial_maps = [p.voxel_grid for p in partial_maps]
    ground_truth_transforms = [p.transform for p in partial_maps]
    
    
    
    
    logging.info(f'Extracting {len(partial_maps)} partial topometric maps')
    p = multiprocessing.Pool(cpu_count) 
    partial_topometric_maps = p.starmap(extract_topometric_map, zip(partial_maps, [map_extract_config]*len(partial_maps)))
    
    
    
    logging.info('Matching partial maps')
    matches = match(partial_topometric_maps, map_merge_config)
    
    
    
    logging.info('Fusing partial maps')
    global_map, result_transforms = fuse(matches, map_merge_config)
    
    
    
    if kwargs["analyse_performance"]:
        logging.info(f'Reading ground truth data from {kwargs["point_cloud"]}')
        ground_truth = TopometricMap.from_segmented_point_cloud(kwargs["point_cloud"], 
                                                                kwargs["graph"], 
                                                                map_extract_config.leaf_voxel_size)
    
        map_extract_perf = analyse_extract_performance(ground_truth, global_map)
        map_match_perf = analyse_match_performance(ground_truth, partial_topometric_maps, matches)
        map_fuse_perf = analyse_fusion_performance(global_map, ground_truth, result_transforms, ground_truth_transforms)
       
    
if __name__ == "__main__":
    # Read configuration from YAML files in config directory
    preprocess_config = PreProcessingParameters.read('./config/pre_process.yaml')
    map_extract_config = MapExtractionParameters.read('./config/map_extract.yaml')
    map_merge_config = MapMergeParameters.read('./config/map_merge.yaml')

    dataset: Dataset = s3dis_area_3_dataset
    
    run(
        simulate_partial_maps = False,
        write_partial_maps = True,
        analyse_performance = True,
        
        partial_maps = dataset.partial_maps,
        point_cloud = dataset.point_cloud,
        graph = dataset.graph, 
        trajectories = dataset.trajectories
    )
