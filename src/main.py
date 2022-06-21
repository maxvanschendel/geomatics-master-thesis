from processing.parameters import *
from processing.map_extract import extract
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

def write_multiple(fns, objs, write_func):
    for i, p in enumerate(objs):
        out_fn = fns[i]
        write_func(p, out_fn)
    
def run(**kwargs):
    """ 
    Pipeline entrypoint, executes steps in order using the provided configuration. 
    """
    
    # Either simulate partial maps or load them from files
    if kwargs["simulate_partial_maps"]:
        logging.info(f'Simulating partial maps')
        
        point_cloud = read_point_cloud(kwargs["point_cloud"])
        trajectories = read_trajectory(kwargs["trajectories"])
        
        partial_maps = simulate_partial_maps(
            point_cloud, 
            trajectories, 
            map_extract_config.isovist_range,
            map_extract_config.leaf_voxel_size)
        
        # Optionally write simulation output to be used later
        if kwargs["write_partial_maps"]:
            logging.info(f'Writing partial maps')
            write_multiple(kwargs["partial_maps"], partial_maps, lambda p, fn: p.write(fn))
    else:
        logging.info(f'Loading partial maps')
        partial_maps = [PartialMap.read(fn) for fn in kwargs["partial_maps"]]
    
    
    partial_maps = [p.voxel_grid for p in partial_maps]
    ground_truth_transforms = [p.transform for p in partial_maps]
    
    
    # Either extract partial topometric maps from partial maps or load them from files
    if kwargs["extract"]:
        logging.info(f'Extracting {len(partial_maps)} partial topometric maps')
        p = multiprocessing.Pool(cpu_count) 
        topometric_maps = p.starmap(extract, zip(partial_maps, [map_extract_config]*len(partial_maps)))
        
        if kwargs["write_extract"]:
            logging.info(f'Writing topometric maps')
            write_multiple(kwargs["topometric_maps"], topometric_maps, lambda p, fn: p.write(fn))
    else:
        topometric_maps = [TopometricMap.read(fn) for fn in kwargs["topometric_maps"]]
        
    if kwargs["analyse_extract"]:
        ground_truth = TopometricMap.from_segmented_point_cloud(kwargs["point_cloud"], 
                                                                kwargs["graph"], 
                                                                map_extract_config.leaf_voxel_size)
        for t in topometric_maps:
            map_extract_perf = mean_similarity(ground_truth, t)
            logging.info(f"Map extract performance: {map_extract_perf}")
    
    
    # Map matching
    if kwargs["match"]:
        logging.info('Matching partial maps')
        matches = match(topometric_maps, map_merge_config)
    else:
        raise NotImplementedError("Not implemented")
    
    if kwargs["analyse_match"]:
        for a, b in matches.keys():
            map_match_perf = analyse_match_performance(a, b, ground_truth, matches[(a,b)])
            logging.info(f'Map match performance for partial maps {a} and {b}: {map_match_perf}')
    
    
    # Map fusion
    if kwargs["fuse"]:
        logging.info('Fusing partial maps')
        global_map, result_transforms = fuse(matches, 'pnlk')
    else:
        raise NotImplementedError("Not implemented")
    
    if kwargs["analyse_fuse"]:
        map_fuse_perf = analyse_fusion_performance(global_map, ground_truth, result_transforms, ground_truth_transforms)
        logging.info(f'Map fusion performance: {map_fuse_perf}')
       
    
if __name__ == "__main__":
    # Read configuration from YAML files in config directory
    preprocess_config = PreProcessingParameters.read('./config/pre_process.yaml')
    map_extract_config = MapExtractionParameters.read('./config/map_extract.yaml')
    map_merge_config = MapMergeParameters.read('./config/map_merge.yaml')

    dataset: Dataset = cslam_flat_dataset
    
    run(
        simulate_partial_maps = False,
        write_partial_maps = True,
        
        extract = False,
        write_extract = True,
        
        match = True,
        
        fuse = False,
        write_fuse = False,
        
        analyse_extract = True,
        analyse_match = True,
        analyse_fuse = True,
        
        analyse_performance = True,
        
        partial_maps = dataset.partial_maps,
        point_cloud = dataset.point_cloud,
        graph = dataset.graph, 
        trajectories = dataset.trajectories,
        topometric_maps = dataset.topometric_maps,
    )
