from processing.parameters import *
from processing.pre_process import *
from processing.map_extract import *
from processing.map_match import *
from processing.map_fuse import *
from utils.datasets import simulate_scan

import logging

def simulate_partial_maps(pcd_fn, trajectory_fns, voxel_size):
    trajectories = map(lambda fn: np.genfromtxt(fn), trajectory_fns)
    ground_truth = PointCloud.read_ply(pcd_fn)
    
    ground_truth_grid = ground_truth.voxelize(voxel_size)
    partial_maps = [simulate_scan(ground_truth_grid, t, map_extract_config.isovist_range) for t in trajectories]
    
    return partial_maps
    

def run(preprocess_config: PreProcessingParameters, map_extract_config: MapExtractionParameters,
        map_merge_config: MapMergeParameters, pipeline_config: PipelineParameters):
    """ 
    Pipeline entrypoint, executes steps in order using the provided configuration. 
    Processing steps can optionally be skipped using pipeline config, in which case
    intermediate data from previous runs is substituted.
    """
    
    
    logging.info(f'Simulating scans')
    partial_maps = simulate_partial_maps(pipeline_config.ground_truth_pcd,
                                         pipeline_config.simulated_trajectories, 
                                         map_extract_config.leaf_voxel_size,)
    
    logging.info(f'Extracting {len(partial_maps)} partial topometric maps')
    partial_topometric_maps = [extract_topometric_map(m, map_extract_config) for m in partial_maps]
    
    logging.info('Matching partial maps')
    matches = match_maps(partial_topometric_maps, map_merge_config)
    
    logging.info('Fusing partial maps')
    global_map = fuse(partial_topometric_maps, matches, map_merge_config)
    
    


if __name__ == "__main__":
    # Read configuration from YAML files in config directory
    preprocess_config = PreProcessingParameters.read('../config/pre_process.yaml')
    map_extract_config = MapExtractionParameters.read('../config/map_extract.yaml')
    map_merge_config = MapMergeParameters.read('../config/map_merge.yaml')
    pipeline_config = PipelineParameters()

    run(preprocess_config, map_extract_config, map_merge_config, pipeline_config)
