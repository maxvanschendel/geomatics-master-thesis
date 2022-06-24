from typing import Callable
from processing.parameters import *
from processing.map_extract import extract_analyse, extract_create, extract_read, extract_visualize, extract_write
from processing.map_match import match_create, match_read, match_write, match_analyse, match_visualize
from processing.map_fuse import fuse_create, fuse_read, fuse_write, fuse_analyze, fuse_visualize
from evaluation.map_extract_performance import *
from evaluation.map_match_performance import *
from evaluation.map_fuse_performance import *
from utils.datasets import *

import logging


logging.getLogger().setLevel(logging.INFO)


class PipelineException(Exception):
    pass


def process_step(create: bool, write: bool, visualize: bool, analyse: bool,
                 create_func: Callable, write_func: Callable, read_func: Callable, visualize_func: Callable, analyse_func: Callable,
                 kwargs):

    # Step in processing timeline, in each step either create some data and optionally write it to disk, or read the data from disk.
    # After loading or creating data the data can be visualized and analysed. These actions are applicable to every step in the timeline
    # processing pipeline.

    step_failed = False

    if create:
        try:
            created = create_func(kwargs)
        except Exception as e:
            step_failed = True
            raise e

        if write:
            try:
                write_func(created, kwargs)
            except Exception as e:
                step_failed = True
                raise e
    else:
        try:
            created = read_func(kwargs)
        except Exception as e:
            step_failed = True
            raise e

    if analyse:
        try:
            analyse_func(created, kwargs)
        except Exception as e:
            step_failed = True
            raise e
            
    if visualize:
        try:
            visualize_func(created, kwargs)
        except Exception as e:
            step_failed = True
            raise e
            
    if step_failed:
        raise PipelineException("Pipeline step failed")
    else:
        return created


def run(**kwargs):
    """ 
    Pipeline entrypoint, executes steps in order using the provided configuration. 
    """

    # Either simulate partial maps or load them from files
    partial_maps: List[SimulatedPartialMap] = process_step(kwargs["simulate_partial_maps"],
                                                           kwargs["write_partial_maps"],
                                                           kwargs["visualize_partial_maps"],
                                                           kwargs["analyse_partial_maps"],
                                                           
                                                           lambda args: simulate_create(
                                                               extract_cfg, args),
                                                           simulate_write,
                                                           simulate_read,
                                                           simulate_visualize,
                                                           None,
                                                           kwargs
                                                           )

    # Get ground truth topometric maps that are aligned with voxel grid partial maps.
    # Results are compared to these to determine performance characteristics.
    ground_truth = aligned_ground_truth(
        partial_maps, extract_cfg.leaf_voxel_size*(2**extract_cfg.segmentation_lod), kwargs["graph"])

    # Extract topometric maps from voxel grid partial maps
    topometric_maps = process_step(kwargs["extract"],
                                   kwargs["write_extract"],
                                   kwargs["visualize_extract"],
                                   kwargs["analyse_extract"],
                                   
                                   lambda args: extract_create(
                                       [p.voxel_grid for p in partial_maps], extract_cfg, args),
                                   extract_write,
                                   extract_read,
                                   extract_visualize,
                                   lambda ts, args: extract_analyse(
                                       ground_truth, ts, args),
                                   kwargs,
                                   )

    # Map matching using attributed graph embedding
    matches = process_step(
        kwargs["match"],
        kwargs["write_match"],
        kwargs["visualize_match"],
        kwargs["analyse_match"],
        
        lambda args: match_create(topometric_maps, args),
        match_write,
        match_read,
        lambda m, args: match_visualize(topometric_maps, m, args),
        lambda m, args: match_analyse(ground_truth, m, topometric_maps, args),
        kwargs
    )

    # Map fusion of partial topometric maps into global topometric map
    global_map = process_step(
        kwargs["fuse"],
        kwargs["write_fuse"],
        kwargs["fuse_visualize"],
        kwargs["fuse_analyse"],
        
        lambda args: fuse_create(matches, args),
        fuse_write,
        fuse_read,
        fuse_visualize,
        lambda m, t, args: fuse_analyze(
            m, ground_truth, partial_maps, t, args),
        kwargs
    )


if __name__ == "__main__":
    extract_cfg_fn: str = './config/map_extract.yaml'
    merge_cfg_fn: str = './config/map_merge.yaml'
    dataset: Dataset = s3dis_area_3_dataset

    # Read configuration from YAML files in config directory
    extract_cfg = MapExtractionParameters.read(extract_cfg_fn)
    merge_cfg = MapMergeParameters.read(merge_cfg_fn)

    run(
        simulate_partial_maps=False,
        write_partial_maps=True,
        visualize_partial_maps=False,
        analyse_partial_maps=False,

        extract=False,
        write_extract=True,
        visualize_extract=False,
        analyse_extract=False,

        match=True,
        write_match=False,
        visualize_match=True,
        analyse_match=True,

        fuse=True,
        write_fuse=False,
        fuse_visualize=False,
        fuse_analyse=True,

        partial_maps=dataset.partial_maps,
        point_cloud=dataset.point_cloud,
        graph=dataset.graph,
        trajectories=dataset.trajectories,
        topometric_maps=dataset.topometric_maps,
    )
