
import logging

from processing.configuration import *
from processing.extract import (extract_analyse, extract_create, extract_read,
                                extract_visualize, extract_write)
from processing.fuse import (fuse_analyze, fuse_create, fuse_read,
                             fuse_visualize, fuse_write)
from processing.match import (match_analyse, match_create, match_read,
                              match_visualize, match_write)

from utils.datasets import *
from utils.pipeline import process_step

logging.getLogger().setLevel(logging.INFO)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


def run(**kwargs):
    """ 
    Pipeline entrypoint, executes steps in order using the provided configuration. 
    """

    if kwargs["logging"]:
        import logging
        logging.getLogger().setLevel(logging.INFO)

    # Either simulate partial maps or load them from files
    partial_maps = process_step(kwargs["simulate_partial_maps"],
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

    if False:
        # Get ground truth topometric maps that are aligned with voxel grid partial maps.
        # Results are compared to these to determine performance characteristics.
        ground_truth = aligned_ground_truth(partial_maps,
                                            extract_cfg.leaf_voxel_size *
                                            (2**extract_cfg.segmentation_lod),
                                            kwargs["graph"])
    else:
        ground_truth = None

    # Extract topometric maps from voxel grid partial maps
    topometric_maps = process_step( kwargs["extract"],
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
        logging=True,

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
        analyse_match=False,

        fuse=False,
        write_fuse=False,
        fuse_visualize=False,
        fuse_analyse=True,

        partial_maps=dataset.partial_maps,
        point_cloud=dataset.point_cloud,
        graph=dataset.graph,
        trajectories=dataset.trajectories,
        topometric_maps=dataset.topometric_maps,
    ) 
