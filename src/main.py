from processing.parameters import *
from processing.pre_process import *
from processing.map_extract import *
from processing.map_match import *
from processing.map_fuse import *


def run(preprocess_config: PreProcessingParameters, map_extract_config: MapExtractionParameters,
        map_merge_config: MapMergeParameters, pipeline_config: PipelineParameters):
    """ 
    Pipeline entrypoint, executes steps in order using the provided configuration. 
    Processing steps can optionally be skipped using pipeline config, in which case
    intermediate data from previous runs is substituted.
    """

    # Optionally skip map extraction by loading previous results from disk
    if pipeline_config.skip_extract:
        print(f'Reading topometric maps from disk\n \
                A: {pipeline_config.htmap_a_fn}\n \
                B: {pipeline_config.htmap_b_fn}')

        htmap_a = HierarchicalTopometricMap.read(pipeline_config.htmap_a_fn)
        htmap_b = HierarchicalTopometricMap.read(pipeline_config.htmap_b_fn)

    # Load point clouds from ply files, preprocess and perform topometric map extraction
    else:
        print(f'Reading point cloud partial maps from disk\n \
                A: {pipeline_config.partial_map_a}\n \
                B: {pipeline_config.partial_map_b}')

        cloud_a = PointCloud.read_ply(pipeline_config.partial_map_a)
        cloud_b = PointCloud.read_ply(pipeline_config.partial_map_b)

        print('Preprocessing partial maps')
        cloud_a_preprocess = pre_process(cloud_a, preprocess_config)
        cloud_b_preprocess = pre_process(cloud_b, preprocess_config)

        print('Extracting topometric map from partial map A')
        htmap_a = extract_map(cloud_a_preprocess, map_extract_config)

        print('Extracting topometric map from partial map B')
        htmap_b = extract_map(cloud_b_preprocess, map_extract_config)

        # Optionally write map extraction results to disk for later use
        if pipeline_config.write_htmap:
            print(f'Writing topometric maps to disk\n \
                    A: {pipeline_config.htmap_a_fn}\n \
                    B: {pipeline_config.htmap_b_fn}')

            htmap_a.write(pipeline_config.htmap_a_fn)
            htmap_b.write(pipeline_config.htmap_b_fn)

    # Find matches between topometric partial maps
    if pipeline_config.skip_match:
        matches = None  # TODO
    else:
        matches = match_maps(htmap_a, htmap_b)

    # Merge topometric partial maps into global map based on matches
    if pipeline_config.skip_merge:
        global_map = None  # TODO
    else:
        global_map = fuse(htmap_a, htmap_b, matches)


if __name__ == "__main__":
    # Read configuration from YAML files in config directory
    preprocess_config = PreProcessingParameters.read('./config/pre_process.yaml')
    map_extract_config = MapExtractionParameters.read('./config/map_extract.yaml')
    map_merge_config = MapMergeParameters.read('./config/map_merge.yaml')
    pipeline_config = PipelineParameters()

    run(preprocess_config, map_extract_config, map_merge_config, pipeline_config)
