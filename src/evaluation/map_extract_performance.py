from processing.parameters import MapExtractionParameters
from itertools import product
from typing import Dict
from model.topometric_map import HierarchicalTopometricMap, Hierarchy, TopometricNode
from model.voxel_grid import VoxelGrid
from utils.array import mean_dict_value, sort_dict_by_value


def analyse_extract_performance(ground_truth: VoxelGrid, topometric_map: HierarchicalTopometricMap):
    # Compute the overlap between the voxels of every segmented room
    # and every ground truth label.
    rooms = topometric_map.get_node_level(Hierarchy.ROOM)
    room_label_overlap: Dict[(TopometricNode, int), float] = {}
    ground_truth_labels = ground_truth.unique_attr(VoxelGrid.ground_truth_attr)

    for room, label in product(rooms, ground_truth_labels):
        if room.geometry.cell_size != ground_truth.cell_size:
            raise ValueError(
                f'Room cell size ({room.geometry.cell_size}) and \
                  ground truth cell size ({ground_truth.cell_size}) not equal.')

        # Get a subset of the ground truth that has a specific label
        label_voxels = ground_truth.get_attr(VoxelGrid.ground_truth_attr, label)
        label_voxel_grid = ground_truth.voxel_subset(list(label_voxels))

        # Find overlap of extracted room voxels and ground truth subset
        overlap = room.geometry.symmetric_overlap(label_voxel_grid)
        room_label_overlap[(room, label)] = overlap

    # Map every room to every label with the highest overlap, only if
    # both room and label have not been mapped to another already
    descending_overlap = sort_dict_by_value(room_label_overlap, reverse=True)
    room_label_bijection: Dict[TopometricNode, int] = {}
    bijection_overlap: Dict[(TopometricNode, int), float] = {}

    for match, overlap in descending_overlap:
        room, label = match
        if room not in room_label_bijection and \
                label not in room_label_bijection.values():

            # We also store the overlap for each mapping to use for
            # performance analysis
            room_label_bijection[room] = label
            bijection_overlap[(room, label)] = overlap

    return {'mean_room_overlap': mean_dict_value(bijection_overlap)}


def benchmark_stanford(dir: str, config: MapExtractionParameters, cached_result=None, subset=None):
    from processing.map_extract import extract_topometric_map
    from utils.datasets import load_stanford, merge_dataset_subset

    # Load stanford dataset from disk and merge a subset into a partial map
    room_point_clouds = load_stanford(dir)
    merged_point_clouds = merge_dataset_subset(room_point_clouds, subset)

    # Make y-axis the vertical axis instead of z-axis
    merged_point_clouds = merged_point_clouds.rotate(-90, [1, 0, 0])

    # Voxelize ground truth dataset containing room labels and get its
    # level of detail at the same level that room segmentation is performed at.
    ground_truth = merged_point_clouds.voxelize(config.leaf_voxel_size*(2**config.segmentation_lod))

    print(f'Loading topometric map' if cached_result else 'Extracting topometric map')
    topometric_map = cached_result if cached_result else extract_topometric_map(
        merged_point_clouds, config)

    print('Analysing topometric map extraction performance')
    performance = analyse_extract_performance(ground_truth, topometric_map)

    return performance


if __name__ == '__main__':
    from utils.io import select_directory, load_pickle, select_file

    # Benchmark parameters
    config_fn = './config/map_extract.yaml'
    ground_truth_path = '../data/Stanford3dDataset_v1.2/Area_1'
    cached_result_fn = None

    # Prepare data
    config = MapExtractionParameters.read(config_fn)
    ground_truth = ground_truth_path
    cached_result = load_pickle(cached_result_fn) if cached_result_fn else None

    # Performance benchmark
    stanford_benchmark = benchmark_stanford(
        dir=ground_truth,
        config=config,
        cached_result=cached_result)

    # Display results
    print(stanford_benchmark)
