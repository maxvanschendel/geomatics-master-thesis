from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from itertools import repeat
from multiprocessing import Pool

import numpy as np
from model.map_representation import *


@dataclass(frozen=True)
class MapExtractionParameters:
    voxel_size_high : float
    voxel_size_low : float
    n_parallel: int

def extract_map(partial_map_pcd: PointCloudRepresentation) -> TopoMetricRepresentation:
    pass

def segment_floor_area(partial_map: VoxelRepresentation) -> SpatialGraphRepresentation:
    
    # For all cells in voxel map, check if any neighbours in stick kernel
    # Executed in parallel to reduce execution time
    with Pool(12) as p: 
        kernel = VoxelRepresentation.stick_kernel(partial_map.cell_size[0]/0.05)
        voxel_kernel_pairs = zip(partial_map.voxels, repeat(kernel))

        # Execute in parallel to speed things up
        kernel_points = p.starmap(                                              #
            partial_map.kernel_contains_neighbours,
            voxel_kernel_pairs
            )

    # Create new voxel map with only cells that did not have any neighbours in kernel
    floor_voxels = filter(lambda pts: pts[1] == False, zip(partial_map.voxels, kernel_points)) 
    floor_voxel_map = VoxelRepresentation(
        shape=partial_map.shape,
        cell_size=partial_map.cell_size,
        origin=partial_map.origin,
        voxels={pt[0]: partial_map[pt[0]] for pt in floor_voxels}
    )

    # Dilate floor area upwards to connect stairs and convert to nb6 connected 'dense' graph
    dilated_voxel_map = floor_voxel_map.dilate(VoxelRepresentation.cylinder(
        1, 1 + int(5 // (partial_map.cell_size[0] / 0.05))))                        
    dilated_graph_map = dilated_voxel_map.to_graph()

    # Find largest connected 
    components = dilated_graph_map.connected_components()
    floor_graph = SpatialGraphRepresentation(
        dilated_graph_map.scale,
        dilated_graph_map.origin,
        dilated_graph_map.graph.subgraph(components[0]))

    return floor_graph

def skeletonize_floor_area(floor_graph):
    floor_graph = floor_graph.minimum_spanning_tree()
    betweenness = floor_graph.betweenness_centrality(n_target_points=50)
    for node in betweenness:
        floor_graph.graph.nodes[node]['betweenness'] = betweenness[node]

    floor_voxel = floor_graph.to_voxel()
    floor_filter = floor_voxel.subset(
        lambda v, **kwargs: floor_voxel[v]['betweenness'] > kwargs['threshold'], threshold=0.1)

    floor_filter.origin = deepcopy(floor_filter.origin)
    floor_filter.origin += np.array([0, 1.8, 0.])

    return floor_filter

def cast_isovists(map_voxel_low, floor_filter):
    with Pool(12) as p:
        isovists = p.map(map_voxel_low.isovist, [list(
            floor_filter.voxel_coordinates(v)) for v in floor_filter.voxels])

    return isovists

def cluster_isovists(isovists):
    distance_matrix = np.zeros((len(isovists), len(isovists)))
    n_isovist = len(isovists)
    for i in range(len(isovists)):
        for j in range(len(isovists)):
            distance_matrix[i][j] = 1 - \
                (len(isovists[i] & isovists[j]) / len(isovists[i]))

    from sklearn.cluster import OPTICS
    clustering = OPTICS(
        min_samples=8, metric='precomputed').fit(distance_matrix)

    return clustering

def room_segmentation(isovists, map_voxel_low: VoxelRepresentation, clustering):
    map_voxel_low.for_each(map_voxel_low.set_attribute, attr='clusters', val=[])

    for iso_i, cluster in enumerate(clustering.labels_):
            if cluster != -1:
                for v in isovists[iso_i]:
                    map_voxel_low[v]['clusters'].append(cluster)

    for v in map_voxel_low.voxels:
        if len('clusters') == 0:
            kernel = VoxelRepresentation.nb6()

            cluster_found = False
            while not cluster_found:
                v_nbs = map_voxel_low.get_kernel(v, kernel)

                for v_nb in v_nbs:
                    if 'clusters' in map_voxel_low[v_nb]:
                        map_voxel_low[v]['clusters'] += map_voxel_low[v_nb]['clusters']
                        cluster_found = True
                kernel = kernel.dilate(VoxelRepresentation.nb6())

        majority_cluster = Counter(map_voxel_low[v]['clusters'])[0]
        map_voxel_low[v]['cluster'] = majority_cluster

    return map_voxel_low
