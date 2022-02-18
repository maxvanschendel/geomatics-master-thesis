from copy import deepcopy
from dataclasses import dataclass
from itertools import repeat
from multiprocessing import Pool
from typing import List

import numpy as np
from model.map_representation import *


@dataclass(frozen=True)
class MapExtractionParameters:
    voxel_size_high: float
    voxel_size_low: float
    n_parallel: int


def extract_map(partial_map_pcd: PointCloudRepresentation) -> TopoMetricRepresentation:
    pass


def segment_floor_area(partial_map: VoxelRepresentation) -> SpatialGraphRepresentation:

    # For all cells in voxel map, check if any neighbours in stick kernel
    # Executed in parallel to reduce execution time
    with Pool(8) as p:
        kernel = VoxelRepresentation.stick_kernel(
            partial_map.cell_size[0]/0.05)
        voxel_kernel_pairs = zip(partial_map.voxels, repeat(kernel))

        # Execute in parallel to speed things up
        kernel_points = p.starmap(                                              #
            partial_map.kernel_contains_neighbours,
            voxel_kernel_pairs
        )

    # Create new voxel map with only cells that did not have any neighbours in kernel
    floor_voxels = filter(lambda pts: pts[1] == False, zip(
        partial_map.voxels, kernel_points))
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
    isovists = []
    for v in floor_filter.voxels:
        isovist = map_voxel_low.isovist(floor_filter.voxel_coordinates(v))
        isovists.append(isovist)
    
    return isovists


def mutual_visibility_graph(isovists):
    n_isovist = len(isovists)
    distance_matrix = np.zeros((n_isovist, n_isovist))

    

    for i in range(n_isovist):
        for j in range(n_isovist):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                if len(isovists[i]):
                    overlap = len(isovists[i] & isovists[j]) / len(isovists[i])
                    distance_matrix[i][j] = 1 - overlap
                else:
                    distance_matrix[i][j] = 1

    print(distance_matrix)
    return distance_matrix


def cluster_isovists(isovists, min_samples):
    from sklearn.cluster import OPTICS

    mutual_visibility = mutual_visibility_graph(isovists)
    clustering = OPTICS(min_samples=min_samples,
                        metric='precomputed').fit(mutual_visibility)

    return clustering.labels_


def room_segmentation(isovists, map_voxel: VoxelRepresentation, clustering: np.ndarray):
    map_segmented = deepcopy(map_voxel)
    map_segmented.for_each(map_segmented.set_attribute,
                           attr='cluster', val=None)

    print(clustering)
    for iso_i, cluster in enumerate(clustering):
        if cluster != -1:
            for v in isovists[iso_i]:
                map_segmented[v]['cluster'] = cluster

    return map_segmented.subset(lambda v: map_segmented[v]['cluster'] is not None)


def traversability_graph(map_segmented: VoxelRepresentation, nav_graph: SpatialGraphRepresentation) -> SpatialGraphRepresentation:
    from itertools import combinations

    unique_clusters = np.unique(list(map_segmented.list_attribute('cluster')))
    graph = networkx.Graph()

    for cluster in unique_clusters:
        cluster_voxels = list(
            map_segmented.get_by_attribute('cluster', cluster))

        voxel_coordinates = [
            map_segmented.voxel_coordinates(v) for v in cluster_voxels]
        if voxel_coordinates:
            voxel_centroid = np.mean(voxel_coordinates, axis=0)
            nearest_nav_node = nav_graph.nearest_neighbour(
                voxel_centroid.reshape(1, -1))

            graph.add_node(cluster)
            graph.nodes[cluster]['pos'] = voxel_centroid
            graph.nodes[cluster]['nav_node'] = nearest_nav_node

    potential_edges = list(combinations(graph.nodes, r=2))
    print(potential_edges)

    for start_node, end_node in potential_edges:
        start, end = graph.nodes[start_node]['nav_node'], graph.nodes[end_node]['nav_node']
        path = networkx.shortest_path(
            nav_graph.graph, source=start, target=end)
        for n in path:
            n_projected = map_segmented.project(map_segmented.get_voxel(nav_graph.nodes[n]['pos']), 1, -1)

            if n_projected in map_segmented.voxels:
                cluster = map_segmented[n_projected]['cluster']
                print(cluster, n_projected, n, nav_graph.nodes[n])
