from __future__ import annotations

import logging
import multiprocessing
from collections import Counter
from copy import deepcopy
from itertools import combinations
from random import random
from typing import List, Tuple

import markov_clustering as mc
import networkx as nx
import numpy as np
import skopt

from model.spatial_graph import SpatialGraph
from model.topometric_map import Hierarchy, TopometricMap, TopometricNode
from model.voxel_grid import Kernel, VoxelGrid
from processing.configuration import MapExtractionParameters
from utils.datasets import write_multiple
from utils.visualization import visualize_htmap


def extract_create(partial_maps: List[VoxelGrid], config: MapExtractionParameters, kwargs):
    logging.info(f'Extracting {len(partial_maps)} partial topometric maps')

    cpu_count = multiprocessing.cpu_count()
    p = multiprocessing.Pool(1)

    topometric_maps = p.starmap(extract, zip(
        partial_maps, [config]*len(partial_maps)))

    return topometric_maps


def extract_write(topometric_maps, kwargs):
    logging.info(f'Writing topometric maps')
    write_multiple(kwargs["topometric_maps"],
                   topometric_maps, lambda p, fn: p.write(fn))


def extract_read(kwargs):
    return [TopometricMap.read(fn) for fn in kwargs["topometric_maps"]]


def extract_visualize(topometric_maps, kwargs):
    for t in topometric_maps:
        visualize_htmap(t)


def extract_analyse(truths, topometric_maps, kwargs):
    logging.info("Analysing topometric map extraction performance")
    for i, t in enumerate(topometric_maps):
        map_extract_perf = mean_similarity(truths[i], t)
        logging.info(f"Map extract performance: {map_extract_perf}")


def extract(leaf_voxels: VoxelGrid, p: MapExtractionParameters, **kwargs) -> TopometricMap:
    try:
        # Map representation that is result of map extraction
        topometric_map = TopometricMap()
        traversability_lod = leaf_voxels.level_of_detail(p.traversability_lod)

        logging.info('Extracting traversable volume')
        # Extract the traversable volume and get building voxels at its bottom (the floor)
        nav_volume_voxel, floor_voxel = segment_floor_area(traversability_lod, p.kernel_scale, p.leaf_voxel_size)
        floor_voxel = deepcopy(floor_voxel)

        logging.info('Estimating optimal isovist positions')
        # Attempt to find the positions in the map from which as much of the
        # the map is visible as possible.
        isovist_voxels = optimal_isovist_voxels(
            floor_voxel, p.isovist_height, p.isovist_spacing)
        isovist_centroids = isovist_voxels.voxel_centroids()

        logging.info(
            f'Voxelizing partial map at LoD{p.segmentation_lod} for room segmentation')
        segmentation_lod = leaf_voxels.level_of_detail(p.segmentation_lod)

        logging.info(f'Casting {len(isovist_centroids)} isovists')
        visibilities = cast_visibilities(
            origins=isovist_centroids,
            map_grid=segmentation_lod,
            subsample=p.isovist_subsample,
            max_dist=p.isovist_range
        )

        logging.info(f'Clustering {len(visibilities)} isovists')
        visibility_graph = mutual_visibility_graph(visibilities)
        visibility_clustering = cluster_graph_mcl(
            distance_matrix=visibility_graph,
            weight_threshold=p.weight_threshold,
            min_inflation=p.min_inflation,
            max_inflation=p.max_inflation
        )

        logging.info(f'Segmenting rooms')
        map_rooms = room_segmentation(
            isovists=visibilities,
            map_voxel=segmentation_lod,
            clustering=visibility_clustering
        )

        if not len(map_rooms.voxels):
            raise Exception('Room segmentation is empty')

        logging.info(f'Propagating labels') 
        map_rooms = map_rooms.propagate_attr(
            attr=VoxelGrid.cluster_attr,
            prop_kernel=Kernel.sphere(r=2), max_its=10)

        logging.info(f'Finding connected clusters')
        map_rooms_split = map_rooms.split_by_attr(VoxelGrid.cluster_attr)

        connection_kernel = Kernel.sphere(r=2)
        connected_clusters = VoxelGrid(map_rooms.cell_size, map_rooms.origin)

        n_cluster = 0
        for room in map_rooms_split:
            room_components = room.connected_components(connection_kernel)

            for c in room_components:
                c.set_attr_uniform(attr=VoxelGrid.cluster_attr, val=n_cluster)
                n_cluster += 1
                connected_clusters += c

        logging.info(f'Extracting topometric map')
        topology = traversability_graph(
            map_segments=connected_clusters,
            floor_voxels=nav_volume_voxel,
            min_voxels=p.min_voxels)

        node_dict = {n: TopometricNode(
            geometry=topology.graph.nodes[n]['geometry']) for n in topology.graph.nodes}

        for node in node_dict:
            topometric_map.add_node(node_dict[node])

        for node in node_dict:
            node_edges = topology.graph.edges(node)
            for a, b in node_edges:
                if a != b:
                    topometric_map.add_edge(
                        node_dict[a], node_dict[b], directed=False)

        return topometric_map
    except Exception as e:
        logging.error(f"Failed to extract topometric map: {e}")
        raise e


def segment_floor_area(voxel_map: VoxelGrid, kernel_scale: float = 0.05, voxel_size: float = 0.1) -> SpatialGraph:
    '''Uses stick-shaped structuring element to extract 3D traversable volume from voxel map. 
    Based on Gorte et al (2019).
    Args:
        voxel_map (VoxelRepresentation): Input voxel map from which the traversable volume is extracted.
    Returns:
        SpatialGraphRepresentation: Graph representation of traversable volume, nodes are connected to 6-neighbourhood.
    '''

    # For all cells in voxel map, check if any neighbours in stick kernel
    # Create new voxel map with only cells that did not have any neighbours in kernel
    stick_kernel = Kernel.stick_kernel(voxel_size / kernel_scale)
    candidate_voxels = voxel_map.filter_gpu_kernel_nbs(stick_kernel)

    # Dilate floor area upwards to connect stairs and convert to nb6 connected 'dense' graph
    # TODO: fix kernel scales, argument should just be its dimensions in meters.
    dilation_kernel = Kernel.cylinder(
        1, 1 + int(6 // (voxel_size / kernel_scale)))
    nav_volume = candidate_voxels.dilate(dilation_kernel)

    # Find the largest connected component that is traversable
    nav_kernel = Kernel.nb6()
    nav_components = nav_volume.connected_components(nav_kernel)
    nav_manifold = nav_components[0]

    nav_voxels = candidate_voxels.subset(
        lambda v: v in nav_manifold)
    return nav_manifold, nav_voxels


def local_distance_field_maxima(vg, radius, min=0) -> VoxelGrid:
    distance_field = vg.distance_field()

    local_maxima = set()
    for vx, vx_dist in distance_field.items():
        vx_nbs = vg.radius_search(vg.voxel_centroid(vx), radius)
        vx_nbs_dist = [distance_field[nb] for nb in vx_nbs]

        if vx_dist >= max(vx_nbs_dist) and vx_dist >= min:
            local_maxima.add(vx)

    return vg.subset(lambda v: v in local_maxima)


def mean_similarity(ground_truth: TopometricMap, extracted: TopometricMap) -> float:
    o2o_similarity = extracted.match_nodes(ground_truth)
    mean_similarity = np.mean(list(o2o_similarity.values()))

    return mean_similarity


def optimal_isovist_voxels(floor: VoxelGrid, height: Tuple[float, float], radius=7, min_dist=1) -> VoxelGrid:
    # Find voxels that are the furthest away from the boundary of the map within a given radius
    # Add height and random jitter to distance field maxima to simulate
    # a scanner moving through the environment.
    df_maxima = local_distance_field_maxima(floor, radius, min_dist)
    df_maxima.origin += np.array([0, height[0] +
                                 (random()*(height[1]-height[0])), 0.])

    return df_maxima


def cast_visibilities(origins: List[Tuple], map_grid: VoxelGrid, subsample: float, max_dist: float):
    isovists = [map_grid.visibility(o, max_dist)
                for o in origins if random() < subsample]

    return isovists


def mutual_visibility_graph(isovists) -> np.array:
    n_isovist = len(isovists)
    distance_matrix = np.zeros((n_isovist, n_isovist))

    pairs = combinations(range(n_isovist), r=2)
    for i, j in pairs:
        overlap = isovists[i].jaccard_index(isovists[j])

        # Mutual visibility is symmetric between isovists
        distance_matrix[i][j] = 1 - overlap
        distance_matrix[j][i] = 1 - overlap

    return distance_matrix


def cluster_graph_mcl(distance_matrix, weight_threshold: float, min_inflation: float, max_inflation: float) -> List[int]:
    G = nx.convert.to_networkx_graph(distance_matrix)
    edge_weights = nx.get_edge_attributes(G, 'weight')
    G.remove_edges_from(
        (e for e, w in edge_weights.items() if w > weight_threshold))
    matrix = nx.to_scipy_sparse_matrix(G)

    SPACE = [skopt.space.Real(
        min_inflation, max_inflation, name='inflation', prior='log-uniform'), ]

    @skopt.utils.use_named_args(SPACE)
    def markov_cluster(**params):
        result = mc.run_mcl(matrix, inflation=params['inflation'])
        clusters = mc.get_clusters(result)
        Q = mc.modularity(matrix=result, clusters=clusters)

        return 1-Q

    # Find hyperparameters that produce optimal clustering, then perform clustering using them
    optimized_parameters = skopt.forest_minimize(
        markov_cluster, SPACE, n_calls=10, n_random_starts=10, n_jobs=-1).x
    result = mc.run_mcl(matrix, inflation=optimized_parameters[0])
    clusters = mc.get_clusters(result)

    labeling = np.ones((len(distance_matrix)))*-1
    for i, c in enumerate(clusters):
        for iso in c:
            labeling[iso] = i

    return labeling


def room_segmentation(isovists: List[VoxelGrid], map_voxel: VoxelGrid, clustering: np.ndarray, min_observations: int = 0) -> VoxelGrid:
    map_segmented = map_voxel.clone()
    for v in map_segmented.voxels:
        map_segmented[v]['clusters'] = Counter()

    for iso_i, cluster in enumerate(clustering):
        if cluster != -1:
            iso_cur = isovists[iso_i]
            for v in iso_cur.voxels:
                map_segmented[v]['clusters'][cluster] += 1

    for v in map_segmented.voxels:
        if map_segmented[v]['clusters']:
            most_common_cluster = map_segmented[v]['clusters'].most_common(1)[
                0]
            if most_common_cluster[1] > min_observations:
                map_segmented[v][VoxelGrid.cluster_attr] = most_common_cluster[0]

    clustered_map = map_segmented.subset(
        lambda v: VoxelGrid.cluster_attr in map_segmented[v])
    return clustered_map


def traversability_graph(map_segments: VoxelGrid, floor_voxels: VoxelGrid, min_voxels: float = 32) -> SpatialGraph:
    G = nx.Graph()
    cluster_attr = VoxelGrid.cluster_attr

    unique_clusters = np.unique(list(map_segments.list_attr(cluster_attr)))

    for cluster in unique_clusters:
        cluster_voxels = list(map_segments.get_attr(cluster_attr, cluster))
        voxel_coordinates = [
            map_segments.voxel_centroid(v) for v in cluster_voxels]

        if voxel_coordinates:
            voxel_centroid = np.mean(voxel_coordinates, axis=0)
            voxel_centroid_index = tuple(voxel_centroid)

            G.add_node(voxel_centroid_index)
            G.nodes[voxel_centroid_index]['pos'] = voxel_centroid
            G.nodes[voxel_centroid_index][cluster_attr] = cluster
            G.nodes[voxel_centroid_index]['geometry'] = map_segments.subset(
                lambda v: v in cluster_voxels)

    kernel = Kernel.sphere(2)
    cluster_borders = map_segments.attr_borders(cluster_attr, kernel)

    for v in cluster_borders.voxels:
        if floor_voxels.contains_point(cluster_borders.voxel_centroid(v)):
            v_cluster = cluster_borders[v][cluster_attr]
            v_node = [x for x, y in G.nodes(
                data=True) if y[cluster_attr] == v_cluster][0]

            v_nbs = cluster_borders.get_kernel(v, kernel)
            for v_nb in v_nbs:
                if floor_voxels.contains_point(cluster_borders.voxel_centroid(v_nb)):
                    v_nb_cluster = cluster_borders[v_nb][cluster_attr]
                    v_nb_node = [x for x, y in G.nodes(
                        data=True) if y[cluster_attr] == v_nb_cluster][0]

                    G.add_edge(v_node, v_nb_node)

    connected_nodes = [n for (n, d) in G.nodes(
        data=True) if len(d['geometry'].voxels) > min_voxels]
    traversability_graph = SpatialGraph(np.array([1, 1, 1]),
                                        np.array([0, 0, 0]),
                                        G.subgraph(connected_nodes))

    return traversability_graph
