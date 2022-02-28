from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass

from typing import List
from analysis.visualizer import MapVisualization, Visualizer
from misc.helpers import random_color
import matplotlib.pyplot as plt
import numpy as np
from model.map_representation import *
from yaml import load, dump
import skopt


@dataclass(frozen=True)
class MapExtractionParameters:
    """_summary_

    Returns:
        _type_: _description_
    """

    class MapExtractionParametersException(Exception):
        pass

    voxel_size_high: float
    voxel_size_low: float
    kernel_scale: float
    n_target: int
    path_height: float
    btw_thresh: float

    isovist_subsample: float
    isovist_range: float

    weight_threshold: float
    min_inflation: float
    max_inflation: float

    label_propagation_max_iterations: int

    def __post_init__(self):
        pass

    @staticmethod
    def deserialize(data: str) -> MapExtractionParameters:
        return load(data)

    @staticmethod
    def read(fn: str) -> MapExtractionParameters:
        with open(fn, "r") as read_file:
            file_contents = read_file.read()
        return MapExtractionParameters.deserialize(file_contents)

    def serialize(self) -> str:
        return dump(self)

    def write(self, fn: str) -> None:
        with open(fn, "rw") as write_file:
            write_file.write(self.serialize())


def extract_map(partial_map_pcd: PointCloud, p: MapExtractionParameters) -> TopometricMap:
    print("Extracting topological-metric map")

    print('- Voxelizing point cloud')
    map_voxel_high = partial_map_pcd.voxelize(p.voxel_size_high)
    map_voxel_low = partial_map_pcd.voxelize(p.voxel_size_low)

    print("- Finding traversable volume")
    floor_graph = segment_floor_area(map_voxel_high)
    floor_graph_voxel = floor_graph.to_voxel()
    floor_voxel = map_voxel_high.subset(lambda v: v in floor_graph_voxel)
    
    floor_voxel.detect_peaks(axis=1)

    print('- Computing traversable volume skeleton')
    floor_filter = skeletonize_floor_area(
        floor_graph, p.n_target, p.btw_thresh, p.path_height)

    print('- Casting isovists')
    origins = floor_filter.map(floor_filter.voxel_coordinates)
    isovists = cast_isovists(origins, map_voxel_low, p.isovist_subsample, p.isovist_range)

    print(f"- Clustering {len(isovists)} isovists")
    mutual_visibility = mutual_visibility_graph(isovists)
    plt.matshow(mutual_visibility)
    plt.savefig("mutual_visibility.png")

    clustering = cluster_graph(
        distance_matrix=mutual_visibility,
        weight_threshold=p.weight_threshold,
        min_inflation=p.min_inflation,
        max_inflation=p.max_inflation
    )

    print("- Segmenting rooms")
    map_rooms = room_segmentation(isovists, map_voxel_low, clustering)

    print("- Propagating labels")
    map_rooms_prop = map_rooms.propagate_attribute(
        'cluster', p.label_propagation_max_iterations)

    cluster_colors = {label: random_color() for label in np.unique(clustering)}
    map_rooms_prop.for_each(lambda v: map_rooms_prop.set_attribute(
        v, 'color', cluster_colors[map_rooms_prop[v]['cluster']]))

    map_rooms_split = map_rooms_prop.split_by_attribute('cluster')
    connected_clusters = VoxelGrid(
        map_rooms_prop.shape, map_rooms_prop.cell_size, map_rooms_prop.origin)

    n_cluster = 0
    for split in map_rooms_split:
        kernel = VoxelGrid.nb6().dilate(VoxelGrid.nb6())
        split_components = split.connected_components(kernel)

        for c in split_components:
            c.colorize(random_color())
            c.for_each(c.set_attribute, attr='cluster', val=n_cluster)
            n_cluster += 1
            connected_clusters += c

    print(" - Extracting topometric map")
    topo_map = traversability_graph(
        connected_clusters, floor_graph, floor_graph_voxel)

    
    topometric_map = None

    # Visualisation
    print("Visualising map")
    viz = Visualizer([
        [
            MapVisualization(partial_map_pcd.to_o3d(),
                             Visualizer.default_pcd_mat())
        ],
        [
            MapVisualization(floor_voxel.to_o3d(),
                             Visualizer.default_pcd_mat()),
            MapVisualization(floor_filter.to_graph().to_o3d(
                has_color=False), Visualizer.default_graph_mat())
        ],
        [
            MapVisualization(connected_clusters.to_o3d(has_color=True),
                             Visualizer.default_pcd_mat(pt_size=10))
        ],
        [
            MapVisualization(connected_clusters.attribute_borders('cluster').to_o3d(has_color=True),
                             Visualizer.default_pcd_mat(pt_size=10))
        ] +
        [
            MapVisualization(floor_graph_voxel.to_o3d(has_color=False),
                             Visualizer.default_pcd_mat(pt_size=3))
        ],
        [
            MapVisualization(connected_clusters.to_o3d(has_color=True),
                             Visualizer.default_pcd_mat(pt_size=3))
        ] +
        [
            MapVisualization(topo_map.to_o3d(),
                             Visualizer.default_graph_mat())
        ],
    ])

    return topometric_map


def segment_floor_area(voxel_map: VoxelGrid, kernel_scale: float = 0.05, voxel_size: float = 0.1) -> SpatialGraph:
    """Uses stick-shaped structuring element to extract 3D traversable volume from voxel map. 
    Based on Gorte et al (2019).

    Args:
        voxel_map (VoxelRepresentation): Input voxel map from which the traversable volume is extracted.

    Returns:
        SpatialGraphRepresentation: Graph representation of traversable volume, nodes are connected to 6-neighbourhood.
    """

    # For all cells in voxel map, check if any neighbours in stick kernel
    kernel = VoxelGrid.stick_kernel(voxel_size / kernel_scale)
    floor_voxels = voxel_map.filter_gpu_kernel_nbs(kernel)

    # Create new voxel map with only cells that did not have any neighbours in kernel
    floor_voxel_map = voxel_map.subset(lambda vox: vox in floor_voxels)

    # Dilate floor area upwards to connect stairs and convert to nb6 connected 'dense' graph
    dilation_kernel = VoxelGrid.cylinder(
        1, 1 + int(6 // (voxel_size / kernel_scale)))
    traversable_volume_voxel = floor_voxel_map.dilate(dilation_kernel)
    # traversable_volume_voxel = traversable_volume_voxel.dilate(VoxelRepresentation.nb4())
    traversable_volume_graph = traversable_volume_voxel.to_graph()

    # Find largest connected component of traversable volume
    components = traversable_volume_graph.connected_components()
    largest_component = traversable_volume_graph.graph.subgraph(components[0])
    floor_graph = SpatialGraph(
        traversable_volume_graph.scale,
        traversable_volume_graph.origin,
        largest_component)

    return floor_graph


def skeletonize_floor_area(traversable_volume: SpatialGraph, n_target: int, betweenness_threshold: float, path_height: float) -> VoxelGrid:
    """_summary_

    Args:
        traversable_volume (SpatialGraphRepresentation): _description_
        n_target (int): _description_
        betweenness_threshold (float): _description_

    Returns:
        VoxelRepresentation: _description_
    """

    trav_vol_mst = traversable_volume.minimum_spanning_tree()
    betweenness = trav_vol_mst.betweenness_centrality(n_target_points=n_target)
    for node in betweenness:
        trav_vol_mst.graph.nodes[node]['betweenness'] = betweenness[node]

    floor_voxel = trav_vol_mst.to_voxel()
    floor_filter = floor_voxel.subset(
        lambda v, **kwargs: floor_voxel[v]['betweenness'] > betweenness_threshold)

    floor_filter.origin = deepcopy(floor_filter.origin)
    floor_filter.origin += np.array([0, path_height, 0.])
    return floor_filter


def cast_isovists(origins: List[Tuple], map_voxel: VoxelGrid, subsample: float, max_dist: float):
    isovists = []

    for v in origins:
        if random() < subsample:
            isovist = map_voxel.isovist(v, max_dist)
            isovists.append(isovist)

    return isovists


def mutual_visibility_graph(isovists) -> np.array:
    from itertools import combinations

    n_isovist = len(isovists)
    distance_matrix = np.zeros((n_isovist, n_isovist))

    pairs = combinations(range(n_isovist), r=2)
    for i, j in pairs:
        if len(isovists[i].voxels) and len(isovists[j].voxels):
            overlap = len(isovists[i].intersect(
                isovists[j])) / min([len(isovists[j].voxels), len(isovists[i].voxels)])

            distance_matrix[i][j] = 1 - overlap
            distance_matrix[j][i] = 1 - overlap

    return distance_matrix


def cluster_graph(distance_matrix, weight_threshold: float, min_inflation: float, max_inflation: float) -> List[int]:
    import markov_clustering as mc

    G = networkx.convert.to_networkx_graph(distance_matrix)
    edge_weights = networkx.get_edge_attributes(G, 'weight')
    G.remove_edges_from(
        (e for e, w in edge_weights.items() if w > weight_threshold))
    matrix = networkx.to_scipy_sparse_matrix(G)

    SPACE = [skopt.space.Real(
        min_inflation, max_inflation, name='inflation', prior='log-uniform'), ]

    @skopt.utils.use_named_args(SPACE)
    def markov_cluster(**params):
        result = mc.run_mcl(matrix, inflation=params['inflation'])
        clusters = mc.get_clusters(result)
        Q = mc.modularity(matrix=result, clusters=clusters)

        return 1-Q

    optimized_parameters = skopt.forest_minimize(
        markov_cluster, SPACE, n_calls=15, n_random_starts=10, n_jobs=-1).x
    result = mc.run_mcl(matrix, inflation=optimized_parameters[0])
    clusters = mc.get_clusters(result)

    # print(clusters)
    labeling = np.ones((len(distance_matrix)))*-1
    for i, c in enumerate(clusters):
        for iso in c:
            labeling[iso] = i

    return labeling


def room_segmentation(isovists: List[VoxelGrid], map_voxel: VoxelGrid, clustering: np.ndarray, min_observations: int = 0) -> VoxelGrid:
    map_segmented = map_voxel.clone()
    for v in map_segmented.voxels:
        map_segmented[v]['clusters'] = Counter()

    print(clustering)
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
                map_segmented[v]['cluster'] = most_common_cluster[0]

    clustered_map = map_segmented.subset(
        lambda v: 'cluster' in map_segmented[v])
    clustered_map.remove_attribute('clusters')

    return clustered_map


def traversability_graph(map_segmented: VoxelGrid, nav_graph: SpatialGraph, floor_voxels) -> SpatialGraph:
    unique_clusters = np.unique(list(map_segmented.list_attribute('cluster')))
    G = networkx.Graph()

    for cluster in unique_clusters:
        cluster_voxels = list(
            map_segmented.get_by_attribute('cluster', cluster))
        voxel_coordinates = [
            map_segmented.voxel_coordinates(v) for v in cluster_voxels]

        if voxel_coordinates:
            voxel_centroid = np.mean(voxel_coordinates, axis=0)

            G.add_node(tuple(voxel_centroid))
            G.nodes[tuple(voxel_centroid)]['pos'] = voxel_centroid
            G.nodes[tuple(voxel_centroid)]['cluster'] = cluster

    cluster_borders = map_segmented.attribute_borders('cluster')

    for v in cluster_borders.voxels:
        if floor_voxels.contains_point(cluster_borders.voxel_coordinates(v)):
            v_cluster = cluster_borders[v]['cluster']
            v_node = [x for x, y in G.nodes(
                data=True) if y['cluster'] == v_cluster][0]
            v_nbs = cluster_borders.get_kernel(v, VoxelGrid.nb6())

            for v_nb in v_nbs:
                if floor_voxels.contains_point(cluster_borders.voxel_coordinates(v_nb)):
                    v_nb_cluster = cluster_borders[v_nb]['cluster']
                    v_nb_node = [x for x, y in G.nodes(
                        data=True) if y['cluster'] == v_nb_cluster][0]

                    G.add_edge(v_node, v_nb_node)

    return SpatialGraph(np.array([1, 1, 1]), np.array([0, 0, 0]), G)
