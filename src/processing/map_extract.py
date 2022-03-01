from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import List

import numpy as np
import skopt
from yaml import dump, load

from analysis.visualizer import MapViz, Viz
from misc.helpers import random_color
from model.map_representation import *

@dataclass(frozen=True)
class MapExtractionParameters:
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

    label_prop_max_its: int

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


def extract_map(partial_map_pcd: PointCloud, p: MapExtractionParameters) -> HierarchicalTopometricMap:
    print("Extracting topological-metric map")
    topometric_map = HierarchicalTopometricMap()

    print('- Voxelizing point cloud')
    building_voxel_high = partial_map_pcd.voxelize(p.voxel_size_high)
    building_node = TopometricNode(Hierarchy.BUILDING, building_voxel_high)
    topometric_map.add_node(building_node)

    print("- Finding traversable volume")
    nav_graph = segment_floor_area(building_voxel_high)
    nav_graph_voxel = nav_graph.to_voxel()
    floor_voxel = building_voxel_high.subset(lambda v: v in nav_graph_voxel)


    print("- Detecting storeys")
    storeys, connections = detect_storeys(floor_voxel, building_voxel_high, buffer=10)
    storey_nodes = list(map(lambda s: TopometricNode(Hierarchy.STOREY, s), storeys))
    
    topometric_map.add_nodes(storey_nodes)
    for s in storey_nodes:
        topometric_map.add_edge(building_node, s, EdgeType.HIERARCHY)

    for a, b in connections:
        topometric_map.add_edge(storey_nodes[a], storey_nodes[b], EdgeType.TRAVERSABILITY)

    print('- Computing traversable volume skeleton')
    floor_filter = skeletonize_floor_area(
         building_voxel_high.subset(lambda v: v in nav_graph_voxel), p.n_target, p.btw_thresh, p.path_height)

    print(f"- Segmenting {len(storey_nodes)} storey(s)")
    for i, storey in enumerate(storey_nodes):
        print(f"- Segmenting storey {i+1}/{len(storey_nodes)}")
        
        print(f'    - Voxelizing storey with cell size {p.voxel_size_low} ')
        storey_geometry_low = storey.geometry.to_pcd().voxelize(p.voxel_size_low)
        storey_centroids = storey_geometry_low.voxel_centroids()
        min_bbox, max_bbox = storey_centroids.min(
            axis=0), storey_centroids.max(axis=0)

        print(f'    - Casting isovists')
        origins = floor_filter.voxel_centroids()
        origins = list(filter(lambda o: np.all(np.greater(o, min_bbox)) and np.all(np.less(o, max_bbox)), origins))

        isovists = cast_isovists(
            origins=origins,
            map_voxel=storey_geometry_low,
            subsample=p.isovist_subsample,
            max_dist=p.isovist_range)


        print(f'    - Clustering {len(isovists)} isovists')
        mutual_visibility = mutual_visibility_graph(isovists)
        clustering = cluster_graph(
            distance_matrix=mutual_visibility,
            weight_threshold=p.weight_threshold,
            min_inflation=p.min_inflation,
            max_inflation=p.max_inflation
        )

        print(f"    - Segmenting rooms")
        map_rooms = room_segmentation(
            isovists=isovists,
            map_voxel=storey_geometry_low,
            clustering=clustering)

        print(f"    - Propagating labels")
        prop_kernel = VoxelGrid.nb6()
        prop_kernel = prop_kernel.translate(-prop_kernel.origin)
        prop_kernel = prop_kernel.dilate(VoxelGrid.nb6()).dilate(VoxelGrid.nb6())
        
        map_rooms = map_rooms.propagate_attribute(
            attr=VoxelGrid.cluster_attr,
            max_iterations=p.label_prop_max_its,
            prop_kernel=prop_kernel)

        map_rooms_split = map_rooms.split_by_attr(VoxelGrid.cluster_attr)
        
        n_cluster = 0
        connection_kernel = VoxelGrid.nb6().dilate(VoxelGrid.nb6())
        connected_clusters = VoxelGrid(map_rooms.shape, map_rooms.cell_size, map_rooms.origin)
        
        for room in map_rooms_split:
            room_components = room.connected_components(connection_kernel)

            for c in room_components:
                c.set_attr_uniform(attr=VoxelGrid.cluster_attr, val=n_cluster)
                n_cluster += 1
                connected_clusters += c


        print(f"    - Extracting topometric map")
        topo_map = traversability_graph(
            map_segmented=connected_clusters,
            nav_graph=floor_voxel,
            floor_voxels=nav_graph_voxel)

        node_dict = {n: TopometricNode(
            Hierarchy.ROOM, topo_map.graph.nodes[n]['geometry']) for n in topo_map.graph.nodes}

        for node in node_dict:
            topometric_map.add_node(node_dict[node])
            topometric_map.add_edge(
                storey, node_dict[node], EdgeType.HIERARCHY)

        for node in node_dict:
            node_edges = topo_map.graph.edges(node)
            for a, b in node_edges:
                if a != b:
                    topometric_map.add_edge(
                        node_dict[a], node_dict[b], EdgeType.TRAVERSABILITY)

    # Visualisation
    print("Visualising map")
    viz = Viz([
        [
            MapViz(partial_map_pcd.to_o3d(),
                   Viz.pcd_mat())
        ],
        [
            MapViz(floor_voxel.to_o3d(has_color=True),
                   Viz.pcd_mat()),
            MapViz(floor_filter.to_o3d(
                has_color=False), Viz.pcd_mat())
        ],

        # Topometric map visualization at ROOM level
        [MapViz(o, Viz.pcd_mat(pt_size=4)) for o in topometric_map.to_o3d(Hierarchy.ROOM)[0]] +
        [MapViz(topometric_map.to_o3d(Hierarchy.ROOM)[1], Viz.graph_mat())] +
        [MapViz(o, Viz.pcd_mat()) for o in topometric_map.to_o3d(Hierarchy.ROOM)[2]],
    ])
    
    topometric_map.draw_graph('graph_topology.png')

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
    # Create new voxel map with only cells that did not have any neighbours in kernel
    kernel = VoxelGrid.stick_kernel(voxel_size / kernel_scale)
    floor_voxel_map = voxel_map.filter_gpu_kernel_nbs(kernel)

    # Dilate floor area upwards to connect stairs and convert to nb6 connected 'dense' graph
    dilation_kernel = VoxelGrid.cylinder(
        1, 1 + int(6 // (voxel_size / kernel_scale)))
    traversable_volume_voxel = floor_voxel_map.dilate(dilation_kernel)
    traversable_volume_voxel = traversable_volume_voxel.dilate(VoxelGrid.nb4())
    traversable_volume_graph = traversable_volume_voxel.to_graph()

    # Find largest connected component of traversable volume
    components = traversable_volume_graph.connected_components()
    largest_component = traversable_volume_graph.graph.subgraph(components[0])
    floor_graph = SpatialGraph(
        traversable_volume_graph.scale,
        traversable_volume_graph.origin,
        largest_component)

    return floor_graph


def detect_storeys(floor_voxel_grid: VoxelGrid, voxel_grid: VoxelGrid, buffer: int):
    y_peaks = floor_voxel_grid.detect_peaks(axis=1, height=500)

    voxel_grid.colorize([0, 0, 0])

    for i, peak in enumerate(y_peaks):
        next_i = i + 1
        if next_i == len(y_peaks):
            next_peak = math.inf
        else:
            next_peak = y_peaks[next_i]

        color = random_color()
        color_floor = random_color()
        color_stairs = random_color()

        for v in voxel_grid.filter(lambda v: peak - buffer <= v[1] < next_peak + buffer):
            voxel_grid[v]['color'] = color
            voxel_grid[v]['storey'] = i

        for v in floor_voxel_grid.filter(lambda v: peak - buffer <= v[1] < next_peak + buffer):
            voxel_grid[v]['color'] = color_floor

            voxel_grid[v]['stairs'] = abs(v[1] - (peak - buffer)) > 5
            if voxel_grid[v]['stairs']:
                voxel_grid[v]['color'] = color_stairs

    storeys = voxel_grid.split_by_attr('storey')
    stairs = [storey.get_by_attr('stairs', True) for storey in storeys]

    connections = set()
    nb_kernel = VoxelGrid.nb6()

    for stair in stairs:
        for v in stair:
            nbs = voxel_grid.get_kernel(v, nb_kernel)
            for nb in nbs:
                if voxel_grid[nb]['storey'] != voxel_grid[v]['storey']:
                    connections.add(
                        (voxel_grid[nb]['storey'], voxel_grid[v]['storey']))
                    connections.add(
                        (voxel_grid[v]['storey'], voxel_grid[nb]['storey']))

    return storeys, connections


def skeletonize_floor_area(floor: VoxelGrid, n_target: int, betweenness_threshold: float, path_height: float) -> VoxelGrid:
    """_summary_

    Args:
        traversable_volume (SpatialGraphRepresentation): _description_
        n_target (int): _description_
        betweenness_threshold (float): _description_

    Returns:
        VoxelRepresentation: _description_
    """
    
    skeleton = floor.grass_fire_thinning()  
    skeleton.origin += np.array([0, path_height, 0.])
    
    return skeleton


def cast_isovists(origins: List[Tuple], map_voxel: VoxelGrid, subsample: float, max_dist: float):
    isovists = []

    for v in origins:
        if random() < subsample:
            isovist = map_voxel.visibility(v, max_dist)
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
    G.remove_edges_from((e for e, w in edge_weights.items() if w > weight_threshold))
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

    clustered_map = map_segmented.subset(lambda v: VoxelGrid.cluster_attr in map_segmented[v])
    clustered_map.remove_attr('clusters')

    return clustered_map


def traversability_graph(map_segmented: VoxelGrid, nav_graph: SpatialGraph, floor_voxels) -> SpatialGraph:
    unique_clusters = np.unique(
        list(map_segmented.list_attr(VoxelGrid.cluster_attr)))
    G = networkx.Graph()

    for cluster in unique_clusters:
        cluster_voxels = list(map_segmented.get_by_attr(VoxelGrid.cluster_attr, cluster))
        voxel_coordinates = [map_segmented.voxel_coordinates(v) for v in cluster_voxels]

        if voxel_coordinates:
            voxel_centroid = np.mean(voxel_coordinates, axis=0)
            voxel_centroid_index = tuple(np.mean(voxel_coordinates, axis=0))

            G.add_node(voxel_centroid_index)
            G.nodes[voxel_centroid_index]['pos'] = voxel_centroid
            G.nodes[voxel_centroid_index][VoxelGrid.cluster_attr] = cluster
            G.nodes[voxel_centroid_index]['geometry'] = map_segmented.subset(lambda v: v in cluster_voxels)

    kernel = VoxelGrid.sphere(3)
    cluster_borders = map_segmented.attr_borders(VoxelGrid.cluster_attr, kernel)

    for v in cluster_borders.voxels:
        if floor_voxels.contains_point(cluster_borders.voxel_coordinates(v)):
            v_cluster = cluster_borders[v][VoxelGrid.cluster_attr]
            v_node = [x for x, y in G.nodes(
                data=True) if y[VoxelGrid.cluster_attr] == v_cluster][0]
            
            v_nbs = cluster_borders.get_kernel(v, kernel)
            for v_nb in v_nbs:
                if floor_voxels.contains_point(cluster_borders.voxel_coordinates(v_nb)):
                    v_nb_cluster = cluster_borders[v_nb][VoxelGrid.cluster_attr]
                    v_nb_node = [x for x, y in G.nodes(
                        data=True) if y[VoxelGrid.cluster_attr] == v_nb_cluster][0]

                    G.add_edge(v_node, v_nb_node)

    return SpatialGraph(np.array([1, 1, 1]), np.array([0, 0, 0]), G)
