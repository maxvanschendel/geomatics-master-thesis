from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List
import numpy as np
import skopt
from yaml import dump, load

from analysis.visualizer import MapViz, Viz
from model.topometric_map import *
from model.voxel_grid import *
from model.spatial_graph import *


@dataclass(frozen=True)
class MapExtractionParameters:
    class MapExtractionParametersException(Exception):
        pass

    leaf_voxel_size: float
    traversability_lod: int
    segmentation_lod: int
    
    kernel_scale: float
    isovist_height: float
    isovist_spacing: float

    isovist_subsample: float
    isovist_range: float

    weight_threshold: float
    min_inflation: float
    max_inflation: float

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


def extract_map(partial_map_pcd: model.point_cloud.PointCloud, p: MapExtractionParameters) -> HierarchicalTopometricMap:
    print("Extracting topological-metric map")
    # Map representation that is result of map extraction
    topometric_map = HierarchicalTopometricMap()

    print('- Voxelizing point cloud')
    # Voxelize point cloud partial map at high resolution
    leaf_voxels = partial_map_pcd.voxelize(p.leaf_voxel_size)
    building_voxels = leaf_voxels.level_of_detail(p.traversability_lod)

    # Add building level node to hierarchical topometric map
    building_node = TopometricNode(Hierarchy.BUILDING, building_voxels)
    topometric_map.add_node(building_node)

    print("- Extracting traversable volume")
    # Extract the traversable volume and get building voxels at its bottom (the floor)
    nav_volume_voxel = segment_floor_area(building_voxels, p.kernel_scale, p.leaf_voxel_size)
    floor_voxel = building_voxels.subset(lambda v: v in nav_volume_voxel)

    print("- Segmenting storeys")
    # Split building into multiple storeys and determine their adjacency
    storeys, storey_adjacency = segment_storeys(floor_voxel, building_voxels, buffer=10, height=500)

    # Create a node for each storey in the building and edges for
    # both hierarchy within building and traversability between storeys
    storey_nodes = [TopometricNode(Hierarchy.STOREY, s) for s in storeys]
    storey_hierarchy = [(building_node, s) for s in storey_nodes]
    storey_adjacency = [(storey_nodes[a], storey_nodes[b]) for (a, b) in storey_adjacency]

    # Add storey nodes and edges to hierarchical topometric map
    topometric_map.add_nodes(storey_nodes)
    topometric_map.add_edges(storey_hierarchy, EdgeType.HIERARCHY)
    topometric_map.add_edges(storey_adjacency, EdgeType.TRAVERSABILITY)

    print('- Estimating optimal isovist positions')
    # Attempt to find the positions in the map from which as much of the
    # the map is visible as possible.
    isovist_positions = optimal_isovist_positions(floor_voxel, p.isovist_height, p.isovist_spacing)

    print(f"- Segmenting {len(storey_nodes)} storey(s)")
    for i, storey in enumerate(storey_nodes):
        print(f"- Segmenting storey {i+1}/{len(storey_nodes)}")

        print(f'    - Voxelizing storey at LoD{p.segmentation_lod} ')
        storey_geometry_low = storey.geometry.level_of_detail(p.segmentation_lod)

        storey_centroids = storey_geometry_low.voxel_centroids()
        min_bbox, max_bbox = storey_centroids.min(axis=0), storey_centroids.max(axis=0)

        print(f'    - Casting isovists')
        origins = isovist_positions.voxel_centroids()
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
        prop_kernel = Kernel.sphere(r=4)
        map_rooms = map_rooms.propagate_attr(
            attr=VoxelGrid.cluster_attr,
            prop_kernel=prop_kernel)

        map_rooms_split = map_rooms.split_by_attr(VoxelGrid.cluster_attr)

        n_cluster = 0
        connection_kernel = Kernel.sphere(r=2)
        connected_clusters = VoxelGrid(
            map_rooms.shape, map_rooms.cell_size, map_rooms.origin)

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
            floor_voxels=nav_volume_voxel)

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
            MapViz(floor_voxel.to_o3d(),
                   Viz.pcd_mat()),
            MapViz(isovist_positions.to_o3d(
                has_color=False), Viz.pcd_mat())
        ],

        # Topometric map visualization at room level
        [MapViz(o, Viz.pcd_mat(pt_size=6)) for o in topometric_map.to_o3d(Hierarchy.ROOM)[0]] +
        [MapViz(topometric_map.to_o3d(Hierarchy.ROOM)[1], Viz.graph_mat())] +
        [MapViz(o, Viz.pcd_mat())
         for o in topometric_map.to_o3d(Hierarchy.ROOM)[2]],
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
    kernel = Kernel.stick_kernel(voxel_size / kernel_scale)
    floor_voxel_map = voxel_map.filter_gpu_kernel_nbs(kernel)

    # Dilate floor area upwards to connect stairs and convert to nb6 connected 'dense' graph
    dilation_kernel = Kernel.cylinder(1, 1 + int(6 // (voxel_size / kernel_scale)))
    traversable_volume_voxel = floor_voxel_map.dilate(dilation_kernel)
    traversable_volume_voxel = traversable_volume_voxel.dilate(Kernel.nb4())
    traversable_volume_graph = traversable_volume_voxel.to_graph(Kernel.nb6())

    # Find largest connected component of traversable volume
    components = traversable_volume_graph.connected_components()
    largest_component = traversable_volume_graph.graph.subgraph(components[0])
    floor_graph = SpatialGraph(
        traversable_volume_graph.scale,
        traversable_volume_graph.origin,
        largest_component)

    return floor_graph.to_voxel()


def segment_storeys(floor_voxel_grid: VoxelGrid, voxel_grid: VoxelGrid, buffer: int, height: int = 500):
    height_peaks = floor_voxel_grid.detect_peaks(axis=1, height=height)

    for i, peak in enumerate(height_peaks):
        next_i = i + 1
        next_peak = height_peaks[next_i] if next_i < len(height_peaks) else math.inf
        
        for vox in voxel_grid.filter(lambda v: peak - buffer <= v[1] < next_peak + buffer):
            voxel_grid[vox]['storey'] = i
            
            if vox in floor_voxel_grid:
                voxel_grid[vox]['stairs'] = abs(vox[1] - (peak - buffer)) > 5

    storeys = voxel_grid.split_by_attr('storey')
    stairs = [storey.get_attr('stairs', True) for storey in storeys]

    connections = set()
    connectivity_kernel = Kernel.nb6()

    for stair in stairs:
        for vox in stair:
            nbs = voxel_grid.get_kernel(vox, connectivity_kernel)
            for nb in nbs:
                if voxel_grid[nb]['storey'] != voxel_grid[vox]['storey']:
                    connections.add((voxel_grid[nb]['storey'], voxel_grid[vox]['storey']))
                    connections.add((voxel_grid[vox]['storey'], voxel_grid[nb]['storey']))

    return storeys, connections


def optimal_isovist_positions(floor: VoxelGrid, path_height: float, kernel_radius=7) -> VoxelGrid:
    skeleton = floor.local_distance_field_maxima(kernel_radius)
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
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph import graphviz_layout
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

    plt.clf()
    plt.matshow(distance_matrix)
    plt.title("Mutual visibility")
    plt.colorbar()
    plt.savefig("mutual_visibility_matrix.png")

    plt.clf()

    G = networkx.from_numpy_matrix(distance_matrix)
    edge_weights = networkx.get_edge_attributes(G, 'weight')

    threshold = 0.2
    G.remove_edges_from((e for e, w in edge_weights.items() if w > threshold))

    pos = graphviz_layout(G, "neato")
    networkx.draw(G, pos=pos, node_size=5)
    plt.title(f"Mutual visibility graph with threshold {threshold}")
    plt.savefig("mutual_visibility_graph.png")

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


def traversability_graph(map_segmented: VoxelGrid, nav_graph: SpatialGraph, floor_voxels) -> SpatialGraph:
    unique_clusters = np.unique(
        list(map_segmented.list_attr(VoxelGrid.cluster_attr)))
    G = networkx.Graph()

    for cluster in unique_clusters:
        cluster_voxels = list(map_segmented.get_attr(
            VoxelGrid.cluster_attr, cluster))
        voxel_coordinates = [
            map_segmented.voxel_coordinates(v) for v in cluster_voxels]

        if voxel_coordinates:
            voxel_centroid = np.mean(voxel_coordinates, axis=0)
            voxel_centroid_index = tuple(np.mean(voxel_coordinates, axis=0))

            G.add_node(voxel_centroid_index)
            G.nodes[voxel_centroid_index]['pos'] = voxel_centroid
            G.nodes[voxel_centroid_index][VoxelGrid.cluster_attr] = cluster
            G.nodes[voxel_centroid_index]['geometry'] = map_segmented.subset(
                lambda v: v in cluster_voxels)

    kernel = Kernel.sphere(2)
    cluster_borders = map_segmented.attr_borders(
        VoxelGrid.cluster_attr, kernel)

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

    connected_nodes = [n for (n, d) in G.nodes(
        data=True) if len(d["geometry"].voxels) > 100]
    return SpatialGraph(np.array([1, 1, 1]), np.array([0, 0, 0]), G.subgraph(connected_nodes))
