from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import List
import numpy as np
import skopt
import markov_clustering as mc

from processing.parameters import MapExtractionParameters

from model.topometric_map import *
from model.voxel_grid import *
from model.spatial_graph import *




def extract_map(partial_map_pcd: model.point_cloud.PointCloud, p: MapExtractionParameters) -> HierarchicalTopometricMap:
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
    nav_volume_voxel, floor_voxel = segment_floor_area(building_voxels, p.kernel_scale, p.leaf_voxel_size)
    floor_voxel = deepcopy(floor_voxel)

    print("- Segmenting storeys")
    # Split building into multiple storeys and determine their adjacency
    storeys, storey_adjacency = segment_storeys(floor_voxel, building_voxels, buffer=p.storey_buffer, height=p.storey_height)

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
        storey_geometry = storey.geometry.level_of_detail(p.segmentation_lod)

        print(f'    - Casting isovists')
        origin_voxels = isovist_positions.range_search(storey_geometry.bbox())
        origins = [isovist_positions.voxel_centroid(v) for v in origin_voxels]

        isovists = cast_isovists(
            origins=origins,
            map_voxel=storey_geometry,
            subsample=p.isovist_subsample,
            max_dist=p.isovist_range)
        if not isovists:
            continue
        
        print(f'    - Clustering {len(isovists)} isovists')
        mutual_visibility = mutual_visibility_graph(isovists)
        clustering = cluster_graph_mcl(
            distance_matrix=mutual_visibility,
            weight_threshold=p.weight_threshold,
            min_inflation=p.min_inflation,
            max_inflation=p.max_inflation
        )

        print(f"    - Segmenting rooms")
        map_rooms = room_segmentation(
            isovists=isovists,
            map_voxel=storey_geometry,
            clustering=clustering)

        print(f"    - Propagating labels")
        map_rooms = map_rooms.propagate_attr(
            attr=VoxelGrid.cluster_attr,
            prop_kernel=Kernel.sphere(r=2), max_its=10)

        print(f"    - Finding connected clusters")
        map_rooms_split = map_rooms.split_by_attr(VoxelGrid.cluster_attr)

        n_cluster = 0
        connection_kernel = Kernel.sphere(r=2)
        connected_clusters = VoxelGrid(map_rooms.cell_size, map_rooms.origin)

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
            floor_voxels=nav_volume_voxel,
            min_voxels=p.min_voxels)

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
    candidate_voxels = voxel_map.filter_gpu_kernel_nbs(kernel)

    # Dilate floor area upwards to connect stairs and convert to nb6 connected 'dense' graph
    dilation_kernel = Kernel.cylinder(1, 1 + int(6 // (voxel_size / kernel_scale)))
    nav_volume = candidate_voxels.dilate(dilation_kernel)

    # Find largest connected component of traversable volume
    largest_nav_volume_component = nav_volume.connected_components(Kernel.nb6())[0]
    nav_surface = candidate_voxels.subset(lambda v: v in largest_nav_volume_component)
    
    return largest_nav_volume_component, nav_surface


def segment_storeys(floor_voxel_grid: VoxelGrid, voxel_grid: VoxelGrid, buffer: int, height: int = 500):
    height_peaks = floor_voxel_grid.detect_peaks(axis=1, height=height)


    for i, peak in enumerate(height_peaks):
        next_i = i + 1
        next_peak = height_peaks[next_i] if next_i < len(height_peaks) else math.inf
        storey_voxels = voxel_grid.filter(lambda v: peak-buffer <= v[1] < next_peak+buffer)

        for vox in storey_voxels:
            voxel_grid[vox]['storey'] = i
            if vox in floor_voxel_grid:
                voxel_is_stair =  abs(vox[1] - (peak - buffer)) > 5
                voxel_grid[vox]['stairs'] = voxel_is_stair

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


def local_distance_field_maxima(vg, radius, min=0) -> VoxelGrid:
    distance_field = vg.distance_field()

    local_maxima = set()
    for vx, vx_dist in distance_field.items():
        vx_nbs = vg.radius_search(vg.voxel_centroid(vx), radius)
        vx_nbs_dist = [distance_field[nb] for nb in vx_nbs]

        if vx_dist >= max(vx_nbs_dist) and vx_dist >= min:
            local_maxima.add(vx)

    return vg.subset(lambda v: v in local_maxima)


def optimal_isovist_positions(floor: VoxelGrid, path_height: Tuple[float, float], kernel_radius=7, min_boundary_dist=1) -> VoxelGrid:
    skeleton = local_distance_field_maxima(floor, kernel_radius, min_boundary_dist)
    skeleton.origin += np.array([0, path_height[0] + (random()*(path_height[1]-path_height[0])) , 0.])

    return skeleton


def cast_isovists(origins: List[Tuple], map_voxel: VoxelGrid, subsample: float, max_dist: float):
    isovists = []

    for v in origins:
        if random() < subsample:
            isovist = map_voxel.visibility(v, max_dist)
            isovists.append(isovist)

    return isovists


def mutual_visibility_graph(isovists) -> np.array:
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


def cluster_graph_mcl(distance_matrix, weight_threshold: float, min_inflation: float, max_inflation: float) -> List[int]:
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
            most_common_cluster = map_segmented[v]['clusters'].most_common(1)[0]
            if most_common_cluster[1] > min_observations:
                map_segmented[v][VoxelGrid.cluster_attr] = most_common_cluster[0]

    clustered_map = map_segmented.subset(
        lambda v: VoxelGrid.cluster_attr in map_segmented[v])
    return clustered_map


def traversability_graph(map_segmented: VoxelGrid, nav_graph: SpatialGraph, floor_voxels: VoxelGrid, min_voxels: float=100) -> SpatialGraph:
    unique_clusters = np.unique(
        list(map_segmented.list_attr(VoxelGrid.cluster_attr)))
    G = networkx.Graph()

    for cluster in unique_clusters:
        cluster_voxels = list(map_segmented.get_attr(
            VoxelGrid.cluster_attr, cluster))
        voxel_coordinates = [
            map_segmented.voxel_centroid(v) for v in cluster_voxels]

        if voxel_coordinates:
            voxel_centroid = np.mean(voxel_coordinates, axis=0)
            voxel_centroid_index = tuple(np.mean(voxel_coordinates, axis=0))

            G.add_node(voxel_centroid_index)
            G.nodes[voxel_centroid_index]['pos'] = voxel_centroid
            G.nodes[voxel_centroid_index][VoxelGrid.cluster_attr] = cluster
            G.nodes[voxel_centroid_index]['geometry'] = map_segmented.subset(lambda v: v in cluster_voxels)

    kernel = Kernel.sphere(2)
    cluster_borders = map_segmented.attr_borders( VoxelGrid.cluster_attr, kernel)

    for v in cluster_borders.voxels:
        if floor_voxels.contains_point(cluster_borders.voxel_centroid(v)):
            v_cluster = cluster_borders[v][VoxelGrid.cluster_attr]
            v_node = [x for x, y in G.nodes(
                data=True) if y[VoxelGrid.cluster_attr] == v_cluster][0]

            v_nbs = cluster_borders.get_kernel(v, kernel)
            for v_nb in v_nbs:
                if floor_voxels.contains_point(cluster_borders.voxel_centroid(v_nb)):
                    v_nb_cluster = cluster_borders[v_nb][VoxelGrid.cluster_attr]
                    v_nb_node = [x for x, y in G.nodes(
                        data=True) if y[VoxelGrid.cluster_attr] == v_nb_cluster][0]

                    G.add_edge(v_node, v_nb_node)

    connected_nodes = [n for (n, d) in G.nodes(data=True) if len(d["geometry"].voxels) > min_voxels]
    return SpatialGraph(np.array([1, 1, 1]), np.array([0, 0, 0]), G.subgraph(connected_nodes))
