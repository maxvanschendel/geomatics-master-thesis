from sklearn.metrics.pairwise import euclidean_distances
from analysis.visualizer import visualize_matches
from model.topometric_map import *
from MUSAE.musae import embed_attributed_graph, MUSAEParameters


def fpfh(voxel_grid):
    voxel_size = voxel_grid.cell_size
    pcd = voxel_grid.to_pcd().to_o3d()

    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 3

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh.data


def codebook(data: List[VoxelGrid], n_words: int):
    # https://ai.stackexchange.com/questions/21914/what-are-bag-of-features-in-computer-vision
    from sklearn.cluster import KMeans

    features = np.hstack([fpfh(vg) for vg in data])
    kmeans = KMeans(n_clusters=n_words, random_state=0).fit(features.T)
    centroids = kmeans.cluster_centers_

    return centroids


def bag_of_features(voxel_grid: VoxelGrid, codebook: np.array):
    features = fpfh(voxel_grid)
    bof = np.zeros(codebook.shape[0])
    for f in features.T:
        centroid_distance = [np.linalg.norm(c - f) for c in codebook]
        nearest_centroid = np.argmin(centroid_distance)
        bof[nearest_centroid] += 1

    normalized_bof = bof / len(features.T)
    return normalized_bof


def n_smallest_indices(input: np.array, n: int):
    smallest_flat = np.argpartition(input.ravel(), n)[:n]
    smallest_indices = [np.unravel_index(
        i, input.shape) for i in smallest_flat]

    return smallest_indices


def attributed_graph_embedding(map: HierarchicalTopometricMap, geometry_model, node_model) -> np.array:
    rooms = map.get_node_level(Hierarchy.ROOM)
    geometry_model = geometry_model()
    geometry_model.fit([a.geometry.to_nx(Kernel.nb6()) for a in rooms])
    node_embedding = geometry_model.get_embedding()

    if node_model is not None:
        node_model = node_model()
        node_model.fit(map.graph.subgraph(rooms), node_embedding)
        node_embedding = model.get_embedding()

    return node_embedding


def match_maps(map_a: HierarchicalTopometricMap, map_b: HierarchicalTopometricMap):
    from karateclub import FeatherGraph, IGE,  NetLSD,  GeoScattering,  WaveletCharacteristic, Graph2Vec, FeatherNode

    n = 1
    draw_matches = True
    geometry_model = FeatherGraph
    node_model = FeatherNode

    # Attributed graph embedding
    a_embed = attributed_graph_embedding(map_a, geometry_model, node_model)
    b_embed = attributed_graph_embedding(map_b, geometry_model, node_model)

    # Find n room pairs with highest similarity
    distance_matrix = euclidean_distances(a_embed, b_embed)
    matches = n_smallest_indices(distance_matrix, n)

    print(f"Identified matches: {matches}")
    if draw_matches:
        visualize_matches(map_a, map_b, matches)

    return matches


def dense_registration(map_a: VoxelGrid, map_b: VoxelGrid) -> np.array:
    pass


def cluster_transform_hypotheses(transforms):
    pass


def evaluate_merge_hypothesis(map_a, map_b, transform):
    pass


def merge_maps(map_a: HierarchicalTopometricMap, map_b: HierarchicalTopometricMap, 
               matches: List[Tuple[TopometricNode, TopometricNode]]) -> HierarchicalTopometricMap:
    
    """ 
    From a set of matches between nodes in topometric maps, 
    identify best transform to bring maps into alignment and fuse them at topological level.

    Returns:
        HierarchicalTopometricMap: Global map which results from merging partial maps.
    """
    
    # Find transform between both maps based on ICP registration between matched spaces
    match_transforms = [dense_registration(
        a.geometry, b.geometry) for a, b in matches]

    # Cluster similar transforms into transform hypotheses
    transform_hypotheses = cluster_transform_hypotheses(match_transforms)

    # Identify which transform hypothesis leads to the most likely configuration
    hypotheses_evaluation = [evaluate_merge_hypothesis(
        map_a, map_b, t) for t in transform_hypotheses]
    best_transform_hypothesis_index = np.argmax(hypotheses_evaluation)
    best_transform_hypothesis = transform_hypotheses[best_transform_hypothesis_index]

    # Merge maps, fuse geometry and topology after applying best transform hypothesis
    merged_map = map_a.merge(map_b, best_transform_hypothesis)
    return merged_map
