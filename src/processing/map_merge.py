from __future__ import annotations
import itertools
from sklearn import cluster

from sklearn.metrics.pairwise import euclidean_distances
from analysis.visualizer import visualize_matches
from model.topometric_map import *
from yaml import dump, load, Loader
from karateclub import FeatherGraph, IGE,  NetLSD,  GeoScattering,  WaveletCharacteristic, Graph2Vec, FeatherNode, MUSAE, AE, SINE, BANE, FSCNMF, LDP, GL2Vec, FGSD
from learning3d.models import PointNet, DGCNN, PPFNet

@dataclass(frozen=True)
class MapMergeParameters:
    class MapMergeParametersException(Exception):
        pass
    
    @staticmethod
    def deserialize(data: str) -> MapMergeParameters:
        return load(data, Loader)

    @staticmethod
    def read(fn: str) -> MapMergeParameters:
        with open(fn, "r") as read_file:
            file_contents = read_file.read()
        return MapMergeParameters.deserialize(file_contents)

    def serialize(self) -> str:
        return dump(self)

    def write(self, fn: str) -> None:
        with open(fn, "w+") as write_file:
            write_file.write(self.serialize())
            


def n_smallest_indices(input: np.array, n: int):
    smallest_flat = np.argpartition(input.ravel(), n)[:n]
    smallest_indices = [np.unravel_index(
        i, input.shape) for i in smallest_flat]

    return smallest_indices

def codebook(features, n_words: int):
    # https://ai.stackexchange.com/questions/21914/what-are-bag-of-features-in-computer-vision
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_words, random_state=0).fit(features.T)
    centroids = kmeans.cluster_centers_

    return centroids


def bag_of_features(features, codebook: np.array):
    bof = np.zeros(codebook.shape[0])
    for f in features.T:
        centroid_distance = [np.linalg.norm(c - f) for c in codebook]
        nearest_centroid = np.argmin(centroid_distance)
        bof[nearest_centroid] += 1

    normalized_bof = bof / len(features.T)
    return normalized_bof

def dgcnn_global_embedding(pcd, dim):
    import torch
    
    dgcnn =  DGCNN(emb_dims=dim, input_shape='bnc')
    dgcnn_embed = dgcnn(torch.from_numpy(pcd.reshape((1, pcd.shape[0], pcd.shape[1]))).float())
    dgcnn_embed = dgcnn_embed.detach().numpy()[0,:,:].T
    
    print(dgcnn_embed.shape)
    return dgcnn_embed
    
def attributed_graph_embedding(map: HierarchicalTopometricMap, geometry_model, node_model) -> np.array:
    import torch
    
    embed_dim = 256
    pca_dim = 256
    
    
    rooms = map.get_node_level(Hierarchy.ROOM)
    raw_embed = [dgcnn_global_embedding(r.geometry.to_pcd().points, embed_dim) for r in rooms]
    
    node_embedding = [r[:pca_dim] for r in raw_embed]
    node_embedding = np.vstack([np.sort(np.real(np.linalg.eigvals(e))) for e in node_embedding])

    if node_model is not None:
        node_model = node_model()
        room_subgraph = networkx.convert_node_labels_to_integers(map.graph.subgraph(rooms))
        node_model.fit(room_subgraph, torch.from_numpy(node_embedding))
        node_embedding = node_model.get_embedding()

    return node_embedding


def match_maps(map_a: HierarchicalTopometricMap, map_b: HierarchicalTopometricMap, draw_matches: bool = True):
    m = 2
    geometry_model = FeatherGraph
    node_model = None
    
    rooms_a = map_a.get_node_level(Hierarchy.ROOM)
    rooms_b = map_b.get_node_level(Hierarchy.ROOM)

    # Attributed graph embedding
    a_embed = attributed_graph_embedding(map_a, geometry_model, node_model)
    b_embed = attributed_graph_embedding(map_b, geometry_model, node_model)

    # Find n room pairs with highest similarity
    distance_matrix = euclidean_distances(a_embed, b_embed)
    matches = [(x,y) for x in range(distance_matrix.shape[0]) for y in range(distance_matrix.shape[1])]

    embedding_dist = [distance_matrix[i] for i in matches]
    fitness_sorted_matches = sorted(zip(embedding_dist, matches))
    m_best_matches = [match for _, match in fitness_sorted_matches[:m]]
    
    print(f'Attributed graph distance: {list(zip(embedding_dist, matches))}')
    print(f'Sorted combined matches: {fitness_sorted_matches}')
    print(f'Best {m} matches: {m_best_matches}')
    
    if draw_matches:
        visualize_matches(map_a, map_b, m_best_matches)
    return [(rooms_a[a], rooms_b[b]) for a, b in m_best_matches]


# Iterative closest point algorithm
def dense_registration(map_a: VoxelGrid, map_b: VoxelGrid) -> np.array:
    from processing.registration import registration
    
    a_pcd = map_a.to_pcd()
    b_pcd = map_b.to_pcd()
    
    linear_transformation = registration(a_pcd, b_pcd, voxel_size=map_a.cell_size)
    return linear_transformation


def cluster_transform_hypotheses(transforms):
    print(transforms)
    
    distance_matrix = np.zeros(((len(transforms), len(transforms))))
    for i in range(len(transforms)):
        for j in range(len(transforms)):
            i_t, j_t = transforms[i].transformation, transforms[j].transformation
            
            dist =  np.linalg.norm(i_t - j_t)
            distance_matrix[i][j] = dist
            
    clustering = cluster.OPTICS(max_eps=2, min_samples=1, metric='precomputed').fit(distance_matrix)
    return clustering.labels_

def merge_transforms(transforms):
    return sum(transforms) / len(transforms)

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
    match_transforms = [dense_registration(a.geometry, b.geometry) for a, b in matches]

    # Cluster similar transforms into transform hypotheses
    transform_clusters = cluster_transform_hypotheses(match_transforms)
    unique_transform_hypotheses = np.unique(transform_clusters)
    
    max_cluster = np.max(unique_transform_hypotheses)
    for i, l in enumerate(transform_clusters):
        if l == -1:
            transform_clusters[i] = max_cluster + (i + 1)

    for c in unique_transform_hypotheses:
        c_i = np.argwhere(transform_clusters == c)
        cluster_transform = merge_transforms(transform_clusters[c_i])
        
        map_b_transformed = map_b.transform(cluster_transform)
            
            