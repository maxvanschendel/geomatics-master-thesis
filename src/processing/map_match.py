from __future__ import annotations
from itertools import combinations

import torch
from karateclub import (AE, BANE, FGSD, FSCNMF, IGE, LDP, MUSAE, SINE,
                        FeatherGraph, FeatherNode, GeoScattering, GL2Vec,
                        Graph2Vec, NetLSD, WaveletCharacteristic)

from model.topometric_map import *
from sklearn.metrics.pairwise import euclidean_distances
from utils.visualization import visualize_matches


def dgcnn(pcd, dim):
    from learning3d.models import DGCNN
    
    dgcnn = DGCNN(emb_dims=dim, input_shape='bnc')
    dgcnn_embed = dgcnn(torch.from_numpy(
        pcd.reshape((1, pcd.shape[0], pcd.shape[1]))).float())
    dgcnn_embed = dgcnn_embed.detach().numpy()[0, :, :].T

    return dgcnn_embed


def feature_embedding(map: TopometricMap, node_model, embed_dim=256) -> np.array:
    rooms = map.get_node_level(Hierarchy.ROOM)
    geometry_embedding = {r: dgcnn(r.geometry.to_pcd().points, embed_dim) for r in rooms}
    geometry_embedding = {r: np.linalg.eigvals(p[:embed_dim]) for r, p in geometry_embedding.items()}

    # node_embedding = [r[:embed_dim] for r in geometry_embedding]

    # if node_model is not None:
    #     node_model = node_model()
    #     room_subgraph = networkx.convert_node_labels_to_integers(map.graph.subgraph(rooms))
    #     node_model.fit(room_subgraph, torch.from_numpy(node_embedding))
    #     node_embedding = node_model.get_embedding()

    return geometry_embedding


def match(maps: List[TopometricMap], node_model=None, **kwargs):
    features = {map: feature_embedding(map, node_model) for map in maps}
        
    node_matches = dict()
    for map_a, map_b in combinations(maps, 2):
        if map_a != map_b:
            f_a, f_b = features[map_a], features[map_b]
            
            print(f_a, f_b)
            print(f_a.shape, f_b.shape)
            
            # Find n room pairs with highest similarity
            distance_matrix = euclidean_distances(np.array(f_a.values()), np.array(f_b.values()))
            matches = [(x, y) for x in range(distance_matrix.shape[0]) for y in range(distance_matrix.shape[1])]
            embedding_dist = [distance_matrix[i] for i in matches]
            
            node_matches[(map_a, map_b)] = sorted(zip(embedding_dist, matches))[:3]
            
            # if kwargs['visualize']:
            visualize_matches(map_a, map_b, matches)
        
    return node_matches
