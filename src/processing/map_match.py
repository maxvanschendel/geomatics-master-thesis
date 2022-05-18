from __future__ import annotations

import torch
from karateclub import (AE, BANE, FGSD, FSCNMF, IGE, LDP, MUSAE, SINE,
                        FeatherGraph, FeatherNode, GeoScattering, GL2Vec,
                        Graph2Vec, NetLSD, WaveletCharacteristic)
from learning3d.models import DGCNN
from model.topometric_map import *
from sklearn.metrics.pairwise import euclidean_distances
from utils.visualization import visualize_matches


def dgcnn(pcd, dim):
    dgcnn = DGCNN(emb_dims=dim, input_shape='bnc')
    dgcnn_embed = dgcnn(torch.from_numpy(
        pcd.reshape((1, pcd.shape[0], pcd.shape[1]))).float())
    dgcnn_embed = dgcnn_embed.detach().numpy()[0, :, :].T

    return dgcnn_embed


def attributed_graph_embedding(map: HierarchicalTopometricMap, node_model, embed_dim=256, pca_dim=256) -> np.array:
    rooms = map.get_node_level(Hierarchy.ROOM)
    raw_embed = [dgcnn(r.geometry.to_pcd().points, embed_dim) for r in rooms]

    node_embedding = [r[:pca_dim] for r in raw_embed]
    node_embedding = np.vstack([np.sort(np.real(np.linalg.eigvals(e))) for e in node_embedding])

    if node_model is not None:
        node_model = node_model()
        room_subgraph = networkx.convert_node_labels_to_integers(map.graph.subgraph(rooms))
        node_model.fit(room_subgraph, torch.from_numpy(node_embedding))
        node_embedding = node_model.get_embedding()

    return node_embedding


def match_maps(map_a: HierarchicalTopometricMap, map_b: HierarchicalTopometricMap, draw_matches: bool = True, m: int = 10):
    node_model = None

    rooms_a = map_a.get_node_level(Hierarchy.ROOM)
    rooms_b = map_b.get_node_level(Hierarchy.ROOM)

    # Attributed graph embedding
    a_embed = attributed_graph_embedding(map_a, node_model)
    b_embed = attributed_graph_embedding(map_b, node_model)

    # Find n room pairs with highest similarity
    distance_matrix = euclidean_distances(a_embed, b_embed)
    matches = [(x, y) for x in range(distance_matrix.shape[0])
                for y in range(distance_matrix.shape[1])]

    embedding_dist = [distance_matrix[i] for i in matches]
    fitness_sorted_matches = sorted(zip(embedding_dist, matches))
    m_best_matches = [match for _, match in fitness_sorted_matches[:m]]

    print(f'Attributed graph distance: {list(zip(embedding_dist, matches))}')
    print(f'Sorted combined matches: {fitness_sorted_matches}')
    print(f'Best {m} matches: {m_best_matches}')

    if draw_matches:
        visualize_matches(map_a, map_b, m_best_matches)
    return [(rooms_a[a], rooms_b[b]) for a, b in m_best_matches]
