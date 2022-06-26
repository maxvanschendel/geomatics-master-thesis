from __future__ import annotations
from collections import defaultdict

from itertools import combinations
from typing import Callable

from karateclub import (AE, BANE, FGSD, FSCNMF, IGE, LDP, MUSAE, SINE, TADW,
                        TENE, FeatherGraph, FeatherNode, GeoScattering, GL2Vec,
                        Graph2Vec, NetLSD, WaveletCharacteristic)

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances

from evaluation.map_match_performance import analyse_match_performance
from model.topometric_map import *
from utils.visualization import visualize_matches


ptnet_model_fn = './learning3d/pretrained/exp_classifier/models/best_ptnet_model.t7'
dgcnn_model_fn = './dgcnn/pretrained/model.cls.2048.t7'


def grow_hypothesis(map_a: TopometricMap, map_b: TopometricMap, start_pair, cost_matrix: np.array):
    hypothesis = set()  # Q
    visited = set()

    pairs = {(start_pair)}
    a_nodes = map_a.get_node_level()
    b_nodes = map_b.get_node_level()

    while len(pairs):
        p_a, p_b = pairs.pop()
        hypothesis.add((p_a, p_b))

        nbs_a, nbs_b = map_a.neighbours(p_a), map_b.neighbours(p_b)

        if nbs_a and nbs_b:
            nbs_a_idx = [map_a.node_index(n)
                         for n in nbs_a if n not in visited]
            nbs_b_idx = [map_b.node_index(n)
                         for n in nbs_b if n not in visited]

            nbs_product = product(nbs_a_idx, nbs_b_idx)

            nbs_cost = np.reshape(
                [cost_matrix[m] for m in nbs_product], (len(nbs_a_idx), len(nbs_b_idx)))

            bipartite_matching = linear_assign(nbs_cost)
            node_matching = [(a_nodes[nbs_a_idx[i]], b_nodes[nbs_b_idx[j]])
                             for i, j in bipartite_matching]

            for m_a, m_b in node_matching:
                pairs.add((m_a, m_b))
                
                visited.add(m_a)
                visited.add(m_b)

    return hypothesis


def pointnet(pcd, dim):
    import torch
    from learning3d.models import PointNet

    # load pointnet model from storage
    model = PointNet(emb_dims=dim, input_shape='bnc', use_bn=True)
    model.load_state_dict(torch.load(
        ptnet_model_fn))

    # reshape to bnc format and create torch tensor
    pcd_reshape = pcd.reshape((1, pcd.shape[0], pcd.shape[1]))
    pcd_torch = torch.from_numpy(pcd_reshape).float()

    # get global feature as 1D numpy array
    embed = model(pcd_torch).cpu().detach() \
                            .numpy() \
                            .squeeze()

    return embed


def dgcnn(pcd, dim, k=75, dropout=0.5):
    import torch
    from dgcnn.model import DGCNN_cls
    from torch.nn import DataParallel

    # load DGCNN model from storage
    device = torch.device("cuda")
    model = DGCNN_cls(k=k, emb_dims=dim, dropout=dropout).to(device)
    model = DataParallel(model)
    model.load_state_dict(torch.load(dgcnn_model_fn))

    # reshape to bnc format and create torch tensor
    pcd_reshape = pcd.reshape((1, pcd.shape[0], pcd.shape[1]))
    pcd_reshape = np.swapaxes(pcd_reshape, 2, 1)
    pcd_torch = torch.from_numpy(pcd_reshape).float()

    # get global feature as numpy array
    embed = model(pcd_torch).cpu().detach() \
                            .numpy() \
                            .squeeze()

    return embed


def engineered_features():
    pass


def graph_features():
    pass


def feature_embedding(map: TopometricMap, node_model, embed_dim) -> np.array:
    from scipy import sparse

    rooms = map.get_node_level()
    room_subgraph = networkx.convert_node_labels_to_integers(
        map.graph.subgraph(rooms))
    room_geometry = [r.geometry.to_pcd().points for r in rooms]

    features_a = [pointnet(g, embed_dim) for g in room_geometry]
    features_b = [dgcnn(g, embed_dim) for g in room_geometry]

    features = np.hstack([features_a, features_b])

    if node_model:
        node_model = node_model()
        node_model.fit(room_subgraph, sparse.coo_matrix(np.array(features)))
        node_features = node_model.get_embedding()

        return node_features

    return features

def linear_assign(m):
    a, b = linear_sum_assignment(m)
    bipartite_matching = list(zip(list(a), list(b)))
    
    return bipartite_matching
    

def match(maps: List[TopometricMap], node_model: Callable = None, **kwargs):
    features = {map: feature_embedding(map, node_model, 1024) for map in maps}

    matches = defaultdict(lambda: {})
    
    for map_a, map_b in combinations(maps, 2):
        if map_a != map_b:
            nodes_a, nodes_b = map_a.get_node_level(), map_b.get_node_level()

            # compute distance in feature space between every pair of nodes
            f_a, f_b = features[map_a], features[map_b]
            
            cost_matrix = euclidean_distances(f_a, f_b)
            bipartite_matching = linear_assign(cost_matrix)
            
            hypotheses = set()
            for a, b in bipartite_matching:

                # get node with maximum similarity
                # max_similar_a, max_similar_b = tuple(np.unravel_index(np.argmin(cost_matrix, axis=None), cost_matrix.shape))
                hypothesis = grow_hypothesis(
                    map_a, map_b,
                    (nodes_a[a], nodes_b[b]),
                    cost_matrix)

                hypotheses.add(frozenset(hypothesis))
                
            size_sorted_hypotheses = sorted([(len(h), h) for h in hypotheses], reverse=True)
            hypothesis = list(size_sorted_hypotheses)[1][1]

            for node_a, node_b in hypothesis:
                a, b = map_a.node_index(node_a), map_b.node_index(node_b)

                # Store the matching between nodes and their cost in cost matrix
                matches[map_a, map_b][node_a, node_b] = cost_matrix[a, b]

    return matches


def match_create(topometric_maps, kwargs):
    logging.info('Matching partial maps')

    matches = match(topometric_maps)
    return matches


def match_write(matches, kwargs):
    raise NotImplementedError("")


def match_read(kwargs):
    raise NotImplementedError("")


def match_visualize(topometric_maps, matches, kwargs):
    for a, b in matches:
        visualize_matches(a, b, matches[a, b].keys())


def match_analyse(truths, matches, topometric_maps, kwargs):
    for a, b in matches.keys():
        ground_truth_a = truths[topometric_maps.index(a)]
        ground_truth_b = truths[topometric_maps.index(b)]

        map_match_perf = analyse_match_performance(
            a, b, ground_truth_a, ground_truth_b, matches[(a, b)])

        logging.info(
            f'Map match performance for partial maps {a} and {b}: {map_match_perf}')
