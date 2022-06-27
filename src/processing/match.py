from __future__ import annotations
from collections import defaultdict

from itertools import combinations
from typing import Callable

from karateclub import (AE, BANE, FGSD, FSCNMF, IGE, LDP, MUSAE, SINE, TADW,
                        TENE, FeatherGraph, FeatherNode, GeoScattering, GL2Vec,
                        Graph2Vec, NetLSD, WaveletCharacteristic)

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances

from model.topometric_map import *
from processing.fuse import cluster_transform
from processing.registration import align_least_squares
from utils.array import replace_with_unique
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

            nbs_cost = np.reshape([cost_matrix[m] for m in nbs_product], (len(nbs_a_idx), len(nbs_b_idx)))
            
            # for a, b in nbs_cost.ndindex():
            #     pass

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


def dgcnn(pcd, dim, k=16, dropout=0.5):
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


def linear_assign(m: np.array):
    a, b = linear_sum_assignment(m)
    bipartite_matching = list(
                            zip(
                                list(a), list(b)
                            )
                        )

    return bipartite_matching


def cluster_hypotheses(hypotheses: List[Set[Tuple[TopometricNode, TopometricNode]]]):
    hypotheses_transforms = []
    for h in hypotheses:
        centroids = [(a.geometry.centroid(), b.geometry.centroid())
                     for a, b in h]
        centroids_a, centroids_b = zip(*centroids)
        lstsq = align_least_squares(centroids_a, centroids_b)
        hypotheses_transforms.append(lstsq)

    # Cluster similar transforms
    # Assign unclustered transforms (label=-1) their own unique cluster
    t_clusters = cluster_transform(
        hypotheses_transforms, max_eps=25, min_samples=1)
    t_clusters = replace_with_unique(t_clusters, -1)

    return t_clusters


def merge_clusters(hypotheses: List[Set[Tuple[TopometricNode, TopometricNode]]], clusters: np.array):
    merged_hypotheses = []
    for cluster in clusters.astype(np.int64):
        cluster_idx = np.argwhere(clusters == cluster)
        cluster_hypotheses = [hypotheses[i] for i in cluster_idx]

        merged_hypothesis = set().union(*cluster_hypotheses)
        merged_hypotheses.append(merged_hypothesis)

    return merged_hypotheses


def grow_hypotheses(map_a, map_b, initial_matches, cost_matrix):
    nodes_a, nodes_b = map_a.get_node_level(), map_b.get_node_level()

    hypotheses = []
    for a, b in initial_matches:
        hypothesis = grow_hypothesis(
            map_a, map_b,
            (nodes_a[a], nodes_b[b]),
            cost_matrix)

        hypotheses.append(hypothesis)

    return hypotheses


def hypothesis_quality(h):
    return len(h)


def match(maps: List[TopometricMap], node_model: Callable = None, **kwargs):
    features = {map: feature_embedding(map, node_model, 1024) for map in maps}

    matches = defaultdict(lambda: {})

    for map_a, map_b in combinations(maps, 2):
        if map_a != map_b:
            # compute distance in feature space between every pair of nodes
            f_a, f_b = features[map_a], features[map_b]

            # create initial pairing by finding least cost assignment between nodes
            cost_matrix = euclidean_distances(f_a, f_b)
            assignment = linear_assign(cost_matrix)

            # from initial pairing, grow 
            hypotheses = grow_hypotheses(map_a, map_b, assignment, cost_matrix)
            # t_clusters = cluster_hypotheses(hypotheses)
            # merged_hypotheses = merge_clusters(hypotheses, t_clusters)

            # _, sorted_hypotheses = sort_by_func(
            #     merged_hypotheses, hypothesis_quality, reverse=True)
            best_hypothesis = hypotheses[0]

            for node_a, node_b in best_hypothesis:
                a, b = map_a.node_index(node_a), map_b.node_index(node_b)

                # Store the matching between nodes and their cost in cost matrix
                matches[map_a, map_b][node_a, node_b] = cost_matrix[a, b]

    return matches


def sort_by_func(collection, func, reverse):
    return list(zip(*sorted([(func(h), h) for h in collection]), reverse=reverse))


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

def analyse_match_performance(partial_a, partial_b, ground_truth_a, ground_truth_b, matches):
    
    a_to_ground_truth = partial_a.match_nodes(ground_truth_a)
    b_to_ground_truth = partial_b.match_nodes(ground_truth_b)
    
    a_matches = {a: v for a, v in a_to_ground_truth.keys()}
    b_matches = {v: b for b, v in b_to_ground_truth.keys()}
    
    
    a_nodes = partial_a.get_node_level()
    b_nodes = partial_b.get_node_level()
        
    target_matches = [(a, b_matches[v]) for a, v in a_matches.items() if v in b_matches]
    target_matches = [(a_nodes[a], b_nodes[b]) for a, b in target_matches]
    
    
    true_positive = [
        True for match in matches if match in target_matches]
    false_positive = [
        True for match in matches if match not in target_matches]
    false_negative = [
        True for match in target_matches if match not in matches]

    # TODO: Validate that this is actually correct
    accuracy = sum(true_positive) / len(matches)
    precision = sum(true_positive) / (sum(true_positive) + sum(false_positive))
    recall = sum(true_positive) / (sum(true_positive) + sum(false_negative))
    
    if precision + recall:
        f_1 = 2 * (precision*recall) / (precision+recall)
    else:
        f_1 = 0

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_1': f_1}