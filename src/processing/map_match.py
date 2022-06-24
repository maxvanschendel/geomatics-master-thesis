from __future__ import annotations
from itertools import combinations


from karateclub import (AE, BANE, FGSD, FSCNMF, IGE, LDP, MUSAE, SINE,
                        FeatherGraph, FeatherNode, GeoScattering, GL2Vec,
                        Graph2Vec, NetLSD, WaveletCharacteristic, TENE, TADW)
from matplotlib import backend_bases
from torch import nonzero
from evaluation.map_match_performance import analyse_match_performance

from model.topometric_map import *
from sklearn.metrics.pairwise import euclidean_distances
from utils.array import one_to_one
from utils.visualization import visualize_matches
from scipy.optimize import linear_sum_assignment

ptnet_model_fn = './learning3d/pretrained/exp_classifier/models/best_ptnet_model.t7'
dgcnn_model_fn = './dgcnn/pretrained/model.cls.2048.t7'


def grow_hypotheses(map_a: TopometricMap, map_b: TopometricMap, cost_matrix: np.array):
    
    
    
    
    cost_threshold = 20
    edge_threshold = (0.5, 2)

    def edge_compat(edge_a, edge_b):
        len_a = np.linalg.norm(
            edge_a[0].geometry.centroid() - edge_a[1].geometry.centroid())
        len_b = np.linalg.norm(
            edge_b[0].geometry.centroid() - edge_b[1].geometry.centroid())

        return edge_threshold[0] < len_a / len_b < edge_threshold[1]

    def compatible_edges(node_a, node_b):
        edges_a, edges_b = map_a.incident_edges(node_a, data=False), map_b.incident_edges(node_b, data=False)
        compatible_edges = set()

        for edge_a, edge_b in product(edges_a, edges_b):
            if edge_compat(edge_a, edge_b):
                compatible_edges.add((edge_a, edge_b))

        # To make set hashable it must be immutable
        return frozenset(compatible_edges)

    def node_compat(node_a, node_b):
        node_a_index = map_a.node_index(node_a)
        node_b_index = map_b.node_index(node_b)

        cost_ab = cost_matrix[node_a_index, node_b_index]

        return cost_ab < cost_threshold

    
    compatible_nodes = [(list(map_a.nodes(data=False))[i], list(map_b.nodes(data=False))[j]) for i, j in np.ndindex(cost_matrix.shape)]
    compatible_nodes = [(a, b) for a, b in compatible_nodes if node_compat(a, b)]
    
    valid_hypotheses = set()  # H
    potential_matches = {(a, b, compatible_edges(a, b)) for a, b in compatible_nodes}  # M

    while len(potential_matches):
        pairs = {potential_matches.pop()}  # P
        hypothesis = set()  # Q

        while len(pairs):
            p = pairs.pop()
            
            # compatible edges of vertex pair
            _, _, e = p

            # add vertex pair to hypothesis
            hypothesis.add(p)

            # for every compatible edges of both vertices in the pair
            for e_a, e_b in e:
                # get the target vertex
                _, t_a = e_a
                _, t_b = e_b

                # check if new vertex pair hasn't already been visited
                e_tab = compatible_edges(t_a, t_b)
                
                if (t_a, t_b, e_tab) in pairs | hypothesis:
                    continue

                # if nodes are compatible 
                if (t_a, t_b, e_tab) in potential_matches and edge_compat(e_a, e_b):
                    potential_matches.remove((t_a, t_b, e_tab))
                    pairs.add((t_a, t_b, e_tab))

        if len(hypothesis):
            valid_hypotheses.add(frozenset(hypothesis))

    return list(valid_hypotheses)


def pointnet(pcd, dim):
    import torch
    from learning3d.models import PointNet

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
    from torch import from_numpy
    import networkx as nx

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


def match(maps: List[TopometricMap], node_model=None, **kwargs):
    from networkx.algorithms import isomorphism
    
    features = {map: feature_embedding(map, node_model, 1024) for map in maps}

    matches = dict()
    for map_a, map_b in combinations(maps, 2):
        if map_a != map_b:
            matches[(map_a, map_b)] = {}

            f_a, f_b = features[map_a], features[map_b]
            rooms_a, rooms_b = map_a.get_node_level(), map_b.get_node_level()

            # compute distance between node features
            cost_matrix = euclidean_distances(f_a, f_b)
            
            for i, n in enumerate(rooms_a):
                map_a.graph.nodes[n]['feature'] = f_a[i]
                
            for i, n in enumerate(rooms_b):
                map_b.graph.nodes[n]['feature'] = f_b[i]
            
            gm = isomorphism.ISMAGS(map_a.graph, map_b.graph, node_match=lambda a, b: np.linalg.norm(a['feature'] - b['feature']) < 20) 
            largest_common_subgraph = list(gm.largest_common_subgraph())[0]
    
    
            # # use Hungarian method to assign nodes from map a to map b with least cost
            # i, j = linear_sum_assignment(cost_matrix)

            # bipartite_matching = list(zip(list(i), list(j)))

            # hypotheses = grow_hypotheses(map_a, map_b, cost_matrix,)
            
            # hypothesis_sizes = [len(h) for h in hypotheses]
            # largest_hypothesis_index = np.argmax(hypothesis_sizes)
            
            # largest_hypothesis = hypotheses[largest_hypothesis_index]
            

            for node_a, node_b in largest_common_subgraph.items():
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
