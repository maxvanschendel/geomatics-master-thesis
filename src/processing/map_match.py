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
    # features = [pointnet(g, embed_dim) for g in room_geometry]
    features = [dgcnn(g, embed_dim) for g in room_geometry]

    if node_model:       
        node_model = node_model()
        node_model.fit(room_subgraph, sparse.coo_matrix(np.array(features)))
        node_features = node_model.get_embedding()

        return node_features
    
    return features


def match(maps: List[TopometricMap], node_model=nonzero, **kwargs):
    features = {map: feature_embedding(map, node_model, 1024) for map in maps}

    matches = dict()
    for map_a, map_b in combinations(maps, 2):
        if map_a != map_b:
            matches[(map_a, map_b)] = {}
            
            f_a, f_b = features[map_a], features[map_b]
            rooms_a, rooms_b = map_a.get_node_level(), map_b.get_node_level()

            cost_matrix = euclidean_distances(f_a, f_b)                     # compute distance between node features           
            i, j = linear_sum_assignment(cost_matrix)                       # use Hungarian method to assign nodes from map a to map b with least cost
            
            bipartite_matching = list(zip(list(i), list(j)))
            for a, b in bipartite_matching:
                # Get nodes belonging to indices from matching
                node_a, node_b = rooms_a[a], rooms_b[b]
                
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