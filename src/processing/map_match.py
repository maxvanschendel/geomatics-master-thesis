from __future__ import annotations
from itertools import combinations

import torch
from karateclub import (AE, BANE, FGSD, FSCNMF, IGE, LDP, MUSAE, SINE,
                        FeatherGraph, FeatherNode, GeoScattering, GL2Vec,
                        Graph2Vec, NetLSD, WaveletCharacteristic)

from model.topometric_map import *
from sklearn.metrics.pairwise import euclidean_distances
from utils.visualization import visualize_matches

ptnet_model_fn = './learning3d/pretrained/exp_classifier/models/best_ptnet_model.t7'

def pointnet(pcd, dim):
    from learning3d.models import PointNet

    model = PointNet(emb_dims=dim, input_shape='bnc', use_bn=True)
    model.load_state_dict(torch.load(ptnet_model_fn, map_location=torch.device('cpu')))
    
    # reshape to bnc format and create torch tensor
    pcd_reshape = pcd.reshape((1, pcd.shape[0], pcd.shape[1]))
    pcd_torch = torch.from_numpy(pcd_reshape).float()
    
    # get global feature as 1D numpy array
    embed = model(pcd_torch).detach() \
                            .numpy() \
                            .squeeze()
    
    return embed


def feature_embedding(map: TopometricMap, node_model, embed_dim=1024) -> np.array:
    rooms = map.get_node_level(Hierarchy.ROOM)
    room_subgraph = networkx.convert_node_labels_to_integers(map.graph.subgraph(rooms))
    
    room_geometry = [r.geometry.to_pcd().points for r in rooms]
    embedding = [pointnet(g, embed_dim) for g in room_geometry]   
    
    if node_model:
        node_model = node_model()
        node_model.fit(room_subgraph, torch.from_numpy(np.array(embedding)))
        embedding = node_model.get_embedding()

    return embedding


def match(maps: List[TopometricMap], node_model=FeatherNode, **kwargs):
    features = {map: feature_embedding(map, None, 1024) for map in maps}
        
    node_matches = dict()
    for map_a, map_b in combinations(maps, 2):
        node_matches[(map_a, map_b)] = {}
        
        if map_a != map_b:
            f_a, f_b = features[map_a], features[map_b]
            rooms_a, rooms_b = map_a.get_node_level(Hierarchy.ROOM), map_b.get_node_level(Hierarchy.ROOM)
            
            # Find n room pairs with highest similarity
            distance_matrix = euclidean_distances(f_a, f_b)
            matches = [(x, y) for x in range(distance_matrix.shape[0]) for y in range(distance_matrix.shape[1])]
            embedding_dist = [distance_matrix[i] for i in matches]
            
            sorted_matches = sorted(zip(embedding_dist, matches))
            best_matches = sorted_matches[:3]
            
            for d, m in best_matches:
                node_a = rooms_a[m[0]]
                node_b = rooms_b[m[1]]
                
                node_matches[(map_a, map_b)][(node_a, node_b)] = d
            
            # if kwargs['visualize']:
            visualize_matches(map_a, map_b, [m for _, m in best_matches])
        
    return node_matches
