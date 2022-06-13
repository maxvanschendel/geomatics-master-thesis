from __future__ import annotations
from itertools import combinations

import torch
from karateclub import (AE, BANE, FGSD, FSCNMF, IGE, LDP, MUSAE, SINE,
                        FeatherGraph, FeatherNode, GeoScattering, GL2Vec,
                        Graph2Vec, NetLSD, WaveletCharacteristic)

from model.topometric_map import *
from sklearn.metrics.pairwise import euclidean_distances
from utils.visualization import visualize_matches


def pointnet(pcd, dim):
    from learning3d.models import PointNet

    model = PointNet(emb_dims=dim, input_shape='bnc', use_bn=True)
    model.load_state_dict(torch.load('./learning3d/pretrained/exp_classifier/models/best_ptnet_model.t7', map_location=torch.device('cpu')))
    
    pcd_reshape = pcd.reshape((1, pcd.shape[0], pcd.shape[1]))
    pcd_torch = torch.from_numpy(pcd_reshape).float()
    
    embed = model(pcd_torch)
    embed = embed.detach().numpy()
    
    return embed.squeeze()


def feature_embedding(map: TopometricMap, node_model, embed_dim=1024) -> np.array:
    rooms = map.get_node_level(Hierarchy.ROOM)
    room_subgraph = networkx.convert_node_labels_to_integers(map.graph.subgraph(rooms))
    
    room_geometry = [r.geometry.to_pcd().points for r in rooms]
    geometry_embedding = [pointnet(g, embed_dim) for g in room_geometry]   
    
    node_model = node_model()
    node_model.fit(room_subgraph, torch.from_numpy(np.array(geometry_embedding)))
    node_embedding = node_model.get_embedding()

    return node_embedding


def match(maps: List[TopometricMap], node_model=FeatherNode, **kwargs):
    features = {map: feature_embedding(map, FeatherNode, 1024) for map in maps}
        
    node_matches = dict()
    for map_a, map_b in combinations(maps, 2):
        if map_a != map_b:
            f_a, f_b = features[map_a], features[map_b]
            
            # Find n room pairs with highest similarity
            distance_matrix = euclidean_distances(f_a, f_b)
            matches = [(x, y) for x in range(distance_matrix.shape[0]) for y in range(distance_matrix.shape[1])]
            embedding_dist = [distance_matrix[i] for i in matches]
            
            sorted_matches = sorted(zip(embedding_dist, matches))
            node_matches[(map_a, map_b)] = sorted_matches
            
            print(sorted_matches)
            
            # if kwargs['visualize']:
            visualize_matches(map_a, map_b, [m for _, m in sorted_matches[:1]])
        
    return node_matches
