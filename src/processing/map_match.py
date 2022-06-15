from __future__ import annotations
from itertools import combinations


from karateclub import (AE, BANE, FGSD, FSCNMF, IGE, LDP, MUSAE, SINE,
                        FeatherGraph, FeatherNode, GeoScattering, GL2Vec,
                        Graph2Vec, NetLSD, WaveletCharacteristic)

from model.topometric_map import *
from sklearn.metrics.pairwise import euclidean_distances
from utils.visualization import visualize_matches

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


def dgcnn(pcd, dim=2048, k=20, dropout=0.5):
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

    # get global feature as 1D numpy array
    embed = model(pcd_torch).cpu().detach() \
                            .numpy() \
                            .squeeze()

    return embed


def engineered_features():
    pass


def graph_features():
    pass


def feature_embedding(map: TopometricMap, node_model, embed_dim=1024) -> np.array:
    rooms = map.get_node_level(Hierarchy.ROOM)
    room_subgraph = networkx.convert_node_labels_to_integers(
        map.graph.subgraph(rooms))

    room_geometry = [r.geometry.to_pcd().points for r in rooms]
    # embedding = [pointnet(g, embed_dim) for g in room_geometry]
    embedding = [dgcnn(g, embed_dim) for g in room_geometry]

    if node_model:
        node_model = node_model()
        node_model.fit(room_subgraph, torch.from_numpy(np.array(embedding)))
        embedding = node_model.get_embedding()

    return embedding


def one_to_one(distance: np.array):
    matches = [(distance[x, y], (x, y)) for x, y in np.ndindex((distance.shape))]

    one_to_one = []
    matched = set()

    for d, m in sorted(matches):
        node_a, node_b = m

        if node_a not in matched and node_b not in matched:
            one_to_one.append((d, m))

            matched.add(node_a)
            matched.add(node_b)

    return one_to_one


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
            o2o_mapping = one_to_one(distance_matrix)
            
            for d, m in o2o_mapping:
                node_a = rooms_a[m[0]]
                node_b = rooms_b[m[1]]
                node_matches[(map_a, map_b)][(node_a, node_b)] = d

            # if kwargs['visualize']:
            visualize_matches(map_a, map_b, [m for _, m in o2o_mapping])

    return node_matches
