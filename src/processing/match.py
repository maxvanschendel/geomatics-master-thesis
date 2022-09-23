from __future__ import annotations
from collections import defaultdict

from itertools import combinations
from tkinter import N
from typing import Callable

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances
from model.point_cloud import PointCloud

from model.topometric_map import *
from processing.fuse import cluster_transform
from processing.registration import align_least_squares, iterative_closest_point
from utils.array import replace_with_unique
from utils.datasets import PartialMap, SimulatedPartialMap
from utils.visualization import visualize_matches, visualize_point_cloud


def product_matrix(func, a, b):
    p = list(product(a, b))
    mat = np.reshape([func(m) for m in p], (len(a), len(b)))

    return mat


def grow_hypothesis(map_a: TopometricMap, map_b: TopometricMap, start_pair, cost_matrix: np.array, d_max: float = 1000000, t_max: float = 2.5, min_similarity: float = 1000000):
    LARGE_NUM = 2**32

    def align_hypothesis(h):
        centroids = [(a.geometry.centroid(), b.geometry.centroid())
                     for a, b in h]
        c_a, c_b = zip(*centroids)

        return align_least_squares(c_a, c_b)

    hypothesis = set()  # Q
    if cost_matrix[map_a.node_index(start_pair[0]), map_b.node_index(start_pair[1])] > min_similarity:
        return hypothesis

    visited = set()

    pairs = {(start_pair)}
    a_nodes = map_a.get_node_level()
    b_nodes = map_b.get_node_level()

    while len(pairs):
        p_a, p_b = pairs.pop()

        hypothesis.add((p_a, p_b))
        visited.add(p_a)
        visited.add(p_b)

        nbs_a, nbs_b = map_a.neighbours(p_a), map_b.neighbours(p_b)

        if nbs_a and nbs_b:
            nbs_a_idx = [map_a.node_index(n)
                         for n in nbs_a if n not in visited]
            nbs_b_idx = [map_b.node_index(n)
                         for n in nbs_b if n not in visited]

            # get cost matrix of only neighbors
            nbs_product = list(
                product(nbs_a_idx, nbs_b_idx)
            )

            nbs_cost = np.reshape(
                [cost_matrix[m] for m in nbs_product],
                (len(nbs_a_idx), len(nbs_b_idx)))

            _, t_h_pre = align_hypothesis(hypothesis)
            for a_idx, b_idx in nbs_product:
                nb_a = a_nodes[a_idx]
                nb_b = b_nodes[b_idx]

                if cost_matrix[a_idx, b_idx] > d_max:
                    nbs_cost[nbs_a_idx.index(a_idx),
                             nbs_b_idx.index(b_idx)] = LARGE_NUM
                    continue

                # add potential
                hypothesis.add((nb_a, nb_b))
                _, t_h_post = align_hypothesis(hypothesis)
                hypothesis.remove((nb_a, nb_b))

                if np.mean(t_h_post) - np.mean(t_h_pre) > t_max:
                    nbs_cost[nbs_a_idx.index(
                        a_idx), nbs_b_idx.index(b_idx)] = LARGE_NUM

            # if all edges are inconsistent, move to next potential match
            if np.all(nbs_cost == LARGE_NUM):
                continue

            # find least cost matching between neighbors
            bipartite_matching = linear_assign(nbs_cost)
            node_matching = [(a_nodes[nbs_a_idx[i]],
                              b_nodes[nbs_b_idx[j]])
                             for i, j in bipartite_matching]

            for m_a, m_b in node_matching:
                pairs.add((m_a, m_b))

    return hypothesis


def lpdnet(pcd, n_points=4096):
    import lpdnet.util.PointNetVlad as PNV
    import torch

    device = torch.device("cuda")

    model = PNV.PointNetVlad(num_points=n_points, featnet='lpdnetorigin',
                             feature_transform=False, xyz_trans=False).to(device)
    model.load_state_dict(torch.load('lpdnet\pretrained\lpdnet.ckpt')[
                          'state_dict'], strict=True)
    model.eval()

    torch.cuda.empty_cache()
    with torch.no_grad():
        np.random.shuffle(pcd)
        pcd = pcd[:n_points, :]
        pcd = pcd.reshape((1, pcd.shape[0], pcd.shape[1]))
        pcd = np.swapaxes(pcd, 2, 1)

        tensor = torch.from_numpy(pcd).float()
        tensor = tensor.to(device)

        embed = model(tensor)

    embed = embed.detach().cpu().numpy().squeeze()
    return embed


def engineered_features(pcd: PointCloud):
    embedding = np.array([
        len(pcd.points),
        pcd.planarity(),
        pcd.volume(),
        pcd.height(),
        pcd.projected_area(),
        pcd.mean_distance_to_centroid(),
        pcd.quotient_of_eigenvalues(),
        pcd.svd()[2],
        pcd.svd()[2] / (pcd.svd()[0] + pcd.svd()[1]),
        pcd.svd()[0] / pcd.svd()[1],
        np.dot(np.array([0, 0, 1]), abs(pcd.pca()[0]
               * pcd.explained_variance_ratio()[0])),
        pcd.roughness(k=32),
    ])

    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def max_pool(ar):
    if len(ar):
        return np.max(ar, axis=0)
    return 0


def embed_geometry(geometry: VoxelGrid, feature: Iterable[str]):
    feature_func = {
        # deep learning
        'deep': lambda n: lpdnet(n.level_of_detail(0).to_pcd().center().normalize(-1, 1).points, 4096),
        'spectral': lambda n: n.level_of_detail(1).shape_dna(Kernel.nb6(), 512),
        'engineered': lambda n: engineered_features(n.to_pcd())
    }

    embedding = np.hstack((feature_func[f](geometry) for f in feature))
    return embedding


def feature_embedding(tmap: TopometricMap, k_max: int, models: List[str], w: float = 1/2, partial_map: SimulatedPartialMap = None) -> np.array:
    def max_pool_nbs(nbs): return max_pool([embed[nb] for nb in nbs])
    def k_weight(k): return w**k
    def graph_conv(n, k=1): return max_pool_nbs(tmap.knbr(n, k))*k_weight(k)

    # geometrical feature embedding
    nodes = tmap.get_node_level()

    print("Embedding 1-step")
    embed = {n: embed_geometry(n.geometry, models) for n in nodes}

    print("Embedding 2-step")
    embed_1step = {n: embed_geometry(VoxelGrid.merge([nbr.geometry for nbr in tmap.knbr(
        n, 1)] + [n.geometry]).level_of_detail(1), models) for n in nodes}

    # print("Embedding 3-step")
    # embed_2step = {n: embed_geometry(VoxelGrid.merge([nbr.geometry for nbr in tmap.knbr(
    #     n, 1)] + [n.geometry] + [nbr.geometry for nbr in tmap.knbr(n, 2)]).level_of_detail(2), models) for n in nodes}

    embed = {n: np.hstack([embed[n], embed_1step[n]])
             for n in nodes}

    # graph convolution
    for _ in range(k_max):
        embed = {n: (1-w)*f + graph_conv(n) for n, f in embed.items()}

    embed = [embed[n] for n in nodes]
    return embed


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
    if not len(h):
        return math.inf

    matches_a, matches_b = zip(*h)
    geometry_a, geometry_b = [n.geometry for n in matches_a], [
        n.geometry for n in matches_b]
    geometry_a, geometry_b = VoxelGrid.merge(
        geometry_a), VoxelGrid.merge(geometry_b)

    _, e = iterative_closest_point(
        source=geometry_a.level_of_detail(1).to_pcd(),
        target=geometry_b.level_of_detail(1).to_pcd(),
        ransac_iterations=1000,
        voxel_size=geometry_a.cell_size)

    return e


def match(maps: List[TopometricMap], partial_maps: List[PartialMap], k_max=0, min_node_size: int = 4096, models: Iterable[str] = ['spectral'], grow: bool = True, min_similarity: float = 30000, **kwargs):
    for i, m in enumerate(maps):
        undersized_nodes = m.filter_nodes(
            lambda n: n.geometry.size < min_node_size)

        m.remove_nodes(undersized_nodes)

    features = {map: feature_embedding(
        map, k_max, models, 1/2, partial_maps[i]) for i, map in enumerate(maps)}

    matches = defaultdict(lambda: {})
    for map_a, map_b in combinations(maps, 2):
        if map_a != map_b:
            # compute distance in feature space between every pair of nodes
            f_a, f_b = np.vstack(features[map_a]), np.vstack(features[map_b])

            # create initial pairing by finding least cost assignment between nodes
            cost_matrix = euclidean_distances(f_a, f_b)

            assignment = linear_assign(cost_matrix)

            # sort assignment by cost
            _, assignment = sort_by_func(
                assignment, lambda a: cost_matrix[a[0], a[1]], reverse=False)

            if grow:
                logging.info("Growing hypotheses")

                print(assignment)

                # from initial pairing, grow hypotheses along navigation graph
                hypotheses = grow_hypotheses(
                    map_a, map_b, assignment, cost_matrix)

                _, sorted_hypotheses = sort_by_func(
                    hypotheses, len, reverse=True)
                best_hypothesis = sorted_hypotheses[0]
            else:
                best_hypothesis = [
                    (map_a.get_node_level()[a],
                     map_b.get_node_level()[b]) for a, b in assignment]

            for node_a, node_b in best_hypothesis:
                a, b = map_a.node_index(node_a), map_b.node_index(node_b)

                if cost_matrix[a, b] < min_similarity:
                    matches[map_a, map_b][node_a, node_b] = cost_matrix[a, b]

    return dict(matches)


def sort_by_func(collection, func, reverse):
    return list(zip(*sorted([(func(h), h) for h in collection], reverse=reverse)))


def match_create(topometric_maps, partial_maps, kwargs):
    logging.info('Matching partial maps')

    matches = match(topometric_maps, partial_maps)
    return matches


def match_write(matches, kwargs):
    from pickle import dump

    with open(kwargs['matches'], 'wb') as match_file:
        dump(matches, match_file)


def match_read(kwargs):
    from pickle import load

    with open(kwargs['matches'], 'rb') as matches:
        return load(matches)


def match_visualize(topometric_maps, matches, kwargs):
    i = 0
    for a, b in matches:
        print(list(matches[a, b].values()))

        mapfn = kwargs["topometric_maps"][i] + '_matches.jpg'
        visualize_matches(a, b, matches[a, b].keys(), mapfn)

        i += 1


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

    target_matches = [(a, b_matches[v])
                      for a, v in a_matches.items() if v in b_matches]
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
