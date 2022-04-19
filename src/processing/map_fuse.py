from __future__ import annotations

from argparse import ArgumentError
from yaml import Loader, dump, load

from karateclub import (AE, BANE, FGSD, FSCNMF, IGE, LDP, MUSAE, SINE,
                        FeatherGraph, FeatherNode, GeoScattering, GL2Vec,
                        Graph2Vec, NetLSD, WaveletCharacteristic)
from learning3d.models import DGCNN, PointNet, PPFNet

from sklearn import cluster


from model.topometric_map import *
from utils.array import euclidean_distance_matrix, replace_with_unique
from utils.helpers import *
from utils.visualization import visualize_map_merge 
from processing.registration import registration


def cluster_transformations(transforms: List[np.array], algorithm: str = 'optics', **kwargs):
    # Only these algorithms are currently supported
    if algorithm not in ['dbscan', 'optics']:
        raise ArgumentError(
            algorithm, f'Clustering algorithm must be either dbscan or optics. Currently {algorithm}.')

    # Compute distance matrix for all transformation matrices by computing the norm of their difference.
    distance_matrix = euclidean_distance_matrix(transforms)

    # Use density-based clustering to find similar transformations. Either OPTICS or DBSCAN are available.
    if algorithm == 'optics':
        clustering = cluster.OPTICS(max_eps=kwargs['max_eps'],
                                    min_samples=kwargs['min_samples'],
                                    metric='precomputed').fit(distance_matrix)
    elif algorithm == 'dbscan':
        pass

    return clustering.labels_


def fuse(map_a: HierarchicalTopometricMap, map_b: HierarchicalTopometricMap,
         matches: List[Tuple[TopometricNode, TopometricNode]], draw_result: bool = True) -> HierarchicalTopometricMap:
    """
    From a set of matches between nodes in topometric maps,
    identify best transform to bring maps into alignment and fuse them at topological level.
    """

    # Find transform between both maps based on ICP registration between matched spaces
    match_transforms = [registration(
        source=a.geometry.to_pcd(),
        target=b.geometry.to_pcd(),
        voxel_size=a.cell_size)
        for a, b in matches]

    # Cluster similar transforms into transform hypotheses
    transformation_clusters = cluster_transformations(match_transforms)
    transformation_clusters = replace_with_unique(transformation_clusters, -1)

    for cluster in np.unique(transformation_clusters):
        # For every transformation cluster, compute the mean transformation
        # then apply this transformation to the partial maps.
        transform_indices = np.argwhere(transformation_clusters == cluster)
        cluster_transform = np.mean(match_transforms)[transform_indices]
        map_b_transformed = map_b.transform(cluster_transform)

        if draw_result:
            visualize_map_merge(map_a, map_b_transformed)
