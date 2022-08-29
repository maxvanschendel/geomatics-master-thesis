from __future__ import annotations

from model.topometric_map import *

from utils.array import replace_with_unique
from utils.visualization import visualize_map_merge
from processing.registration import cluster_transform, registration


def fuse_topology(map_a: TopometricMap, map_b: TopometricMap,
                  matches: List[Tuple[TopometricNode, TopometricNode]]) -> TopometricMap:
    pass


def fuse_geometry(map_a: TopometricMap, map_b: TopometricMap,
                  transform: np.array) -> TopometricMap:
    pass


def fuse(matches, registration_method: str) -> Dict[Tuple[TopometricMap, TopometricMap], np.array]:
    """
    From a set of matches between nodes in topometric maps,
    identify best transform to bring maps into alignment and fuse them at topological level.
    """

    transforms = {}
    for map_a, map_b in matches.keys():
        node_matches: Dict[Tuple[TopometricNode,
                                 TopometricNode], float] = matches[(map_a, map_b)]

        # Find transform between both maps based on ICP registration between matched spaces
        match_transforms = [registration(
                source=a.geometry.to_pcd(),
                target=b.geometry.to_pcd(),
                registration_methods=registration_method,
                voxel_size=a.geometry.cell_size)
            for a, b in node_matches.keys()]
        
        logging.info(f"Identified {len(match_transforms)} transform clusters")

        if len(match_transforms) > 1:
            # Cluster similar transforms into transform hypotheses
            # Assign unclustered transforms (label=-1) their own unique cluster
            transformation_clusters = cluster_transform(
                match_transforms, max_eps=5, min_samples=1)
            transformation_clusters = replace_with_unique(
                transformation_clusters, -1)
        else:
            transformation_clusters = np.zeros((1))
            

        for cluster in np.unique(transformation_clusters):
            # For every transformation cluster, compute the mean transformation
            # then apply this transformation to the partial maps.
            transform_indices = np.argwhere(transformation_clusters == cluster)
            cluster_transforms = np.array(match_transforms)[
                transform_indices.T.flatten()].squeeze()

            mean_transform = np.mean(cluster_transforms, axis=2).squeeze() if len(
                cluster_transforms.shape) == 3 else cluster_transforms
            transforms[(map_a, map_b)] = mean_transform

            map_b_transformed = map_b.transform(mean_transform)

            visualize_map_merge(map_a, map_b_transformed)

    return transforms


def fuse_create(matches, kwargs):
    logging.info('Fusing partial maps')
    global_map, result_transforms = fuse(matches, 'icp')

    return global_map, result_transforms


def fuse_write(global_map, kwargs):
    raise NotImplementedError("")


def fuse_read(kwargs):
    raise NotImplementedError("")


def fuse_visualize(global_map, kwargs):
    raise NotImplementedError("")


def fuse_analyze(global_map, ground_truths, partial_maps, result_transforms, kwargs):
    return analyse_fusion_performance(global_map, ground_truths, result_transforms, [p.transform for p in partial_maps])


def analyse_fusion_performance(result_global_map: TopometricMap, target_global_map: TopometricMap,
                               result_transform: np.array, target_transform: np.array):

    transform_distance = np.linalg.norm(result_transform - target_transform)

    return {'transform_distance': transform_distance}
