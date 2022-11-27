from __future__ import annotations

from sklearn import cluster

from model.topometric_map import *
from processing.extract import extract, extract_topometric_map
from processing.icp import nearest_neighbor

from utils.datasets import SimulatedPartialMap
from utils.array import euclidean_distance_matrix, replace_with_unique
from utils.linalg import rot_to_transform
from utils.visualization import visualize_htmap, visualize_map_merge, visualize_point_cloud, visualize_point_clouds, visualize_voxel_grid
from processing.registration import cluster_transform, iterative_closest_point


def fuse_topology(map_a: TopometricMap, map_b: TopometricMap,
                  matches: List[Tuple[TopometricNode, TopometricNode]]) -> TopometricMap:
    pass


def fuse_geometry(map_a: TopometricMap, map_b: TopometricMap, transform: np.array) -> TopometricMap:
    pass


def cluster_transform(transforms: List[np.array], algorithm: str = 'optics', **kwargs) -> np.array:
    # Compute distance matrix for all transformation matrices by computing the norm of their difference.
    distance_matrix = euclidean_distance_matrix(transforms)

    # Use density-based clustering to find similar transformations.
    # Either OPTICS or DBSCAN are currently available.
    if algorithm == 'optics':
        clustering = cluster.DBSCAN(eps=kwargs['max_eps'],
                                    min_samples=kwargs['min_samples'],
                                    metric='precomputed').fit(distance_matrix)
    elif algorithm == 'dbscan':
        raise NotImplementedError()

    labels = clustering.labels_
    return labels


def fuse(matches, registration_method: str, extract_cfg, partial_maps: List[SimulatedPartialMap]) -> Dict[Tuple[TopometricMap, TopometricMap], np.array]:
    """
    From a set of matches between nodes in topometric maps,
    identify best transform to bring maps into alignment and fuse them at topological level.
    """

    min_similarity = 10000
    max_error = .2
    ransac_iterations = 10000
    initial_rotation = 0

    global_maps = {}
    for map_a, map_b in matches.keys():
        node_matches: Dict[Tuple[TopometricNode,
                                 TopometricNode], float] = matches[(map_a, map_b)]
        matches = list(node_matches.keys())
        
        map_a = map_a.transform(rot_to_transform(180))

        # Find transform between matches
        match_transforms = [iterative_closest_point(
            source=a.geometry.to_pcd().transform(rot_to_transform(180)),
            target=b.geometry.to_pcd(),
            ransac_iterations=1000,
            voxel_size=a.geometry.cell_size)
            if node_matches[(a, b)] < min_similarity else (np.identity(4), math.inf)
            for a, b in matches]
        match_transforms, match_errors = zip(*match_transforms)

        print(match_transforms)
        print([node_matches[(a, b)] for a, b in matches])
        print(match_errors)

        # Filter out matches with large registration error
        match_errors, match_transforms = np.array(
            match_errors), np.array(match_transforms)

        good_matches = np.argwhere(match_errors < max_error)
        good_transforms = match_transforms[good_matches]
        translations = [t[:, :3] for t in good_transforms]

        if len(translations) > 1:
            # Cluster similar transforms into transform hypotheses
            # Assign unclustered transforms (label=-1) their own unique cluster
            transformation_clusters = cluster_transform(
                translations, max_eps=1, min_samples=1)
            transformation_clusters = replace_with_unique(
                transformation_clusters, -1)
        else:
            transformation_clusters = np.zeros((1))
            
            
        filtered_matches = [matches[int(i)] for i in np.argwhere(match_errors < max_error)]

        # Merge remaining matches into a single voxel grid for each partial map
        a_geometry, b_geometry = zip(*[(a.geometry, b.geometry) for a, b in filtered_matches])
        a_geometry, b_geometry = VoxelGrid.merge(a_geometry), VoxelGrid.merge(b_geometry)
            
        min_error = math.inf
        best_transform_cluster = None
        
        for c in np.unique(transformation_clusters):
            largest_cluster = c
            
            # For every transformation cluster, compute the mean transformation
            # then apply this transformation to the partial maps.
            transform_indices = np.argwhere(transformation_clusters == largest_cluster)
            cluster_transforms = np.array(good_transforms)[transform_indices.T.flatten()]
            
            mean_transform = (sum(cluster_transforms)/len(cluster_transforms)).squeeze()
            local_transform = mean_transform.squeeze() if len(mean_transform.shape) == 3 else mean_transform
            
            distances, _ = nearest_neighbor(a_geometry.to_pcd().transform(rot_to_transform(180)).transform(local_transform).points, b_geometry.to_pcd().points)

            print(np.mean(distances))
            if np.mean(distances) < min_error:
                min_error = np.mean(distances)
                best_transform_cluster = local_transform
                
                
        # Use transform found in previous step as coarse transform for final fine registration between the whole partial maps
        result, target = map_a.to_voxel_grid().to_pcd().transform(best_transform_cluster), map_b.to_voxel_grid().to_pcd()
        visualize_point_cloud(result.merge(target))
            
        # global_transform, global_error = iterative_closest_point(
        #     source=result,
        #     target=target,
        #     global_align=False,
        #     voxel_size=a_geometry.cell_size)

        # print(global_error)
        # # Concatenate local and global transform into final map transform that aligns partial map a with partial map b
        # map_transform = global_transform.dot(best_transform_cluster)

        # partial_map_a, partial_map_b = partial_maps[0].voxel_grid, partial_maps[1].voxel_grid
        # partial_map_c = VoxelGrid.merge([partial_map_a.transform(rot_to_transform(180)).transform(map_transform), partial_map_b])
        
        # visualize_voxel_grid(partial_map_c, 'global')

        # global_map = extract(partial_map_c.voxelize(
        #     extract_cfg.leaf_voxel_size), extract_cfg)

        global_maps[(map_a, map_b)
                    ] = partial_map_c, map_transform, global_error

    return global_maps


def fuse_create(matches, extract_cfg, partial_maps, kwargs):
    logging.info('Fusing partial maps')
    global_map = fuse(matches, 'icp', extract_cfg, partial_maps)

    return global_map


def fuse_write(global_map, kwargs):
    from pickle import dump

    with open(kwargs['fuse_fn'], 'wb') as fuse_file:
        dump(global_map, fuse_file)


def fuse_read(kwargs):
    from pickle import load

    with open(kwargs['fuse_fn'], 'rb') as global_map:
        return load(global_map)


def fuse_visualize(global_maps, kwargs):
    for map_a, map_b in global_maps:
        global_map, _, _ = global_maps[(map_a, map_b)]

        visualize_point_cloud(global_map)


def fuse_analyze(global_map, ground_truths, partial_maps, result_transforms, kwargs):
    return analyse_fusion_performance(global_map, ground_truths, result_transforms, [p.transform for p in partial_maps])


def analyse_fusion_performance(result_global_map: TopometricMap, target_global_map: TopometricMap,
                               result_transform: np.array, target_transform: np.array):

    transform_distance = np.linalg.norm(result_transform - target_transform)

    return {'transform_distance': transform_distance}
