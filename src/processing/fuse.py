from __future__ import annotations

from model.topometric_map import *

from utils.array import replace_with_unique
from utils.visualization import visualize_map_merge, visualize_point_clouds
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
        node_matches: Dict[Tuple[TopometricNode, TopometricNode], float] = matches[(map_a, map_b)]
        matches = list(node_matches.keys())

        # Find transform between both maps based on ICP registration between matched spaces
        match_transforms = [registration(
                source=a.geometry.to_pcd(),
                target=b.geometry.to_pcd(),
                registration_methods=registration_method,
                voxel_size=a.geometry.cell_size)
            for a, b in matches]
        
        match_transforms, match_errors = zip(*match_transforms)
        match_errors = np.array(match_errors)
        match_transforms = np.array(match_transforms)
        
        filtered_matches = [matches[int(i)] for i in np.argwhere(match_errors < 0.15)]
        filtered_geometry = [(a.geometry, b.geometry) for a, b in filtered_matches]
        
        a_geometry, b_geometry = zip(*filtered_geometry)
        a_geometry, b_geometry = VoxelGrid.merge(a_geometry), VoxelGrid.merge(b_geometry)
        a_pcd, b_pcd = a_geometry.to_pcd(), b_geometry.to_pcd()
        
        final_transform, final_error = registration(
                source=a_pcd,
                target=b_pcd,
                registration_methods=registration_method,
                voxel_size=a_geometry.cell_size)
        
        result, target = map_a.to_voxel_grid().to_pcd().transform(final_transform), map_b.to_voxel_grid().to_pcd()
        final_transform, final_error = registration(
                source=result,
                target=target,
                registration_methods=registration_method,
                voxel_size=a_geometry.cell_size)
        
        
        
        visualize_point_clouds([result.merge(target), map_a.to_voxel_grid().to_pcd().transform(final_transform).merge(map_b.to_voxel_grid().to_pcd())])
        
        transforms[(map_a, map_b)] = final_transform


        
        # print(match_errors)
        
        # if len(match_transforms) > 1:
        #     # Cluster similar transforms into transform hypotheses
        #     # Assign unclustered transforms (label=-1) their own unique cluster
        #     transformation_clusters = cluster_transform(
        #         match_transforms, max_eps=np.inf, min_samples=3)
        #     transformation_clusters = replace_with_unique(
        #         transformation_clusters, -1)
        # else:
        #     transformation_clusters = np.zeros((1))
                
        # filtered_clusters = transformation_clusters[match_errors < 0.15]
        # u, c = np.unique(filtered_clusters, return_counts=True)
        
        # for cluster in u:
        #     cluster = int(cluster)
        #     cluster_transformations = match_transforms[transformation_clusters == cluster]
            
        #     mean_transformation = sum(cluster_transformations)/len(cluster_transformations)
            
        #     a_pcd = map_a.to_voxel_grid().to_pcd().transform(cluster_transformations[0])
        #     b_pcd = map_b.to_voxel_grid().to_pcd()
            
        #     visualize_point_cloud(a_pcd.merge(b_pcd))
        
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
