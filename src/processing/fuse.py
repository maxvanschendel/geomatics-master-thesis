from __future__ import annotations

from model.topometric_map import *
from processing.extract import extract, extract_topometric_map

from utils.datasets import SimulatedPartialMap
from utils.array import replace_with_unique
from utils.visualization import visualize_htmap, visualize_map_merge, visualize_point_clouds, visualize_voxel_grid
from processing.registration import cluster_transform, iterative_closest_point


def fuse_topology(map_a: TopometricMap, map_b: TopometricMap,
                  matches: List[Tuple[TopometricNode, TopometricNode]]) -> TopometricMap:
    pass


def fuse_geometry(map_a: TopometricMap, map_b: TopometricMap, transform: np.array) -> TopometricMap:
    pass


def fuse(matches, registration_method: str, extract_cfg, partial_maps: List[SimulatedPartialMap]) -> Dict[Tuple[TopometricMap, TopometricMap], np.array]:
    """
    From a set of matches between nodes in topometric maps,
    identify best transform to bring maps into alignment and fuse them at topological level.
    """

    global_maps = {}
    for map_a, map_b in matches.keys():
        node_matches: Dict[Tuple[TopometricNode, TopometricNode], float] = matches[(map_a, map_b)]
        matches = list(node_matches.keys())


        
        # Find transform between both maps based on ICP registration between matched spaces
        match_transforms = [iterative_closest_point(
                                source=a.geometry.to_pcd(),
                                target=b.geometry.to_pcd(),
                                voxel_size=a.geometry.cell_size) if node_matches[(a, b)] < 1 else (np.identity(4), math.inf)
                            for a, b in matches]
        
        match_transforms, match_errors = zip(*match_transforms)
        match_errors = np.array(match_errors)
        match_transforms = np.array(match_transforms)
        
        print(list(node_matches.values()))
        print(match_errors)
        
        filtered_matches = [matches[int(i)] for i in np.argwhere(match_errors < 0.1)]
        filtered_geometry = [(a.geometry, b.geometry) for a, b in filtered_matches]
        
        a_geometry, b_geometry = zip(*filtered_geometry)
        a_geometry, b_geometry = VoxelGrid.merge(a_geometry), VoxelGrid.merge(b_geometry)
        a_pcd, b_pcd = a_geometry.to_pcd(), b_geometry.to_pcd()
        
        coarse_transform, final_error = iterative_closest_point(
                source=a_pcd,
                target=b_pcd,
                ransac_iterations=10000,
                voxel_size=a_geometry.cell_size)
        
        result, target = map_a.to_voxel_grid().to_pcd().transform(coarse_transform), map_b.to_voxel_grid().to_pcd()
        
        fine_transform, final_error = iterative_closest_point(
                source=result,
                target=target,
                global_align=False,
                voxel_size=a_geometry.cell_size)
        
        map_transform = fine_transform.dot(coarse_transform)
        
        
        partial_map_a, partial_map_b = partial_maps[0].point_cloud, partial_maps[1].point_cloud
        partial_map_c = partial_map_a.transform(map_transform).merge(partial_map_b)
        
        global_map = extract(partial_map_c.voxelize(extract_cfg.leaf_voxel_size), extract_cfg)
        
        
        
        # map_a_transformed = map_a.transform(map_transform)
        
        
        
        # map_b = map_b.transform(np.identity(4))
        
        
        
        # # merge geometry and merge similar nodes
        # global_map = map_a_transformed.merge(map_b)
        # # global_map = global_map.to_voxel_grid()
        
        
        
        
        # global_map = global_map.merge_similar_nodes(0.5)
        # global_map = global_map.merge_similar_nodes(0.5)
        
        # # global_map.edge_transfer(map_a_transformed)
        # # global_map.edge_transfer(map_b)

        # # visualize_htmap(global_map, 'a_transformed.jpg')
        
        # # get the new navigation manifold from global map
        # logging.info("Extracting global topometric map from fused geometry")
        # nav_subsets = []
        # for n in global_map.graph.nodes:
        #     geometry: VoxelGrid = n.geometry
            
        #     geometry.set_attr_uniform(VoxelGrid.cluster_attr, len(nav_subsets))
        #     nav_voxels = geometry.voxel_subset(geometry.get_attr(VoxelGrid.nav_attr, True))
        #     nav_subsets.append(nav_voxels)
            
        # global_nav_manifold = VoxelGrid.merge(nav_subsets)
        
        # # extract new topometric map from global navigation manifold and merged segmentation
        # global_map = extract_topometric_map(extract_cfg.min_voxels, global_nav_manifold, global_map.to_voxel_grid(), connected=False)
        
        # visualize_voxel_grid(global_map.to_voxel_grid(), 'global_map.jpg')
        


        
        
        # global_map = global_map.to_voxel_grid()
        # global_map.clear_attributes()
        
        # visualize_voxel_grid(global_map)
        # global_map = extract(global_map, extract_cfg)
        
        visualize_htmap(global_map, 'global_map.jpg')
        
        
        global_maps[(map_a, map_b)] = global_map, map_transform

        
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
    raise NotImplementedError("")


def fuse_visualize(global_map, kwargs):
    raise NotImplementedError("")


def fuse_analyze(global_map, ground_truths, partial_maps, result_transforms, kwargs):
    return analyse_fusion_performance(global_map, ground_truths, result_transforms, [p.transform for p in partial_maps])


def analyse_fusion_performance(result_global_map: TopometricMap, target_global_map: TopometricMap,
                               result_transform: np.array, target_transform: np.array):

    transform_distance = np.linalg.norm(result_transform - target_transform)

    return {'transform_distance': transform_distance}
