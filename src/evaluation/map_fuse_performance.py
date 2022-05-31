from processing.map_fuse import *


def analyse_fusion_performance(result_global_map: HierarchicalTopometricMap, target_global_map: HierarchicalTopometricMap,
                               result_transform: np.array, target_transform: np.array):

    transform_distance = np.linalg.norm(result_transform - target_transform)

    return {'transform_distance': transform_distance}


if __name__ == "__main__":
    matches: List[(TopometricNode, TopometricNode)] = []

    partial_map_a: HierarchicalTopometricMap = None
    partial_map_b: HierarchicalTopometricMap = None
    target_global_map: HierarchicalTopometricMap = None
    target_transform: np.array = None

    result_global_map, result_transform = fuse(
        partial_map_a, partial_map_b, matches, False)
    map_fusion_performance = analyse_fusion_performance(
        result_global_map, target_global_map, result_transform, target_transform)

    print(map_fusion_performance)
