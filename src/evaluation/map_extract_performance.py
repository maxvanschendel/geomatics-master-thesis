import numpy as np
from processing.parameters import MapExtractionParameters
from itertools import product
from typing import Dict
from model.topometric_map import TopometricMap, Hierarchy, TopometricNode
from model.voxel_grid import VoxelGrid
from utils.array import mean_dict_value, one_to_one, sort_dict_by_value


def analyse_extract_performance(ground_truth: TopometricMap, topometric_map: TopometricMap):
    truth_grid = ground_truth.to_voxel_grid()
    
    # Compute the overlap between the voxels of every segmented room
    # and every ground truth label.
    extracted_nodes = topometric_map.get_node_level(Hierarchy.ROOM)
    ground_truth_nodes = ground_truth.get_node_level(Hierarchy.ROOM)
      
    n_rooms, n_labels = len(extracted_nodes), len(ground_truth_nodes)
    
    similarity_matrix = np.zeros((n_rooms, n_labels))
    for i, j in product(range(n_rooms), range(n_labels)):
        e_node, gt_node = extracted_nodes[i], ground_truth_nodes[j]
        
        # Find overlap of extracted room voxels and ground truth subset
        jaccard_index = e_node.geometry.jaccard__index(gt_node.geometry)
        similarity_matrix[i, j] = jaccard_index

    o2o_mapping = one_to_one(similarity_matrix)
    o2o_similarity = similarity_matrix[o2o_mapping]
    mean_similarity = np.mean(o2o_similarity)
    
    return {'mean_similarity': mean_similarity}


if __name__ == '__main__':
    # Benchmark parameters
    config_fn = './config/map_extract.yaml'
    ground_truth_path = '../data/Stanford3dDataset_v1.2/Area_1'
