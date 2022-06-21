import numpy as np
from processing.parameters import MapExtractionParameters
from itertools import product
from typing import Dict
from model.topometric_map import TopometricMap


def mean_similarity(ground_truth: TopometricMap, extracted: TopometricMap) -> float:  
    o2o_similarity = extracted.match_nodes(ground_truth)
    mean_similarity = np.mean(list(o2o_similarity.values()))
    
    return mean_similarity


if __name__ == '__main__':
    # Benchmark parameters
    config_fn = './config/map_extract.yaml'
    ground_truth_path = '../data/Stanford3dDataset_v1.2/Area_1'
