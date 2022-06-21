from typing import List
from model.topometric_map import TopometricNode
from processing.map_match import *
from utils.io import load_pickle

from processing.parameters import MapMergeParameters


def analyse_match_performance(partial_a, partial_b, ground_truth, matches):
    
    a_to_ground_truth = partial_a.match_nodes(ground_truth)
    b_to_ground_truth = partial_b.match_nodes(ground_truth)
    
    a_matches = {a: v for a, v in a_to_ground_truth.keys()}
    b_matches = {v: b for b, v in b_to_ground_truth.keys()}
    
    target_matches = [(a, b_matches[v]) for a, v in a_matches.items() if v in b_matches]
    
    
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


if __name__ == '__main__':
    map_a_fn = '../data/Stanford3dDataset_v1.2/Area_1/Area_1_merged_extract.pickle'
    map_b_fn = '../data/Stanford3dDataset_v1.2/Area_1/Area_1_merged_extract.pickle'
    config_fn = '../config/map_merge.yaml'

    target_matches: List[(TopometricNode, TopometricNode)] = []

    map_a = load_pickle(map_a_fn)
    map_b = load_pickle(map_b_fn)

    result_matches = match(map_a, map_b, False, m)
    match_performance = (target_matches, result_matches)

    print(match_performance)
