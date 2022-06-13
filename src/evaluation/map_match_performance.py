from typing import List
from model.topometric_map import TopometricNode
from processing.map_match import *
from utils.io import load_pickle

from processing.parameters import MapMergeParameters

m = 10


def analyse_match_performance(target_matches, result_matches):
    true_positive = [
        True for match in result_matches if match in target_matches]
    false_positive = [
        True for match in result_matches if match not in target_matches]
    false_negative = [
        True for match in target_matches if match not in result_matches]

    # TODO: Validate that this is actually correct
    accuracy = sum(true_positive) / len(result_matches)
    precision = sum(true_positive) / (sum(true_positive) + sum(false_positive))
    recall = sum(true_positive) / (sum(true_positive) + sum(false_negative))
    f_1 = 2 * (precision*recall) / (precision+recall)

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
