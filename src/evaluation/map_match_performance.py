from typing import List
from model.topometric_map import TopometricNode
from processing.map_match import *
from utils.io import load_pickle

from processing.parameters import MapMergeParameters


def analyse_match_performance(partial_a, partial_b, ground_truth_a, ground_truth_b, matches):
    
    a_to_ground_truth = partial_a.match_nodes(ground_truth_a)
    b_to_ground_truth = partial_b.match_nodes(ground_truth_b)
    
    a_matches = {a: v for a, v in a_to_ground_truth.keys()}
    b_matches = {v: b for b, v in b_to_ground_truth.keys()}
    
    
    a_nodes = partial_a.get_node_level()
    b_nodes = partial_b.get_node_level()
        
    target_matches = [(a, b_matches[v]) for a, v in a_matches.items() if v in b_matches]
    target_matches = [(a_nodes[a], b_nodes[b]) for a, b in target_matches]
    
    
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