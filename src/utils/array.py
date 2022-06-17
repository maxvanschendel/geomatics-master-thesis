from collections import Counter
from itertools import product
from statistics import mean
from typing import Iterable
import numpy as np
from random import randint

def most_common(elements):
    if elements:
        return Counter(elements).most_common(1)[0][0]
    else:
        raise ValueError('Input must contain at least 1 item.')
    
def n_smallest_indices(input: np.array, n: int):
    smallest_flat = np.argpartition(input.ravel(), n)[:n]
    smallest_indices = [np.unravel_index(
        i, input.shape) for i in smallest_flat]

    return smallest_indices

def replace_with_unique(array: np.array, replace_val: int) -> np.array:
    # We replace every occurence of replace_val  with the maximum value plus an offset.
    # Numbers above the maximum value are guarantueed to not be in the array yet,
    # so are a safe replacement value.
    out_array = np.empty(array.shape)
    max_value = np.max(array)

    for i, val in enumerate(array):
        if val == replace_val:
            out_array[i] = max_value + (i + 1)
        else:
            out_array[i] = val

    return out_array

def euclidean_distance_matrix(arrays: Iterable[np.array]) -> np.array:
    n_arrays = len(arrays)
    distance_matrix = np.zeros(((n_arrays, n_arrays)))

    # Get all combinations of array indices, these will be the row
    # and column index of each distance
    array_pairs = product(range(n_arrays), repeat=2)

    # For every pair of arrays compute the norm of their difference
    # and store it in the distance matrix
    for i, j in array_pairs:
        array_distance = np.linalg.norm(arrays[i] - arrays[j])
        distance_matrix[i][j] = array_distance

    return distance_matrix


def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def sort_dict_by_value(d, reverse=False):
    return sorted(d.items(), reverse=reverse, key=lambda i: i[1])

def mean_dict_value(d):
    return mean(d.values())

def random_select(array):
    return array[randint(0, len(array))]

def all_same(array) -> bool:
    return all(p == array[0] for p in array)

def one_to_one(distance: np.array):
    matches = [(distance[x, y], (x, y)) for x, y in np.ndindex((distance.shape))]

    one_to_one = []
    matched = set()

    for d, m in sorted(matches):
        node_a, node_b = m

        if node_a not in matched and node_b not in matched:
            one_to_one.append((d, m))

            matched.add(node_a)
            matched.add(node_b)

    return one_to_one