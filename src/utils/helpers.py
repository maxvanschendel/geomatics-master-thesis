from collections import Counter
from random import random
from typing import List
import numpy as np

def random_color(alpha: bool = False) -> List[float]:
    return [random(), random(), random()]

def most_common(elements):
    return Counter(elements).most_common(1)[0][0]

    
def n_smallest_indices(input: np.array, n: int):
    smallest_flat = np.argpartition(input.ravel(), n)[:n]
    smallest_indices = [np.unravel_index(
        i, input.shape) for i in smallest_flat]

    return smallest_indices