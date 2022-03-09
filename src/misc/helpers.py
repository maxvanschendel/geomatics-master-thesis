from collections import Counter
from random import random
from typing import List

def random_color(alpha: bool = False) -> List[float]:
    return [random(), random(), random()]

def most_common(elements):
    return Counter(elements).most_common(1)[0][0]