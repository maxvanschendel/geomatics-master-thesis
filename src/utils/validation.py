from typing import Iterable


def kwargs_valid(contains: Iterable[str], **kwargs):
    return set(contains).issubset(kwargs.keys())