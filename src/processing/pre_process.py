from __future__ import annotations
from dataclasses import dataclass
from torch import randint
from yaml import load, Loader, dump

import numpy as np
from model.topometric_map import *
from model.point_cloud import *
from model.voxel_grid import *


class PreProcessingParametersException(Exception):
    pass


@dataclass(frozen=True)
class PreProcessingParameters:
    reduce: float
    scale: Tuple[float]

    # input validation
    def __post_init__(self):
        if not 0 <= self.reduce <= 1:
            raise PreProcessingParametersException(
                "Reduce must be a fraction between 0 and 1.")

        if not len(self.scale) == 3:
            raise PreProcessingParametersException(
                "Scale must be a 3-dimensional vector.")

    @staticmethod
    def deserialize(data: str) -> PreProcessingParameters:
        return load(data, Loader)

    @staticmethod
    def read(fn: str) -> PreProcessingParameters:
        with open(fn, "r") as read_file:
            file_contents = read_file.read()
        return PreProcessingParameters.deserialize(file_contents)

    def serialize(self) -> str:
        return dump(self)

    def write(self, fn: str) -> None:
        with open(fn, "w+") as write_file:
            write_file.write(self.serialize())
        


def pre_process(partial_map: PointCloud, params: PreProcessingParameters) -> VoxelGrid:
    partial_map_reduced = partial_map.random_reduce(params.reduce)
    partial_map_scaled = partial_map_reduced.scale(np.array(params.scale))
    partial_map_rot = partial_map_scaled.rotate(random()*360, [0,1,0])
    
    return partial_map_rot
