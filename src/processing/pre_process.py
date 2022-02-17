from dataclasses import dataclass
from yaml import load, dump

import numpy as np
from model.map_representation import *

class PreProcessingParametersException(Exception):
        pass

@dataclass(frozen=True)
class PreProcessingParameters:
    voxel_size: float
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

    def serialize(self) -> str:
        return 

def pre_process(partial_map: PointCloudRepresentation, params: PreProcessingParameters) -> VoxelRepresentation:
    print("Preprocessing")

    partial_map_reduced = partial_map.random_reduce(params.reduce)
    partial_map_scaled = partial_map_reduced.scale(np.array(params.scale))
    partial_map_voxel = partial_map_scaled.voxelize(params.voxel_size)

    return partial_map_voxel
