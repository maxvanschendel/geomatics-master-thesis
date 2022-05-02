from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from yaml import load, Loader, dump


@dataclass
class PipelineParameters:
    """Defines pipeline behaviour."""

    # Map extraction
    skip_extract: bool = True   # Skip map extraction step and go directly to map matching
    write_htmap: bool = True    # Write map extraction results to files

    partial_map_a: str = "../data/cslam_dataset/diningroom2kitchen.ply"
    partial_map_b: str = "../data/cslam_dataset/hall2oldkitchen.ply"
    htmap_a_fn: str = '../data/test/diningroom2kitchen_htmap.pickle'
    htmap_b_fn: str = '../data/test/hall2oldkitchen_htmap.pickle'

    # Map matching
    skip_match: bool = False    # Skip map matching and go directly to map merging

    # Map merging
    skip_merge: bool = False    # Skip map merging and go directly to evaluation

    def post_init(self):
        """Performs validation of pipeline configuration input,
        """

        pass

class PreProcessingParametersException(Exception):
    pass


@dataclass(frozen=True)
class PreProcessingParameters:
    reduce: float
    scale: Tuple[float]
    noise_scale: float

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

@dataclass(frozen=True)
class MapExtractionParameters:
    class MapExtractionParametersException(Exception):
        pass

    leaf_voxel_size: float
    traversability_lod: int
    segmentation_lod: int
    
    kernel_scale: float
    isovist_height: float
    isovist_spacing: float

    isovist_subsample: float
    isovist_range: float

    weight_threshold: float
    min_inflation: float
    max_inflation: float
    
    storey_buffer: int = -5
    storey_height: int = 300
    
    min_voxels: int = 100

    @staticmethod
    def deserialize(data: str) -> MapExtractionParameters:
        return load(data, Loader)

    @staticmethod
    def read(fn: str) -> MapExtractionParameters:
        with open(fn, "r") as read_file:
            file_contents = read_file.read()
        return MapExtractionParameters.deserialize(file_contents)

    def serialize(self) -> str:
        return dump(self)

    def write(self, fn: str) -> None:
        with open(fn, "w+") as write_file:
            write_file.write(self.serialize())

@dataclass(frozen=True)
class MapMergeParameters:
    class MapMergeParametersException(Exception):
        pass

    @staticmethod
    def deserialize(data: str) -> MapMergeParameters:
        return load(data, Loader)

    @staticmethod
    def read(fn: str) -> MapMergeParameters:
        with open(fn, "r") as read_file:
            file_contents = read_file.read()
        return MapMergeParameters.deserialize(file_contents)

    def serialize(self) -> str:
        return dump(self)

    def write(self, fn: str) -> None:
        with open(fn, "w+") as write_file:
            write_file.write(self.serialize())