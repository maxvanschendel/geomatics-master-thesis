from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from yaml import load, Loader, dump


@dataclass
class PipelineParameters:
    ground_truth_pcd: str = "../data/cslam/flat/flat.ply"
    ground_truth_graph: str = "../data/cslam/flat/flat_graph.csv"
    simulated_trajectories: str = ("../data/cslam/flat/flat_trajectory_01.csv", 
                                   "../data/cslam/flat/flat_trajectory_02.csv")
    analyse_performance: bool = True


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

    min_voxels: int

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
