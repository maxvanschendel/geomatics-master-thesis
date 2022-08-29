from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple
import numpy as np

from model.point_cloud import PointCloud, Trajectory
from model.topometric_map import TopometricMap
from model.voxel_grid import VoxelGrid
from utils.io import *
from utils.linalg import random_transform


class PartialMap():
    voxel_grid: VoxelGrid

    def write(self, fn):
        import pickle as pickle
        with open(fn, 'wb') as write_file:
            pickle.dump(self, write_file)

    @staticmethod
    def read(fn):
        import pickle as pickle
        with open(fn, 'rb') as read_file:
            vg = pickle.load(read_file)
        return vg


@dataclass
class SimulatedPartialMap(PartialMap):
    voxel_grid: VoxelGrid
    transform: np.array
    point_cloud: PointCloud
    trajectory: Trajectory


@dataclass
class Dataset:
    point_cloud: str = "../data/cslam/flat/flat.ply"
    graph: str = "../data/cslam/flat/flat_graph.csv"
    trajectories: Tuple[str] = ("../data/cslam/flat/flat_trajectory_01.csv",
                                "../data/cslam/flat/flat_trajectory_02.csv"),
    partial_maps: Tuple[str] = ("../data/cslam/flat/flat_partial_01.pickle",
                                "../data/cslam/flat/flat_partial_02.pickle"),
    topometric_maps: Tuple[str] = ("../data/cslam/flat/flat_topo_01.pickle",
                                   "../data/cslam/flat/flat_topo_02.pickle"),


cslam_flat_dataset = Dataset(
    point_cloud="../data/cslam/flat/flat.ply",
    graph="../data/cslam/flat/flat_graph.csv",
    trajectories=("../data/cslam/flat/flat_trajectory_01.csv",
                  "../data/cslam/flat/flat_trajectory_02.csv"),
    partial_maps=("../data/cslam/flat/flat_partial_01.pickle",
                  "../data/cslam/flat/flat_partial_02.pickle"),
    topometric_maps=("../data/cslam/flat/flat_topo_01.pickle",
                     "../data/cslam/flat/flat_topo_02.pickle"),
)

s3dis_area_3_dataset = Dataset(
    point_cloud="../data/s3dis/area_3/area_3.ply",
    graph="../data/s3dis/area_3/area_3_graph.csv",

    trajectories=("../data/s3dis/area_3/area_3_trajectory_01.csv",
                  "../data/s3dis/area_3/area_3_trajectory_02.csv",
                  #   "../data/s3dis/area_3/area_3_trajectory_03.csv",
                  #   "../data/s3dis/area_3/area_3_trajectory_04.csv"
                  ),
    partial_maps=("../data/s3dis/area_3/area_3_partial_01.pickle",
                  "../data/s3dis/area_3/area_3_partial_02.pickle",
                  #   "../data/s3dis/area_3/area_3_partial_03.pickle",
                  #   "../data/s3dis/area_3/area_3_partial_04.pickle"
                  ),
    topometric_maps=("../data/s3dis/area_3/area_3_topo_01.pickle",
                     "../data/s3dis/area_3/area_3_topo_02.pickle",
                     #   "../data/s3dis/area_3/area_3_topo_03.pickle",
                     #   "../data/s3dis/area_3/area_3_topo_04.pickle"
                     )
)


def read_point_cloud(fn: str) -> PointCloud:
    if fn.endswith(".xyz") or fn.endswith(".csv"):
        return PointCloud.xyz(fn)
    elif fn.endswith(".ply"):
        return PointCloud.read_ply(fn)
    else:
        raise NotImplementedError(f"Failed to read point cloud, \
                                    file extension for file {fn} is not supported.")


def simulate_partial_maps(pcd, trajectories, vis_range, voxel_size) -> List[SimulatedPartialMap]:
    # Apply a random transformation to the ground truth map for each voxel grid
    transforms = [random_transform(10, 360) for _ in trajectories]

    transformed_ground_truth = [pcd.transform(t) for t in transforms]
    transformed_trajectories = [trajectories[i].transform(
        t) for i, t in enumerate(transforms)]

    transformed_voxel_grids = [pcd.voxelize(
        voxel_size) for pcd in transformed_ground_truth]
    simulated_scans = [simulate_scan(
        transformed_voxel_grids[i],
        transformed_trajectories[i],
        vis_range)
        for i, _ in enumerate(transforms)]

    partial_maps = [SimulatedPartialMap(simulated_scans[i], transforms[i], transformed_ground_truth[i], transformed_trajectories[i])
                    for i, _ in enumerate(transforms)]
    return partial_maps


def simulate_scan(voxel_grid: VoxelGrid, trajectory: Trajectory, scan_range: float):
    visible_voxels = set()

    for p in trajectory.points:
        visibility = voxel_grid.visibility(p, scan_range)
        visible_voxels = visible_voxels.union(set(visibility.voxels.keys()))

    return voxel_grid.subset(lambda v: v in visible_voxels)


def read_trajectory(fns: List[str]) -> List[np.array]:
    return list(map(lambda fn: Trajectory.read_xyz(fn), fns))


def simulate_create(config, kwargs):
    logging.info(f'Simulating partial maps')

    point_cloud = read_point_cloud(kwargs["point_cloud"])
    trajectories = read_trajectory(kwargs["trajectories"])

    partial_maps = simulate_partial_maps(
        point_cloud,
        trajectories,
        config.isovist_range,
        config.leaf_voxel_size)

    return partial_maps


def simulate_write(partial_maps, kwargs):
    logging.info(f'Writing partial maps')
    write_multiple(kwargs["partial_maps"], partial_maps,
                   lambda p, fn: p.write(fn))


def simulate_read(kwargs):
    logging.info(f'Loading partial maps')
    return [SimulatedPartialMap.read(fn) for fn in kwargs["partial_maps"]]


def simulate_visualize(partial_maps, kwargs):
    pass


def aligned_ground_truth(partial_maps, voxel_size, graph):
    logging.info(f"Preparing {len(partial_maps)} ground truth topometric maps")

    def create_htmap(p): return TopometricMap.from_segmented_point_cloud(
        p.point_cloud, graph, voxel_size)
    truths = [create_htmap(p) for p in partial_maps]

    return truths


def write_multiple(fns, objs, write_func):
    for i, p in enumerate(objs):
        out_fn = fns[i]
        write_func(p, out_fn)
