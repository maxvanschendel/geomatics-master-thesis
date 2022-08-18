from dataclasses import dataclass
import logging
from typing import Dict, Iterable, List, Tuple
import numpy as np

from model.point_cloud import PointCloud, Trajectory
from model.topometric_map import TopometricMap
from model.voxel_grid import VoxelGrid
from utils.io import *
from utils.linalg import random_transform
from utils.visualization import visualize_voxel_grid


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

cslam_house_dataset = Dataset(
    point_cloud="../data/cslam/house/house.ply",
    graph="../data/cslam/house/house_graph.csv",
    trajectories=("../data/cslam/house/house_trajectory_01.csv",
                  "../data/cslam/house/house_trajectory_02.csv"),
    partial_maps=("../data/cslam/house/house_partial_01.pickle",
                  "../data/cslam/house/house_partial_02.pickle"),
    topometric_maps=("../data/cslam/house/house_topo_01.pickle",
                     "../data/cslam/house/house_topo_02.pickle"),
)

cslam_lab_dataset = Dataset(
    point_cloud="../data/cslam/lab/lab.ply",
    graph="../data/cslam/lab/lab_graph.csv",
    trajectories=("../data/cslam/lab/lab_trajectory_01.csv",
                  "../data/cslam/lab/lab_trajectory_02.csv"),
    partial_maps=("../data/cslam/lab/lab_partial_01.pickle",
                  "../data/cslam/lab/lab_partial_02.pickle"),
    topometric_maps=("../data/cslam/lab/lab_topo_01.pickle",
                     "../data/cslam/lab/lab_topo_02.pickle"),
)

s3dis_area_1_dataset = Dataset(
    point_cloud="../data/s3dis/area_1/area_1_subsample5cm.ply",
    graph="../data/s3dis/area_1/area_1_graph.csv",

    trajectories=("../data/s3dis/area_1/area_1_trajectory_01.csv",
                  "../data/s3dis/area_1/area_1_trajectory_02.csv",
                  ),
    partial_maps=("../data/s3dis/area_1/area_1_partial_01.pickle",
                  "../data/s3dis/area_1/area_1_partial_02.pickle",
                  ),
    topometric_maps=("../data/s3dis/area_1/area_1_topo_01.pickle",
                     "../data/s3dis/area_1/area_1_topo_02.pickle",
                     )
)

s3dis_area_2_dataset = Dataset(
    point_cloud="../data/s3dis/area_2/area_2_subsample5cm.ply",
    graph="../data/s3dis/area_2/area_2_graph.csv",

    trajectories=("../data/s3dis/area_2/area_2_trajectory_01.csv",
                  "../data/s3dis/area_2/area_2_trajectory_02.csv",
                  ),
    partial_maps=("../data/s3dis/area_2/area_2_partial_01.pickle",
                  "../data/s3dis/area_2/area_2_partial_02.pickle",
                  ),
    topometric_maps=("../data/s3dis/area_2/area_2_topo_01.pickle",
                     "../data/s3dis/area_2/area_2_topo_02.pickle",
                     )
)

s3dis_area_3_dataset = Dataset(
    point_cloud="../data/s3dis/area_3/area_3_subsample5cm.ply",
    graph="../data/s3dis/area_3/area_3_graph.csv",

    trajectories=("../data/s3dis/area_3/area_3_trajectory_01.csv",
                  "../data/s3dis/area_3/area_3_trajectory_02.csv",
                  ),
    partial_maps=("../data/s3dis/area_3/area_3_partial_01.pickle",
                  "../data/s3dis/area_3/area_3_partial_02.pickle",
                  ),
    topometric_maps=("../data/s3dis/area_3/area_3_topo_01.pickle",
                     "../data/s3dis/area_3/area_3_topo_02.pickle",
                     )
)

s3dis_area_4_dataset = Dataset(
    point_cloud="../data/s3dis/area_4/area_4_subsample5cm.ply",
    graph="../data/s3dis/area_4/area_4_graph.csv",

    trajectories=("../data/s3dis/area_4/area_4_trajectory_01.csv",
                  "../data/s3dis/area_4/area_4_trajectory_02.csv",
                  ),
    partial_maps=("../data/s3dis/area_4/area_4_partial_01.pickle",
                  "../data/s3dis/area_4/area_4_partial_02.pickle",
                  ),
    topometric_maps=("../data/s3dis/area_4/area_4_topo_01.pickle",
                     "../data/s3dis/area_4/area_4_topo_02.pickle",
                     )
)

s3dis_area_5_dataset = Dataset(
    point_cloud="../data/s3dis/area_5/area_5_subsample5cm.ply",
    graph="../data/s3dis/area_5/area_5_graph.csv",

    trajectories=("../data/s3dis/area_5/area_5_trajectory_01.csv",
                  "../data/s3dis/area_5/area_5_trajectory_02.csv",
                  ),
    partial_maps=("../data/s3dis/area_5/area_5_partial_01.pickle",
                  "../data/s3dis/area_5/area_5_partial_02.pickle",
                  ),
    topometric_maps=("../data/s3dis/area_5/area_5_topo_01.pickle",
                     "../data/s3dis/area_5/area_5_topo_02.pickle",
                     )
)

s3dis_area_6_dataset = Dataset(
    point_cloud="../data/s3dis/area_6/area_6_subsample5cm.ply",
    graph="../data/s3dis/area_6/area_6_graph.csv",

    trajectories=("../data/s3dis/area_6/area_6_trajectory_01.csv",
                  "../data/s3dis/area_6/area_6_trajectory_02.csv",
                  ),
    partial_maps=("../data/s3dis/area_6/area_6_partial_01.pickle",
                  "../data/s3dis/area_6/area_6_partial_02.pickle",
                  ),
    topometric_maps=("../data/s3dis/area_6/area_6_topo_01.pickle",
                     "../data/s3dis/area_6/area_6_topo_02.pickle",
                     )
)

other_elspeet_dataset = Dataset(
    point_cloud="../data/other/elspeet/elspeet_subsample5cm.ply",
    graph="../data/other/elspeet/elspeet_graph.csv",

    trajectories=("../data/other/elspeet/elspeet_trajectory_01.csv",
                  "../data/other/elspeet/elspeet_trajectory_02.csv",
                  ),
    partial_maps=("../data/other/elspeet/elspeet_partial_01.pickle",
                  "../data/other/elspeet/elspeet_partial_02.pickle",
                  ),
    topometric_maps=("../data/other/elspeet/elspeet_topo_01.pickle",
                     "../data/other/elspeet/elspeet_topo_02.pickle",
                     )
)

s3dis_datasets = [s3dis_area_1_dataset, s3dis_area_2_dataset, s3dis_area_3_dataset, s3dis_area_4_dataset, s3dis_area_5_dataset, s3dis_area_6_dataset]
cslam_datasets = [cslam_flat_dataset, cslam_house_dataset, cslam_lab_dataset]
various_datasets = [other_elspeet_dataset]


def read_point_cloud(fn: str) -> PointCloud:
    if fn.endswith(".xyz") or fn.endswith(".csv"):
        return PointCloud.xyz(fn)
    elif fn.endswith(".ply"):
        return PointCloud.read_ply(fn, y_up=False)
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
    return list(map(lambda fn: Trajectory.read_xyz(fn, y_up=False), fns))


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


def simulate_visualize(partial_maps: Iterable[SimulatedPartialMap], kwargs):
    for p in partial_maps:
        vg = p.voxel_grid
        aabb = p.point_cloud.aabb

        def cmap(v): return np.array(
            [(vg.voxel_centroid(v)[1]) / aabb[1][1]]*3)

        vg.for_each(lambda v: vg.set_attr(v, 'color', cmap(v)))
        visualize_voxel_grid(vg)


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
