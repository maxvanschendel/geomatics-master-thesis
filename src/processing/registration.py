
from typing import Callable, List
from model.point_cloud import PointCloud

import numpy as np
import open3d as o3d
from numba import cuda
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from sklearn import cluster
from utils.array import euclidean_distance_matrix

dcp_model = './learning3d/pretrained/exp_dcp/models/best_model.t7'
pnlk_model = './learning3d/pretrained/exp_pnlk/models/best_model.t7'
dgmr_model = './learning3d/pretrained/exp_deepgmr/models/best_model.pth'


def display_open3d(template, source, transformed_source):
    template_ = o3d.geometry.PointCloud()
    source_ = o3d.geometry.PointCloud()
    transformed_source_ = o3d.geometry.PointCloud()
    template_.points = o3d.utility.Vector3dVector(template)
    source_.points = o3d.utility.Vector3dVector(source + np.array([0, 0, 0]))
    transformed_source_.points = o3d.utility.Vector3dVector(transformed_source)
    template_.paint_uniform_color([1, 0, 0])
    source_.paint_uniform_color([0, 1, 0])
    transformed_source_.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries(
        [template_, source_, transformed_source_])


def align_least_squares(a: np.array, b: np.array):
    def f(t): return sum([np.linalg.norm(
        a[i] - (R.from_euler('xyz', [0, t[3], 0]).apply(b[i]) + t[0:3])) for i, _ in enumerate(a)])

    return least_squares(f, np.zeros((4))).x


def pointnet_lk(source: PointCloud, target: PointCloud, **kwargs) -> np.array:
    import torch
    from learning3d.models import PointNetLK, PointNet

    torch.cuda.empty_cache()
    device = torch.device("cuda")

    source.points -= np.mean(source.points, axis=0)
    target.points -= np.mean(target.points, axis=0)

    pts_source = source.to_tensor().to(device)
    pts_target = target.to_tensor().to(device)

    feature_model = PointNet(global_feat=False).to(device)
    pnlk = PointNetLK(feature_model=feature_model,
                      delta=1e-02,
                      xtol=1e-07,
                      p0_zero_mean=False,
                      p1_zero_mean=False,
                      pooling='max',
                      learn_delta=False)

    state_dict = torch.load(pnlk_model)
    pnlk.load_state_dict(state_dict, strict=False)

    # Get 4x4 transformation matrix from results
    registration_result = pnlk(pts_target, pts_source, maxiter=200)
    transformation = registration_result['est_T'].cpu().detach().numpy()

    display_open3d(pts_target.detach().cpu().numpy()[0],
                   pts_source.detach().cpu().numpy()[0],
                   registration_result['transformed_source']
                   .detach()
                   .cpu()
                   .numpy()[0]
                   )

    return transformation


def deepgmr(source: PointCloud, target: PointCloud, **kwargs) -> np.array:
    import torch
    from learning3d.models import DeepGMR, PointNet

    torch.cuda.empty_cache()
    device = torch.device("cuda")

    source = source.translate(-np.mean(source.points, axis=1))
    target = target.translate(-np.mean(target.points, axis=1))

    pts_source = source.to_tensor(bcn=True).to(device)
    pts_target = target.to_tensor(bcn=True).to(device)

    feature_model = PointNet(input_shape='bnc', global_feat=True).to(device)
    dgmr = DeepGMR(use_rri=True, feature_model=feature_model,
                   nearest_neighbors=20)

    state_dict = torch.load(dgmr_model)
    dgmr.load_state_dict(state_dict, strict=False)

    # Get 4x4 transformation matrix from results
    registration_result = dgmr(pts_target, pts_source)
    transformation = registration_result['est_T'].cpu().detach().numpy()

    display_open3d(pts_target.detach().cpu().numpy()[0],
                   pts_source.detach().cpu().numpy()[0],
                   registration_result['transformed_source']
                   .detach()
                   .cpu()
                   .numpy()[0]
                   )

    return transformation


def iterative_closest_point(source: PointCloud, target: PointCloud, iterations: int = 1, **kwargs) -> np.array:
    raise NotImplementedError()


def normal_iterative_closest_point(source: PointCloud, target: PointCloud, iterations: int = 1, **kwargs) -> np.array:
    raise NotImplementedError()


def cluster_transform(transforms: List[np.array], algorithm: str = 'optics', **kwargs) -> np.array:
    # Only these algorithms are currently supported
    if algorithm not in ['dbscan', 'optics']:
        raise ValueError(
            algorithm, f'Clustering algorithm must be either dbscan or optics. Currently {algorithm}.')

    # Compute distance matrix for all transformation matrices by computing the norm of their difference.
    distance_matrix = euclidean_distance_matrix(transforms)

    # Use density-based clustering to find similar transformations.
    # Either OPTICS or DBSCAN are currently available.
    if algorithm == 'optics':
        clustering = cluster.OPTICS(max_eps=kwargs['max_eps'],
                                    min_samples=kwargs['min_samples'],
                                    metric='precomputed').fit(distance_matrix)
    elif algorithm == 'dbscan':
        raise NotImplementedError()

    labels = clustering.labels_
    return labels


def registration(source: PointCloud, target: PointCloud, algo: str, **kwargs) -> np.array:
    """Register two point clouds (find 4x4 transformation matrix that brings them into alignment) using various existing methods.

    Args:
                    source (PointCloud): Misaligned input point cloud
                    target (PointCloud): Input point cloud which source is aligned to
                    iterations (int, optional): How often to repeat the registration to refine results. Defaults to 1.
                    algo (str, optional): Algorithm used for registration. Defaults to 'dcp'. Currently accepts 'dcp', 'icp' and 'nicp'.

    Raises:
                    ArgumentError: Specified registration algorithm is not implemented.

    Returns:
                    np.array: 4x4 transformation matrix which aligns the source with the target point cloud.
    """

    # Available registration algorithms and their corresponding methods
    algo_methods = {'dgmr': deepgmr,
                    'pnlk': pointnet_lk,
                    'icp': iterative_closest_point,
                    'nicp': normal_iterative_closest_point}

    # Raise an error if the supplied registration algorithm is not available
    if algo not in algo_methods.keys():
        raise ValueError(
            algo, "Target registration algorithm is not currently implemented.")

    # Get corresponding method for supplied registration algorithm and execute
    registration_method = algo_methods[algo]
    estimated_transformation = registration_method(
        source, target, kwargs=kwargs)

    return estimated_transformation
