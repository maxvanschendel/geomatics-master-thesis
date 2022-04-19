from argparse import ArgumentError
from typing import List
from model.point_cloud import PointCloud
from learning3d.models import PointNet, PointNetLK, DCP, iPCRNet, PRNet, PPFNet, RPMNet
import numpy as np
import torch


def deep_closest_point(source: PointCloud, target: PointCloud, iterations: int = 1) -> np.array:
    """
    Register two point clouds using Deep Closest Point (DCP) model. 
    https://arxiv.org/abs/1905.03304

    Args:
        source (PointCloud): Misaligned input point cloud
        target (PointCloud): Input point cloud which source is aligned to
        iterations (int, optional): How often to repeat the registration to refine results. Defaults to 1.

    Returns:
        np.array: 4x4 transformation matrix which aligns the source with the target point cloud.
    """

    # Convert numpy array point clouds to Torch tensor
    pts_source = torch.from_numpy(
        source.points[:500, :][np.newaxis, ...]).float()
    pts_target = torch.from_numpy(
        target.points[:500, :][np.newaxis, ...]).float()

    # Deep Closest Point model.
    dcp = DCP()

    # Iteratively refine registration.
    for _ in range(iterations):
        dcp_registration = dcp(pts_source, pts_target)
        pts_source = dcp_registration['transformed_source']

    estimated_transformation = dcp_registration['est_T'].detach().numpy()
    return estimated_transformation


def iterative_closest_point():
    pass


def normal_iterative_closest_point():
    pass

def kwargs_valid(contains: List[str], **kwargs):
    return set(contains).issubset(kwargs.keys())

def registration(source: PointCloud, target: PointCloud, iterations: int = 1, algo: str = 'dcp', **kwargs) -> np.array:
    """Register two point clouds (find 4x4 transformation matrix that brings them into alignment) using various existing methods.

    Args:
        source (PointCloud): Misaligned input point cloud
        target (PointCloud): Input point cloud which source is aligned to
        iterations (int, optional): How often to repeat the registration to refine results. Defaults to 1.
        algo (str, optional): Algorithm used for registration. Defaults to 'dcp'. Currently accepts 'dcp', 'icp' and 'nicp'.

    Raises:
        NotImplementedError: Specified registration algorithm is not implemented.

    Returns:
        np.array: 4x4 transformation matrix which aligns the source with the target point cloud.
    """

    if algo == 'dcp':
        if not kwargs_valid(['pointer', 'head'], kwargs):
            raise ArgumentError(
                f'Missing required keyword arguments: {['pointer', 'head']} for Deep Closest Point registration.')

        # Apply deep closest point registration
        estimated_transformation = deep_closest_point(
            source, target, iterations, kwargs['pointer'], kwargs['head'])

    elif algo == 'icp':
        raise NotImplementedError()
    elif algo == 'nicp':
        raise NotImplementedError()
    else:
        raise NotImplementedError(
            "Target registration algorithm is not currently implemented.")

    return estimated_transformation
