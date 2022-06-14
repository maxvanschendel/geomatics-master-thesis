from model.point_cloud import PointCloud

import numpy as np
from torch import from_numpy as as_tensor

from utils.validation import kwargs_valid


def deep_closest_point(source: PointCloud, target: PointCloud, iterations: int = 1, **kwargs) -> np.array:
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

	# if not kwargs_valid(['pointer', 'head'], kwargs):
	#     raise ArgumentError(
	#         'Missing required keyword arguments for Deep Closest Point registration.')

	import torch
	from learning3d.models import DCP, DGCNN
	
	# Number of points must be equal for both point clouds
	# so we remove the additional points from the larger point cloud
	n_points = min([len(source.points), len(target.points)])

	# Convert numpy array point clouds to Torch tensor
	pts_source = as_tensor(
		source.points[:n_points, :][np.newaxis, ...]).float()
	pts_target = as_tensor(
		target.points[:n_points, :][np.newaxis, ...]).float()

	# Deep Closest Point model.
	dgcnn = DGCNN(emb_dims=512)
	dcp = DCP(feature_model=dgcnn, cycle=True)
	dcp.load_state_dict(torch.load('./learning3d/pretrained/exp_dcp/models/best_model.t7', map_location=torch.device('cpu')), 
                     	strict=False)

	# Iteratively refine registration.
	for _ in range(iterations):
		dcp_registration = dcp(pts_source, pts_target)
		pts_source = dcp_registration['transformed_source']

	# Get 4x4 transformation matrix from results
	estimated_transformation = dcp_registration['est_T'].detach().numpy()
	return estimated_transformation


def iterative_closest_point(source: PointCloud, target: PointCloud, iterations: int = 1, **kwargs) -> np.array:
	raise NotImplementedError()


def normal_iterative_closest_point(source: PointCloud, target: PointCloud, iterations: int = 1, **kwargs) -> np.array:
	raise NotImplementedError()


def registration(source: PointCloud, target: PointCloud, iterations: int = 1, algo: str = 'dcp', **kwargs) -> np.array:
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
	algo_methods = {'dcp': deep_closest_point,
					'icp': iterative_closest_point,
					'nicp': normal_iterative_closest_point}

	# Raise an error if the supplied registration algorithm is not available
	if algo not in algo_methods.keys():
		raise ValueError(
			algo, "Target registration algorithm is not currently implemented.")

	# Get corresponding method for supplied registration algorithm and execute
	registration_method = algo_methods[algo]
	estimated_transformation = registration_method(
		source, target, iterations=iterations, kwargs=kwargs)

	return estimated_transformation
