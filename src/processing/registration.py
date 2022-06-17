
from model.point_cloud import PointCloud

import numpy as np
import open3d as o3d
from numba import cuda


dcp_model = './learning3d/pretrained/exp_dcp/models/best_model.t7'
pnlk_model = './learning3d/pretrained/exp_pnlk/models/best_model.t7'


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

	import torch
	from learning3d.models import DCP, DGCNN

	torch.cuda.empty_cache()
	device = torch.device("cuda")

	# Convert numpy array point clouds to Torch tensor
	pts_source = source.to_tensor().to(device)
	pts_target = target.to_tensor().to(device)
	
	# load DCP model
	dgcnn = DGCNN(emb_dims=512).to(device)
	dcp = DCP(feature_model=dgcnn, cycle=True).to(device)
	dcp.load_state_dict(torch.load(dcp_model), strict=False)

	# iteratively refine registration.
	for _ in range(iterations):
		registration_result = dcp(pts_source, pts_target)
		pts_source = registration_result['transformed_source']

		display_open3d(pts_target.detach().cpu().numpy()[0],
						pts_source.detach().cpu().numpy()[0],
						registration_result['transformed_source']
							.detach()
							.cpu()
							.numpy()[0]
				   )

	# Get 4x4 transformation matrix from results
	transformation = registration_result['est_T'].cpu().detach().numpy()
	return transformation


def pointnet_lk(source: PointCloud, target: PointCloud, **kwargs) -> np.array:
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

	import torch
	from learning3d.models.pointnet import PointNet
	from learning3d.models.pointnetlk import PointNetLK

	torch.cuda.empty_cache()
	device = torch.device("cuda")

	pts_source = source.to_tensor().to(device)
	pts_target = target.to_tensor().to(device)

	feature_model = PointNet(use_bn=True, global_feat=False).to(device)
	pnlk = PointNetLK(feature_model=feature_model,
					  delta=1e-03,
					  xtol=1e-07,
					  p0_zero_mean=False,
					  p1_zero_mean=False,
					  pooling='max',
					  learn_delta=False)
 
	state_dict = torch.load(pnlk_model)
	pnlk.load_state_dict(state_dict, strict=False)

	# Get 4x4 transformation matrix from results
	registration_result = pnlk(pts_target, pts_source, maxiter=100)
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


def registration(source: PointCloud, target: PointCloud, iterations: int = 1, algo: str = 'pnlk', **kwargs) -> np.array:
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
