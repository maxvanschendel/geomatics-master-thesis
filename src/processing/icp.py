import numpy as np
from numba import jit
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import logging
import math

def fpfh_features(pts: np.array):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals()

    return o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=100))


def _transform(source: np.array, R: np.array, T: np.array):
    points = []
    for point in source:
        points.append(np.dot(R, point.reshape(-1, 1)+T))
    return points


def compute_rmse(source: np.array, target: np.array, R: np.array, T: np.array, tree):
    points = _transform(source, R, T)
    
    dist, nearest = tree.query(np.array(points).squeeze())
    return np.mean(dist)


def registration_RANSAC(source: np.array, target: np.array, source_feature: np.array, target_feature: np.array, normals: np.array, ransac_n=3, max_iteration=1000) -> np.array:
    """

    Adapted from: https://github.com/withtimesgo1115/3D-point-cloud-global-registration-based-on-RANSAC

    Args:
        s (_type_): _description_
        t (_type_): _description_
        source_feature (_type_): _description_
        target_feature (_type_): _description_
        normals (_type_): _description_
        ransac_n (int, optional): _description_. Defaults to 5.
        max_iteration (int, optional): _description_. Defaults to 200.

    Returns:
        np.array: _description_
    """

    from scipy import spatial
    from random import randint

    
    def triangle_edge_lengths(vxs):
        ab = np.linalg.norm(vxs[0] - vxs[1])
        bc = np.linalg.norm(vxs[1] - vxs[2])
        ca = np.linalg.norm(vxs[2] - vxs[0])
        
        return np.array([ab, bc, ca])
       
    # the intention of RANSAC is to get the optimal transformation between the source and target point cloud
    sf = np.transpose(source_feature.data)
    tf = np.transpose(target_feature.data)
    
    tree = spatial.KDTree(tf)
    dist, corres_stock = tree.query(sf)
    
    opt_rmse = math.inf
    
    nn_tree = spatial.KDTree(target)


    for i in range(max_iteration):
        # take ransac_n points randomly
        idx = [randint(0, source.shape[0]-1) for j in range(ransac_n)]
        corres_idx = corres_stock[idx]
        source_point = source[idx, ...]
        target_point = target[corres_idx, ...]
        
        source_edge_lengths = triangle_edge_lengths(source_point)
        target_edge_lengths = triangle_edge_lengths(target_point)
        
        edge_ratio = source_edge_lengths * (1/target_edge_lengths)
        max_ratio = .8
        if edge_ratio[edge_ratio < max_ratio].any() or edge_ratio[edge_ratio > 1/max_ratio].any():
            continue
            
        # estimate transformation
        transform, R, T = best_fit_transform(
            source_point, target_point, normals)

        # calculate rmse for all points
        # source_point = source
        # target_point = target[corres_stock, ...]
        rmse = compute_rmse(source, target, R, T, nn_tree)
        
        # compare rmse and optimal rmse and then store the smaller one as optimal values
        if not i:
            opt_rmse = rmse
            opt_t = transform
        else:
            if rmse < opt_rmse:
                opt_rmse = rmse
                opt_t = transform
                
        logging.info(f"{i}/{max_iteration}: {rmse}")

    return opt_t


def best_fit_transform(P, Q, normals):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in 3 spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert P.shape == Q.shape
    from scipy.spatial.transform import Rotation as R

    L = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    A = np.zeros((4, 4))
    b = np.zeros((4, 1))

    for k in range(P.shape[0]):
        p_k, q_k, n_k = P[k], Q[k], normals[k]

        ldot = L.dot(p_k.T)
        c_k = ldot.dot(n_k)

        Av = np.array([[c_k], [n_k[0]], [n_k[1]], [n_k[2]]])
        Ah = np.array([[c_k, n_k[0], n_k[1], n_k[2]]])
        A += Av * Ah

        d_k = (q_k - p_k)
        b += Av * np.dot(d_k, n_k)

    tau = np.linalg.lstsq(A, b)[0]

    t, gamma = tau[1:], tau[0]
    r = R.from_rotvec([0, gamma, 0]).as_matrix()

    T = np.hstack([r, t])
    T = np.vstack([T, [0, 0, 0, 1]])

    return T, r, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)

    return distances.ravel(), indices.ravel()


def knn(src, n):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    neigh = NearestNeighbors(n_neighbors=n)
    neigh.fit(src)
    _, indices = neigh.kneighbors(src, return_distance=True)

    return indices


def fit_plane(pts):
    pca = PCA(n_components=3)
    components = pca.fit(pts).components_

    nbs_normal = components[2]
    return nbs_normal


def estimate_normals(pcd, n_neighbours):
    def local_plane(pcd, neighbour_indices): return fit_plane(
        pcd[neighbour_indices, :])

    pt_neighbours_indices = knn(pcd, n_neighbours)
    normals = np.array([local_plane(pcd, pts)
                       for pts in pt_neighbours_indices])

    return normals


def align_global(A, B, n_neighbours=8, ransac_iterations=1000):
    normals = estimate_normals(A, n_neighbours)
    global_transformation = registration_RANSAC(
        A, B, fpfh_features(A), fpfh_features(B), normals, max_iteration=ransac_iterations)

    return global_transformation


def icp(A, B, max_iterations=1000, tolerance=0.001, n_neighbours=8, ransac_iterations=1000, global_align: bool=True):
    '''
    Adapted from: https://github.com/ClayFlannigan/icp

    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    from functools import reduce

    assert A.shape == B.shape

    transformations = []

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # initialize
        
    if global_align:
        logging.info(f"Finding global alignment")
        global_transformation = align_global(A, B, ransac_iterations=ransac_iterations)
        src = np.dot(global_transformation, src)
        transformations.append(global_transformation)

    logging.info(f"Finding local alignment")
    prev_error = math.inf
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        normals = estimate_normals(src[:m, :].T, n_neighbours)

        # compute the transformation between the current source and nearest destination points -> update the current source
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T, normals)
        src = np.dot(T, src)
        transformations.append(T)

        # check error
        mean_error = np.mean(distances)
        if prev_error - mean_error < tolerance:
            break
        prev_error = mean_error
        
        logging.info(f"{i}/{max_iterations}: {mean_error}")

    composite_transformation = reduce(
        lambda a, b: a.dot(b), reversed(transformations))
    
    return composite_transformation, mean_error


if __name__ == '__main__':
    from utils.datasets import SimulatedPartialMap
    from processing.registration import iterative_closest_point

    dataset_a = '../data/s3dis/area_1/area_1_partial_01.pickle'
    dataset_b = '../data/s3dis/area_1/area_1_partial_02.pickle'

    partial_map_a = SimulatedPartialMap.read(dataset_a)
    partial_map_b = SimulatedPartialMap.read(dataset_b)

    pcd_a = partial_map_a.voxel_grid.level_of_detail(2).to_pcd().center()
    pcd_b = partial_map_b.voxel_grid.level_of_detail(2).to_pcd().center()

    iterative_closest_point(pcd_a, pcd_b)
