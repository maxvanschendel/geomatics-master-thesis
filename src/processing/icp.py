import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from numba import jit

# https://github.com/withtimesgo1115/3D-point-cloud-global-registration-based-on-RANSAC

def feature(pts):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals()

    return o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=100))

@jit
def _transform(source, R, T):
    points = []
    for point in source:
        points.append(np.dot(R, point.reshape(-1, 1)+T))
    return points

@jit
def compute_rmse(source, target, R, T):
    rmse = 0
    number = len(target)
    points = _transform(source, R, T)
    for i in range(number):
        error = target[i].reshape(-1, 1)-points[i]
        rmse += (error[0]**2 + error[1]**2 + error[2]**2)**(1/2)
    return (rmse / number)**(1/2)


def registration_RANSAC(s, t, source_feature, target_feature, normals, ransac_n=5, max_iteration=200, max_validation=100):
    from scipy import spatial
    from random import randint

    # the intention of RANSAC is to get the optimal transformation between the source and target point cloud

    sf = np.transpose(source_feature.data)
    tf = np.transpose(target_feature.data)
    tree = spatial.KDTree(tf)
    corres_stock = tree.query(sf)[1]

    for i in range(max_iteration):
        # take ransac_n points randomly
        idx = [randint(0, s.shape[0]-1) for j in range(ransac_n)]
        corres_idx = corres_stock[idx]
        source_point = s[idx, ...]
        target_point = t[corres_idx, ...]

        # estimate transformation
        transform, R, T = best_fit_transform(source_point, target_point, normals)

        # calculate rmse for all points
        source_point = s
        target_point = t[corres_stock, ...]
        rmse = compute_rmse(source_point, target_point, R, T)
        # compare rmse and optimal rmse and then store the smaller one as optimal values
        if not i:
            opt_rmse = rmse
            opt_t = transform
        else:
            if rmse < opt_rmse:
                opt_rmse = rmse
                opt_t = transform

        print(opt_rmse, rmse)

    return opt_t

def best_fit_transform(P, Q, normals):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
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

    # get number of dimensions
    m = P.shape[1]

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


def icp(A, B, max_iterations=1000, tolerance=0.001, n_neighbours=8):
    '''
    Original code from: https://github.com/ClayFlannigan/icp

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

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    pt_neighbours_indices = knn(src[:m, :].T, n_neighbours)
    normals = np.array([fit_plane(src[:m, pts].T) for pts in pt_neighbours_indices])
    
    init_pose = registration_RANSAC(A, B, feature(A), feature(B), normals)
    src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        print(f'{i}/{max_iterations}')

        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        pt_neighbours_indices = knn(src[:m, :].T, n_neighbours)
        normals = np.array([fit_plane(src[:m, pts].T) for pts in pt_neighbours_indices])

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T, normals)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        print(mean_error)
        
        if np.abs(prev_error - mean_error) < tolerance:
            break
        
        prev_error = mean_error

    pt_neighbours_indices = knn(A, n_neighbours)
    normals = np.array([fit_plane(A[pts, :]) for pts in pt_neighbours_indices])

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T, normals)
    
    return T, src[:m, :].T


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
