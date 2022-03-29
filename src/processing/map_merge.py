from analysis.visualizer import MapViz, Viz
from model.topometric_map import *


def draw_registration_result(source, target, transformation):
    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])
    source.transform(transformation)
    o3d.visualization.draw_geometries([source, target],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):

    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold, 10,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def fpfh(voxel_grid):
    """_summary_

    Args:
        voxel_grid (_type_): _description_

    Returns:
        _type_: _description_
    """

    voxel_size = voxel_grid.cell_size
    pcd = voxel_grid.to_pcd().to_o3d()

    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 3

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh.data


def codebook(data: List[VoxelGrid], n_words: int):
    """_summary_

    Args:
        data (List[VoxelGrid]): _description_
        n_words (int): _description_

    Returns:
        _type_: _description_
    """

    # https://ai.stackexchange.com/questions/21914/what-are-bag-of-features-in-computer-vision

    from sklearn.cluster import KMeans

    features = np.hstack([fpfh(vg) for vg in data])
    kmeans = KMeans(n_clusters=n_words, random_state=0).fit(features.T)
    centroids = kmeans.cluster_centers_

    return centroids


def bag_of_features(voxel_grid: VoxelGrid, codebook: np.array):
    """_summary_

    Args:
        voxel_grid (VoxelGrid): _description_
        codebook (np.array): _description_

    Returns:
        _type_: _description_
    """

    features = fpfh(voxel_grid)
    bof = np.zeros(codebook.shape[0])
    for f in features.T:
        centroid_distance = [np.linalg.norm(c - f) for c in codebook]
        nearest_centroid = np.argmin(centroid_distance)
        bof[nearest_centroid] += 1

    normalized_bof = bof / len(features.T)
    return normalized_bof


def embed_attributed_graph():
    # DANE / DMGI
    pass


def match_maps(map_a: HierarchicalTopometricMap, map_b: HierarchicalTopometricMap):
    rooms_a = map_a.get_node_level(Hierarchy.ROOM)
    rooms_b = map_b.get_node_level(Hierarchy.ROOM)

    cb = codebook([a.geometry for a in rooms_a] +
                  [b.geometry for b in rooms_b], 33)

    a_bof = [bag_of_features(room.geometry, cb) for room in rooms_a]
    b_bof = [bag_of_features(room.geometry, cb) for room in rooms_b]

    similarity_matrix = np.empty((len(a_bof), len(b_bof)), dtype=np.float)
    for i_a, a in enumerate(a_bof):
        for i_b, b in enumerate(b_bof):
            trans_init = np.asarray([[0.0, 0.0, -0.0, 0],
                                     [-0, 0, -0, 0],
                                     [0, 0, 0, 0], [0.0, 0.0, 0.0, 1.0]])
            source = rooms_a[i_a].geometry.to_pcd().to_o3d()
            target = rooms_b[i_b].geometry.to_pcd().to_o3d()
            threshold = .2
            reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            
            evaluation = o3d.pipelines.registration.evaluate_registration(
                source, target, threshold, trans_init)
            print(evaluation)

            # draw_registration_result(source, target, reg_p2p.transformation)

            similarity_matrix[i_a][i_b] = np.linalg.norm(a - b)

    import matplotlib.pyplot as plt
    plt.matshow(similarity_matrix)
    plt.colorbar()
    plt.savefig('sim.png')

    n = 3
    n_most_similar_flat = np.argpartition(similarity_matrix.ravel(), n)[:n]
    n_most_similar = [np.unravel_index(
        i, similarity_matrix.shape) for i in n_most_similar_flat]

    print(n_most_similar)

    line_set = o3d.geometry.LineSet()
    points, lines = [], []

    avg_linkage = []
    for i, n in enumerate(n_most_similar):
        i_a, i_b = n
        m_a = rooms_a[i_a]
        m_b = rooms_b[i_b]

        features_a = fpfh(m_a.geometry).T
        features_b = fpfh(m_b.geometry).T

        pair_dist = 0
        for f_a in features_a:
            for f_b in features_b:
                pair_dist += np.linalg.norm(f_a - f_b)
        avg_linkage.append(pair_dist / (len(features_a) * len(features_b)))

        points.append(m_a.geometry.centroid())
        points.append(m_b.geometry.centroid())
        lines.append((i*2, i*2 + 1))

    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    viz = Viz([
        # Topometric map A visualization at room level
        [MapViz(o, Viz.pcd_mat(pt_size=6)) for o in map_a.to_o3d(Hierarchy.ROOM)[0]] +
        [MapViz(map_a.to_o3d(Hierarchy.ROOM)[1], Viz.graph_mat())] +
        [MapViz(o, Viz.pcd_mat()) for o in map_a.to_o3d(Hierarchy.ROOM)[2]] +

        [MapViz(line_set, Viz.graph_mat(color=[0, 0, 1, 1]))] +

        # Topometric map B visualization at room level
        [MapViz(o, Viz.pcd_mat(pt_size=6)) for o in map_b.to_o3d(Hierarchy.ROOM)[0]] +
        [MapViz(map_b.to_o3d(Hierarchy.ROOM)[1], Viz.graph_mat())] +
        [MapViz(o, Viz.pcd_mat()) for o in map_b.to_o3d(Hierarchy.ROOM)[2]
         ],

    ])

    # 1. detect local features for nodes
    # 2. generate bag of features for nodes
    # 3. embed topometric graph into feature space
    # 4. identify node matches
    # 5. register
