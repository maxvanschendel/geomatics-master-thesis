import itertools
from analysis.visualizer import MapViz, Viz
from model.topometric_map import *
import matplotlib.pyplot as plt


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

def spectral_embedding(voxel_grid: VoxelGrid, dim: float) -> np.array:
    return voxel_grid.shape_dna(Kernel.nb6(), dim)

def merge_maps(map_a: HierarchicalTopometricMap, map_b: HierarchicalTopometricMap, matches) -> HierarchicalTopometricMap:
    pass

def n_smallest_indices(input: np.array, n: int):
    smallest_flat = np.argpartition(input.ravel(), n)[:n]
    smallest_indices = [np.unravel_index( i, input.shape) for i in smallest_flat]
    
    return smallest_indices

def match_maps(map_a: HierarchicalTopometricMap, map_b: HierarchicalTopometricMap):
    global_dim = 50
    local_dim = 50
    n = 1
    use_local_features = False
    use_global_features = True
    
    if not use_local_features and not use_global_features:
        raise ValueError("Both feature embeddings disabled")
    
    rooms_a = map_a.get_node_level(Hierarchy.ROOM)
    rooms_b = map_b.get_node_level(Hierarchy.ROOM)
    a_size, b_size = len(rooms_a), len(rooms_b)
    
    
    
    # Use bag of features to embed local geometric features
    if use_local_features:
        print("Extracting local features")
        cb = codebook([a.geometry for a in rooms_a] + [b.geometry for b in rooms_b], local_dim)
        a_bof = [bag_of_features(room.geometry, cb) for room in rooms_a]
        b_bof = [bag_of_features(room.geometry, cb) for room in rooms_b]
        
    # use ShapeDNA to embed global geometric features
    if use_global_features:
        print("Extracting global features")
        a_spec = [spectral_embedding(a.geometry, global_dim) for a in rooms_a]
        b_spec = [spectral_embedding(b.geometry, global_dim) for b in rooms_b]
    
    # Concatenate local and global features into a single vector if both are enabled
    if use_global_features and use_local_features:
        a_embedding = [np.concatenate((a_spec[i],  a_bof[i])) for i in range(len(rooms_a))]
        b_embedding = [np.concatenate((b_spec[i],  b_bof[i])) for i in range(len(rooms_b))]
    elif use_local_features:
        a_embedding = a_bof
        b_embedding = b_bof
    else:
        a_embedding = a_spec
        b_embedding = b_spec


    # For every room pair, compute Euclidean distance in feature space
    print("Constructing distance matrix")
    g2v_distance_matrix = VoxelGrid.graph2vec([a.geometry for a in rooms_a] + [b.geometry for b in rooms_b])
    g2v_distance_matrix = g2v_distance_matrix[:a_size, a_size:]

    spectral_distance_matrix = np.empty((a_size, b_size), dtype=np.float)
    room_pairs = itertools.product(range(a_size), range(b_size))
    for i_a, i_b in room_pairs:
        if len(a_embedding[i_a]) == len(b_embedding[i_b]):
            spectral_distance_matrix[i_a][i_b] = np.linalg.norm(a_embedding[i_a] - b_embedding[i_b])
        else:
            spectral_distance_matrix[i_a][i_b] = math.inf

    distance_matrix = g2v_distance_matrix * spectral_distance_matrix
    
    # Find n room pairs with highest similarity
    print("Identifying most similar matches")
    n_most_similar = n_smallest_indices(distance_matrix, n)

    print(f"Identified matches: {n_most_similar}")

    line_set = o3d.geometry.LineSet()
    points, lines = [], []

    for i, n in enumerate(n_most_similar):
        i_a, i_b = n
        m_a = rooms_a[i_a]
        m_b = rooms_b[i_b]

        points.append(m_a.geometry.centroid())
        points.append(m_b.geometry.centroid())
        lines.append((i*2, i*2 + 1))

    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    viz = Viz([
        # Topometric map A visualization at room level
        [MapViz(o, Viz.pcd_mat(pt_size=6)) for o in map_a.to_o3d(Hierarchy.ROOM)[0]] +
        # [MapViz(map_a.to_o3d(Hierarchy.ROOM)[1], Viz.graph_mat())] +
        [MapViz(o, Viz.pcd_mat()) for o in map_a.to_o3d(Hierarchy.ROOM)[2]] +

        [MapViz(line_set, Viz.graph_mat(color=[0, 0, 1, 1]))] +

        # Topometric map B visualization at room level
        [MapViz(o, Viz.pcd_mat(pt_size=6)) for o in map_b.to_o3d(Hierarchy.ROOM)[0]] +
        # [MapViz(map_b.to_o3d(Hierarchy.ROOM)[1], Viz.graph_mat())] +
        [MapViz(o, Viz.pcd_mat()) for o in map_b.to_o3d(Hierarchy.ROOM)[2]
        ],
    ])

    # 1. detect local features for nodes
    # 2. generate bag of features for nodes
    # 3. embed topometric graph into feature space
    # 4. identify node matches
    # 5. register
