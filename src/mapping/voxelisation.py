from plyfile import PlyData
import numpy as np

from model.point_cloud import PointCloud

def read_ply(fn):
    with open(fn, 'rb') as f:
        plydata = PlyData.read(f)

    num_points = plydata['vertex'].count

    points = np.zeros(shape=[num_points, 3], dtype=np.float32)
    points[:,0] = plydata['vertex'].data['x']
    points[:,1] = plydata['vertex'].data['y']
    points[:,2] = plydata['vertex'].data['z']

    return PointCloud(points)

def voxelize(pcd: np.array, size: float):
    pass

if __name__ == "__main__":
    fn = "C:/Users/maxva/Documents/RTAB-Map/cloud.ply"
    print(read_ply(fn))