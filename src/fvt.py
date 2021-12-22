import xml.etree.ElementTree as ET
import requests
from PIL import Image
from io import BytesIO
from py360convert import c2e
from pyproj import Proj, transform
import open3d as o3d
import numpy as np
import math
import time

class PanoramaMetadata:
    def __init__(self, id, date):
        self.id = id
        self.date = date


class Pose:
    def __init__(self, x, y, z, pitch, yaw, roll):
        self.position = (x, y, z)
        self.direction = (pitch, yaw, roll)

    def __str__(self):
        return f"{self.position},{self.direction}"


# List all Cyclomedia panorama IDs near address
def get_nearby_panoramas(country, address):
    get_by_adress_url = f"https://atlas.cyclomedia.com/PanoramaRendering/ListByaddress/{country}/{address}/"
    response = requests.get(url=get_by_adress_url, params={"apiKey": api_key}, auth=(usr, pwd))
    xml_tree = ET.fromstring(response.content)

    panoramas = []
    for panorama in xml_tree:
        recording_id = panorama.attrib['recording-id']
        recording_date = panorama.attrib['recording-date']

        panoramas.append(PanoramaMetadata(id=recording_id,
                                          date=recording_date))

    return panoramas


# Download georeferenced panorama by ID from Cyclomedia PanoramaRenderService API
def download_panorama(id):
    url = f"https://atlas.cyclomedia.com/panoramarendering/Render/{id}/"

    cubemap_parameters = [(0, -90), (0, 0), (0, 90), (0, 180), (90, 0), (-90, 0)]

    images = []
    for i, p in enumerate(cubemap_parameters):
        print(f"- Downloading image {i + 1}")
        params = {"apiKey": api_key,
                  "width": 1024,
                  "height": 1024,
                  "yaw": 0,
                  "pitch": p[0],
                  "direction": p[1],
                  "hfov": 90}

        r = requests.get(url=url, params=params, auth=(usr, pwd))

        if r.status_code == 200:
            if i == 1:
                header = r.headers
                print(header)
                inProj = Proj('epsg:4326')
                outProj = Proj('epsg:28992')

                x = float(header['RecordingLocation-x'])
                y = float(header['RecordingLocation-y'])
                z = float(header['RecordingLocation-z']) / 10
                x2, y2, z2 = transform(inProj, outProj, y, x, z)

                pitch = float(header['Render-Pitch'])
                yaw = float(header['Render-Yaw'])
                roll = float(header['Render-Roll'])

                panorama_pose = Pose(x2, y2, z2, pitch, yaw, roll)
        else:
            raise Exception(f"HTTP error {r.status_code}. Failed to download images.\nURL: {r.url}")

        image = Image.open(BytesIO(r.content))
        images.append(image)

    return images, panorama_pose


# Stitch 6 square images into a cubemap
def stitch_cubemap(images, size):
    new_im = Image.new('RGB', size)

    positions = [(0, 1024), (1024, 1024), (2048, 1024), (3072, 1024), (1024, 0), (1024, 2048)]

    for i, im in enumerate(images):
        new_im.paste(im, positions[i])

    return new_im


def stitch_panorama_depth(color, depth):
    new_im = Image.new('RGB', (2048, 2048))
    new_im.paste(color, (0,0))
    new_im.paste(depth, (0, 1024))

    return new_im


def degrees_to_radians(degree: float):
    return degree * (math.pi / 180)


# Generate points on a sphere with a given radius, position and rotation
def points_on_sphere(n_theta: int, n_phi: int,
                     position: (float, float, float),
                     rotation: (float, float, float),
                     radius: float):
    print(f"Generating points on sphere with r={radius}, pos={position}, rot={rotation}")
    angle_theta = math.pi / n_theta
    angle_phi = (2 * math.pi) / n_phi

    sphere_points = []
    for i in range(1,n_theta-1):
        for j in range(n_phi):
            theta = (n_theta - i) * angle_theta + degrees_to_radians(rotation[0])
            phi = (n_phi - j) * angle_phi + degrees_to_radians(rotation[1] / 2) + degrees_to_radians(180)

            x = radius * math.sin(theta) * math.cos(phi) + position[0]
            y = radius * math.sin(theta) * math.sin(phi) + position[1]
            z = radius * math.cos(theta) + position[2]

            sphere_points.append((x, y, z))

    return np.array(sphere_points)


# Get the parameters of the line equation of a line
# Passing through two points
def line_equation(a, b):
    d = b - a

    x1, l = a[0], d[0]
    y1, m = a[1], d[1]
    z1, n = a[2], d[2]

    return x1, y1, z1, l, m, n


# Get cartesian coordinates of voxel at index based on grid origin and voxel size
def voxel_coordinates(current_voxel, voxel_size, voxel_origin):
    return voxel_origin + (current_voxel * voxel_size)


# "Fast" voxel traversal, based on Amanitides & Woo, 1987
# Terminates when first voxel has been found
def dda(point, sphere_center):

    x1, y1, z1, l, m, n = line_equation(point, sphere_center)

    if (l == 0.0):
        l = 0.00000001
    if (m == 0.0):
        m = 0.00000001
    if (n == 0.0):
        n = 0.00000001

    x_sign = l / abs(l)
    y_sign = m / abs(m)
    z_sign = n / abs(n)

    axis_signs = np.array([x_sign, y_sign, z_sign])
    border_distances = axis_signs * voxel_size

    current_position = np.array(camera_location)
    current_voxel = voxel_grid.get_voxel(current_position)

    voxel_found = False

    count = 0
    while (min_bbox[0] < current_position[0] < max_bbox[0] and
           min_bbox[1] < current_position[1] < max_bbox[1] and
           min_bbox[2] < current_position[2] < max_bbox[2] and
           voxel_found is False):

        current_voxel_center = voxel_coordinates(current_voxel, voxel_size, voxel_origin)

        border_plane_coordinates = current_voxel_center + border_distances

        # get coordinates of axis-aligned planes that border cell
        x_edge = border_plane_coordinates[0]
        y_edge = border_plane_coordinates[1]
        z_edge = border_plane_coordinates[2]

        # find intersection of line with cell borders and its distance
        x_intersection = np.array((x_edge, (((x_edge - x1) / l) * m) + y1, (((x_edge - x1) / l) * n) + z1))
        x_vector = x_intersection - current_position
        x_magnitude = np.linalg.norm(x_vector)

        y_intersection = np.array(((((y_edge - y1) / m) * l) + x1, y_edge, (((y_edge - y1) / m) * n) + z1))
        y_vector = y_intersection - current_position
        y_magnitude = np.linalg.norm(y_vector)

        z_intersection = np.array(((((z_edge - z1) / n) * l) + x1, (((z_edge - z1) / n) * m) + y1, z_edge))
        z_vector = z_intersection - current_position
        z_magnitude = np.linalg.norm(z_vector)

        if x_magnitude < y_magnitude and x_magnitude < z_magnitude:
            current_voxel[0] += x_sign
            current_position += x_vector
        elif y_magnitude < x_magnitude and y_magnitude < z_magnitude:
            current_voxel[1] += y_sign
            current_position += y_vector
        elif z_magnitude < x_magnitude and z_magnitude < y_magnitude:
            current_voxel[2] += z_sign
            current_position += z_vector

        if tuple(current_voxel) in voxel_indices:
            voxel_found = True

        count += 1

    if voxel_found is True:
        voxel_distance = np.linalg.norm(current_position - sphere_center)
        distances.append(voxel_distance)
        points.append(current_position)

        line_points.append(sphere_points[i])
        line_points.append(current_position)
        lines.append([len(line_points) - 2, len(line_points) - 1])
    else:
        distances.append(max_dist)


if __name__ == "__main__":
    # Cyclomedia login details
    usr = "maxvanschendel@gmail.com"
    pwd = "3d9sy3wl"
    api_key = ""

    # location parameters
    country = "NL"
    address = "Amsterdam Damstraat 5"

    # panorama parameters
    image_size = (1024, 1024)
    cubemap_size = (image_size[0] * 4, image_size[1] * 3)

    # voxelization and depth map parameters
    input_point_cloud = "./Damstraat_cleaned.ply"
    voxel_size = 0.2
    x_dim = 100
    y_dim = int(x_dim / 2)
    max_dist = 50

    download_panoramas = True

    if download_panoramas:
        ### panorama ###
        nearby_panoramas = get_nearby_panoramas(country, address)
        print(f"Found {len(nearby_panoramas)} panoramas near {address}")

        # download panorama images
        closest_panorama_id = nearby_panoramas[0].id
        images, pose = download_panorama(closest_panorama_id)
        camera_location, camera_rotation = pose.position, pose.direction
        print(f"Downloaded panorama {closest_panorama_id}")

        # stitch images into cubemap
        cubemap = stitch_cubemap(images, cubemap_size)
        cubemap.save(f'{closest_panorama_id}_cubemap.jpg')
        print("Created cubemap from downloaded images")

        # convert cubemap to equirectangular panorama
        equirectangular = c2e(np.asarray(cubemap, np.uint8), 1024, 2048)
        equirectangular_image = Image.fromarray(equirectangular.astype(np.uint8))
        equirectangular_image.save(f'{closest_panorama_id}_equirectangular.jpg')
        print("Created equirectangular panorama from cubemap")
    else:
        camera_location = (121440.95894741485, 487275.44876150216, 4.6726)
        camera_rotation = (0.0, -64.009, 0.0)
        closest_panorama_id = "WE1QUWZ2"

    ### depth map ###
    pcd = o3d.io.read_point_cloud(input_point_cloud)
    print('Read input point cloud')

    min_bbox = pcd.get_min_bound()
    max_bbox = pcd.get_max_bound()
    center_bbox = (min_bbox + max_bbox) / 2

    camera_position_indicator = o3d.geometry.TriangleMesh.create_sphere(radius=0.1).translate(camera_location)

    # construct voxel grid and get filled voxel indices
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    voxels = voxel_grid.get_voxels()
    voxel_indices = set(map(lambda x: tuple(x.grid_index), list(voxels)))
    voxel_origin = voxel_grid.origin
    print('Voxelized point cloud')

    # get points on sphere that correspond to pixels in the panorama
    sphere_points = points_on_sphere(y_dim, x_dim, camera_location, camera_rotation, 1)
    sphere_center = np.array(camera_location)

    # measure dda execution time
    start_time = time.time()

    distances = []
    points = []
    lines = []
    line_points = []

    for i, point in enumerate(sphere_points):
        if i % 100 == 0:
            print(f"Casting {i} out of {x_dim * y_dim}")

        dda(point, sphere_center)

    execution_time = time.time() - start_time
    print(f"Finished voxel traversal in {execution_time} seconds."
          f"\nAverage time per line is {execution_time / len(distances)}")

    dist = np.array(distances).reshape((y_dim-2, x_dim))
    dist_interp = np.interp(dist, (0, dist.max()), (1, 0))

    im = Image.fromarray(np.uint8(dist_interp * 255))
    im = im.resize((2048, 1024))

    stitch_im = stitch_panorama_depth(equirectangular_image, im)
    im_fn = f"{closest_panorama_id}_equirectangular_depth.png"
    stitch_im.save(im_fn)

    print(f"Saved depth map to {im_fn}")

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)

    pcl_sphere = o3d.geometry.PointCloud()
    pcl_sphere.points = o3d.utility.Vector3dVector(sphere_points)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    o3d.visualization.draw_geometries([pcl, pcl_sphere, pcd])
