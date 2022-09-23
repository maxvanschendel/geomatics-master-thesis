from dataclasses import dataclass
from random import random


import numpy as np
import open3d as o3d
from typing import List
from model.point_cloud import PointCloud
from model.voxel_grid import VoxelGrid

from utils.io import load_pickle, select_file


@dataclass
class SceneObject:
    geometry: o3d.geometry
    material: o3d.visualization.rendering.MaterialRecord


class MultiViewScene:
    def pcd_mat(pt_size=3, base_color=[1, 1, 1, 1]):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.point_size = pt_size
        mat.base_color = base_color
        mat.shader = 'defaultLit'

        return mat

    def graph_mat(color=[0., 0., 0., 1.0]):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = .5
        mat.base_color = color

        return mat

    def __init__(self, maps: List[List[SceneObject]], name: str = "", run: bool = False):
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        gui = o3d.visualization.gui
        window = gui.Application.instance.create_window(
            name, 1025, 512)
        widgets = []

        def on_mouse(m):
            if m.type == m.Type.DRAG:
                w_index = int(
                    m.x // (window.content_rect.width / len(widgets)))
                if w_index < len(widgets):
                    w_focus = widgets[w_index]

                    for w_other in widgets:
                        if w_other != w_focus:
                            w_other.scene.camera.copy_from(
                                w_focus.scene.camera)

            return gui.SceneWidget.EventCallbackResult.IGNORED

        def on_layout(theme):
            r = window.content_rect

            for i, widget in enumerate(widgets):
                widget.frame = gui.Rect(
                    r.x + ((r.width / len(widgets)) * i), r.y, r.width / len(widgets), r.height)

        for submap_index, submap in enumerate(maps):
            submap_widget = gui.SceneWidget()
            submap_widget.scene = o3d.visualization.rendering.Open3DScene(
                window.renderer)

            submap_widget.scene.set_background(np.asarray([1, 1, 1, 1]))

            for object_i, submap_object in enumerate(submap):
                submap_widget.scene.add_geometry(
                    f"{submap_index}_{object_i}", submap_object.geometry, submap_object.material)

            submap_widget.setup_camera(
                3, submap_widget.scene.bounding_box, submap_widget.scene.bounding_box.get_center())
            submap_widget.set_on_mouse(on_mouse)

            submap_widget.look_at(submap_widget.scene.bounding_box.get_center(
            ), submap_widget.scene.bounding_box.get_center() + np.array([0, 1250, 0]), np.array([1, 0, 0]))

            import matplotlib
            matplotlib.use("TkAgg")

            from matplotlib import pyplot as plt

            img = app.render_to_image(submap_widget.scene, 2048, 2048)
            plt.imshow(np.asarray(img))
            plt.imsave(f'{name}_{submap_index}.jpg', np.asarray(img))

            window.add_child(submap_widget)
            widgets.append(submap_widget)

        if run:
            window.set_on_layout(on_layout)
            app.run()


pcd_mat = MultiViewScene.pcd_mat


def random_color(alpha: bool = False) -> List[float]:
    return [random(), random(), random()]


def visualize_point_cloud(point_cloud: PointCloud):
    MultiViewScene(
        [[SceneObject(point_cloud.to_o3d(), pcd_mat(pt_size=6))]])


def visualize_point_clouds(point_clouds: List[PointCloud]):
    MultiViewScene(
        [[SceneObject(point_cloud.to_o3d(), pcd_mat(pt_size=6))] for point_cloud in point_clouds])


def visualize_voxel_grid(map: VoxelGrid, color=True, fn=''):
    pcd_map = map.to_pcd(has_color=color).to_o3d()
    vg_map = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd_map, map.cell_size)

    MultiViewScene([
        [SceneObject(vg_map, pcd_mat(pt_size=6))]
    ], name = fn)


def visualize_visibility(map: VoxelGrid, origins):
    pcd_map = map.to_pcd(has_color=False).to_o3d()
    vg_map = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd_map, map.cell_size)

    origin_map = o3d.geometry.PointCloud()
    origin_map.points = o3d.utility.Vector3dVector(origins)

    MultiViewScene([
        [SceneObject(vg_map, pcd_mat(pt_size=6)),
         SceneObject(origin_map, pcd_mat(pt_size=24, base_color=[1, 0, 0, 1]))]
    ])


def visualize_htmap(map, capture_screen_image_name: str):
    geometry, graph, _ = map.to_o3d(randomize_color=True, voxel=False)

    MultiViewScene([
        [SceneObject(o, pcd_mat(pt_size=2)) for o in geometry] + [SceneObject(graph, MultiViewScene.graph_mat())], ], name=capture_screen_image_name)


def visualize_map_merge(map_a, map_b):
    map_a_o3d = map_a.to_o3d()
    map_b_o3d = map_b.to_o3d()

    MultiViewScene([
        # Topometric map A visualization at room level
        [SceneObject(o, pcd_mat(pt_size=6)) for o in map_a_o3d] +
        [SceneObject(o, pcd_mat()) for o in map_a_o3d[2]] +

        # Topometric map B visualization at room level
        [SceneObject(o, pcd_mat(pt_size=6)) for o in map_b_o3d[0]] +
        [SceneObject(o, pcd_mat()) for o in map_b_o3d[2]
         ],
    ])


def visualize_matches(map_a, map_b, matches, capture_screen_image_name: str):
    for room_a, room_b in matches:
        color = random_color()
        room_a.geometry.colorize(color)
        room_b.geometry.colorize(color)

    geometry_a, lines_a, _ = map_a.to_o3d(randomize_color=False, voxel=False)
    geometry_b, lines_b, _ = map_b.to_o3d(randomize_color=False, voxel=False)

    MultiViewScene([
            [SceneObject(o, pcd_mat()) for o in geometry_a] +
            [SceneObject(lines_a, MultiViewScene.graph_mat())],
            [SceneObject(o, pcd_mat()) for o in geometry_b] + [SceneObject(lines_b, MultiViewScene.graph_mat())]],
        name=capture_screen_image_name)


if __name__ == '__main__':
    from model.topometric_map import TopometricMap

    fn = select_file()

    if fn.endswith('.pickle'):
        map = load_pickle(fn)
        map_type = type(map)

        if map_type == TopometricMap:
            visualize_htmap(map)
        elif map_type == PointCloud:
            visualize_point_cloud(map)

    elif fn.endswith('.ply'):
        map = PointCloud.read_ply(fn)
        visualize_point_cloud(map)
