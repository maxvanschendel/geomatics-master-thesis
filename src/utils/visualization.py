from dataclasses import dataclass
from random import random
import numpy as np
import open3d as o3d
from typing import List
from model.point_cloud import PointCloud
from model.voxel_grid import VoxelGrid

from utils.io import load_pickle, select_file


@dataclass
class MapVisualization:
    geometry: o3d.geometry
    material: o3d.visualization.rendering.MaterialRecord


class Visualization:
    def pcd_mat(pt_size=7, base_color = [1,1,1,1]):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.point_size = pt_size
        mat.base_color = base_color
        mat.shader = 'defaultLit'

        return mat

    def graph_mat(color=[1.0, 0, 0, 1.0]):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 5
        mat.base_color = color

        return mat

    def __init__(self, maps: List[List[MapVisualization]]):
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        gui = o3d.visualization.gui
        window = gui.Application.instance.create_window(
            "Two scenes", 1025, 512)
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

            for object_i, submap_object in enumerate(submap):
                submap_widget.scene.add_geometry(
                    f"{submap_index}_{object_i}", submap_object.geometry, submap_object.material)

            submap_widget.setup_camera(
                60, submap_widget.scene.bounding_box, (0, 0, 0))
            submap_widget.set_on_mouse(on_mouse)

            window.add_child(submap_widget)
            widgets.append(submap_widget)

        window.set_on_layout(on_layout)
        app.run()


def random_color(alpha: bool = False) -> List[float]:
    return [random(), random(), random()]


def visualize_point_cloud(point_cloud: PointCloud):
    Visualization(
        [[MapVisualization(point_cloud.to_o3d(), Visualization.pcd_mat(pt_size=6))]])

def visualize_point_clouds(point_clouds: List[PointCloud]):
    Visualization(
        [[MapVisualization(point_cloud.to_o3d(), Visualization.pcd_mat(pt_size=6))] for point_cloud in point_clouds])


def visualize_voxel_grid(map: VoxelGrid):
    from model.topometric_map import Hierarchy

    pcd_map = map.to_pcd(color=False).to_o3d()
    vg_map = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_map, map.cell_size)
    
    Visualization([
        [MapVisualization(vg_map, Visualization.pcd_mat(pt_size=6))]
    ])
    
def visualize_visibility(map: VoxelGrid, origins):
    from model.topometric_map import Hierarchy

    pcd_map = map.to_pcd(color=False).to_o3d()
    vg_map = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_map, map.cell_size)
    
    origin_map = o3d.geometry.PointCloud()
    origin_map.points = o3d.utility.Vector3dVector(origins)
    
    Visualization([
        [MapVisualization(vg_map, Visualization.pcd_mat(pt_size=6)), MapVisualization(origin_map, Visualization.pcd_mat(pt_size=24, base_color=[1,0,0,1]))]
    ])
    
def visualize_htmap(map):
    from model.topometric_map import Hierarchy

    Visualization([
        [MapVisualization(o, Visualization.pcd_mat(pt_size=6))
         for o in map.to_o3d(Hierarchy.ROOM)[0]]
    ])


def visualize_map_merge(map_a, map_b):
    from model.topometric_map import Hierarchy

    rooms_a = map_a.get_node_level(Hierarchy.ROOM)
    rooms_b = map_b.get_node_level(Hierarchy.ROOM)

    visualization = Visualization([
        # Topometric map A visualization at room level
        [MapVisualization(o, Visualization.pcd_mat(pt_size=6)) for o in map_a.to_o3d(Hierarchy.ROOM)[0]] +
        [MapVisualization(o, Visualization.pcd_mat()) for o in map_a.to_o3d(Hierarchy.ROOM)[2]] +

        # Topometric map B visualization at room level
        [MapVisualization(o, Visualization.pcd_mat(pt_size=6)) for o in map_b.to_o3d(Hierarchy.ROOM)[0]] +
        [MapVisualization(o, Visualization.pcd_mat()) for o in map_b.to_o3d(Hierarchy.ROOM)[2]
         ],
    ])


def visualize_matches(map_a, map_b, matches):
    from model.topometric_map import Hierarchy

    rooms_a = map_a.get_node_level(Hierarchy.ROOM)
    rooms_b = map_b.get_node_level(Hierarchy.ROOM)

    print("Visualizing matches")
    line_set = o3d.geometry.LineSet()
    points, lines = [], []

    for i, n in enumerate(matches):
        room_a, room_b = n
        m_a = rooms_a[room_a]
        m_b = rooms_b[room_b]

        points.append(m_a.geometry.centroid())
        points.append(m_b.geometry.centroid())
        lines.append((i*2, i*2 + 1))

    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    viz = Visualization([
        # Topometric map A visualization at room level
        [MapVisualization(o, Visualization.pcd_mat(pt_size=6)) for o in map_a.to_o3d(Hierarchy.ROOM)[0]] +
        # [MapViz(map_a.to_o3d(Hierarchy.ROOM)[1], Viz.graph_mat())] +
        [MapVisualization(o, Visualization.pcd_mat()) for o in map_a.to_o3d(Hierarchy.ROOM)[2]] +

        [MapVisualization(line_set, Visualization.graph_mat(color=[0, 0, 1, 1]))] +

        # Topometric map B visualization at room level
        [MapVisualization(o, Visualization.pcd_mat(pt_size=6)) for o in map_b.to_o3d(Hierarchy.ROOM)[0]] +
        # [MapViz(map_b.to_o3d(Hierarchy.ROOM)[1], Viz.graph_mat())] +
        [MapVisualization(o, Visualization.pcd_mat()) for o in map_b.to_o3d(Hierarchy.ROOM)[2]
         ],
    ])


if __name__ == '__main__':
    from model.topometric_map import TopometricMap

    map = load_pickle(select_file())
    map_type = type(map)

    if map_type == TopometricMap:
        visualize_htmap(map)
    elif map_type == PointCloud:
        visualize_point_cloud(map)
