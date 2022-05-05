from typing import Dict, List
import glob

from model.point_cloud import PointCloud
from utils.io import *
from utils.visualization import visualize_point_cloud
from pickle import dump


def load_stanford(dir: str) -> Dict[str, PointCloud]:
    print(f'Loading Stanford indoor 3D dataset from {dir}')

    # Only the point clouds in the dataset and their associated room
    data: Dict[str, PointCloud] = {}

    # root_dir needs a trailing slash (i.e. /root/dir/)
    txt_files = glob.iglob(dir + "/" + '**/**.txt', recursive=True)

    # skip annotation files as only room-level semantics are used
    xyz_files = filter(
        lambda x: 'Annotations' not in x and 'alignmentAngle' not in x, txt_files)

    for path in xyz_files:
        print(f'Reading {path}')

        try:
            point_cloud = PointCloud.read_xyz(path, ' ')
        except ValueError as e:
            print(f'Failed to parse .xyz file {path}: {e}')
            continue

        data[path] = point_cloud

    return data


def merge_dataset_subset(data: Dict[str, PointCloud], target_rooms: List[str]):
    merged_point_cloud = PointCloud()

    for room, room_point_cloud in data.items():
        if room in target_rooms:
            merged_point_cloud = merged_point_cloud.merge(room_point_cloud)

    return merged_point_cloud


if __name__ == "__main__":
    from utils.io import write_pickle

    folder_selected = select_directory_dialog()
    stanford_dataset = load_stanford(folder_selected)

    stanford_dataset_subsets = stanford_dataset.keys()
    merged_subset = merge_dataset_subset(stanford_dataset, stanford_dataset_subsets)

    visualize_point_cloud(merged_subset)
    write_pickle(save_file_dialog(), merged_subset)
