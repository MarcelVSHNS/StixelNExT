import pandas
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import numpy as np
from typing import List, Tuple
from collections import defaultdict
import cv2
from utilities.visualization import draw_stixels_on_image
from dataloader.stixel_multicut_interpreter import Stixel

def _group_points_by_x(points_array: np.array) -> List[List[np.array]]:
    """ Groups points by their x value.
    Parameters:
        points_array (np.array): A NumPy array of points, where each point is represented as [x, y, class].
    Returns:
        list: A list of lists, where each inner list contains points with the same x value.
    """
    # Using defaultdict to group points by x value
    grouped_points = defaultdict(list)
    # Iterate over the points and group them by x value
    for point in points_array:
        x, y_t, y_b, cls = point
        grouped_points[x].append(point)
    # Convert the grouped points into a list of lists
    grouped_sorted_lists = [sorted(group, key=lambda p: p[1], reverse=True) for group in grouped_points.values()]
    return grouped_sorted_lists


def _fill_mtx_with_points(obj_mtx, points):
    for x, y_t, y_b, cls in np.array(points).astype(int):
        if cls == 1:
            obj_mtx[y_t:y_b + 1, x] = 1


# 0. Implementation of a Dataset
class MultiCutStixelData(Dataset):
    # 1. Implement __init()__
    def __init__(self, data_dir, phase, annotation_dir="targets_from_lidar", img_dir="STEREO_LEFT", transform=None, target_transform=None):
        self.data_dir: str = os.path.join(data_dir, phase)
        self.img_path: str = os.path.join(self.data_dir, img_dir)
        self.annotation_path: str= os.path.join(self.data_dir, annotation_dir)
        filenames: List[str] = os.listdir(os.path.join(self.data_dir, img_dir))
        self.sample_map: List[str] = [os.path.splitext(filename)[0] for filename in filenames]
        self.transform = transform
        self.target_transform = target_transform
        self.name: str = os.path.basename(data_dir)
        self.img_size = self._determine_image_size()

    # 2. Implement __len()__
    def __len__(self) -> int:
        return len(self.sample_map)

    # 3. Implement __getitem()__
    def __getitem__(self, idx) -> Tuple[torch.Tensor,pd.DataFrame]:
        img_path_full: str = os.path.join(self.img_path, self.sample_map[idx] + ".png")
        feature_image: torch.Tensor = read_image(img_path_full, ImageReadMode.RGB).to(torch.float32)
        target_labels: pd.DataFrame = pd.read_csv(os.path.join(self.annotation_path, os.path.basename(self.sample_map[idx]) + ".csv"))
        target_labels = self._preparation_of_target_label(target_labels)
        if self.transform:
            feature_image = self.transform(feature_image)
        if self.target_transform:
            target_labels = self.target_transform(target_labels)
        # data type needs to be like the NN layer like .to(torch.float32)
        return feature_image, target_labels

    def _determine_image_size(self):
        test_img_path = os.path.join(self.img_path, self.sample_map[0] + ".png")
        test_feature_image = read_image(test_img_path, ImageReadMode.RGB)
        channels, height, width = test_feature_image.shape
        return {'height': height, 'width': width, 'channels': channels}

    def _preparation_of_target_label(self, y_target: pandas.DataFrame, grid_step: int = 8) -> torch.tensor:
        coordinates = y_target.loc[:, ['x', 'yT', 'yB', 'class']].to_numpy()
        coordinates = coordinates / (grid_step, grid_step, grid_step, 1)
        assert self.img_size['height'] % grid_step == 0
        assert self.img_size['width'] % grid_step == 0
        mtx_width, mtx_height = int(self.img_size['width'] / grid_step), int(self.img_size['height'] / grid_step)
        cut_mtx = np.zeros((mtx_height, mtx_width))
        bottom_mtx = np.zeros((mtx_height, mtx_width))
        grouped_coordinates_by_col = _group_points_by_x(coordinates)
        for col_points in grouped_coordinates_by_col:
            for x, y_t, y_b, cls in np.array(col_points).astype(int):
                if cls == 1:
                    assert x < mtx_width, f"x-value out of bound ({x},{y_t})."
                    assert y_t < mtx_height, f"y-value top out of bound ({x},{y_t})."
                    assert y_b < mtx_height, f"y-value bottom out of bound ({x},{y_t})."
                    cut_mtx[y_t][x] = 1  # for every top point add a 1 to indicate a cut
                    #cut_mtx[y_b][x] = 1  # for every bottom point add a 1 to indicate a cut
                    #if cls == 1:
                    bottom_mtx[y_b][x] = 1  # for every top point add a 1 to cuts
            # for every bottom_pt to top_pt add ones
            #_fill_mtx_with_points(obj_mtx, col_points)
        # observe sequence: cut, obj, top
        stacked_mtx = np.stack([cut_mtx, bottom_mtx])
        label = torch.from_numpy(stacked_mtx).to(torch.float32)
        return label

    def check_target(self, tensor_image, target_labels):
        coordinates = target_labels.loc[:, ['x', 'yT', 'yB', 'class']].to_numpy()
        stixels = []
        for x, y_t, y_b, cls in coordinates.astype(int):
            stixels.append(Stixel(x, y_t, y_b, cls))
        print(f"len: {len(stixels)}")
        img = draw_stixels_on_image(tensor_image, stixels)
        #img.show()


def feature_transforming(x_features: torch.Tensor) -> torch.Tensor:
    pass


def overlay_original(matrix, original):
    # calculate col-wise to equalize dense points
    max_vals = np.max(matrix, axis=0)
    normalized_matrix = np.divide(matrix, max_vals, out=np.zeros_like(matrix), where=max_vals!=0)
    return np.maximum(normalized_matrix, original)


def target_transform_gaussian_blur(y_target: torch.Tensor) -> torch.Tensor:
    y_target_numpy: np.array = y_target.numpy()
    # possible option is to blur 3-dimensional (like an RGB color img): cut, bottom
    cut_mtx = y_target_numpy[0, :, :]
    bottom_mtx = y_target_numpy[1, :, :]
    # use (1,5) as kernel_size to refer to columns, or softer with (3,5)
    # be aware: close points lead to higher values, lonely points have lower prob.
    # blur with cv2 filter
    cut_mtx_blur = cv2.GaussianBlur(cut_mtx, (3, 7), sigmaX=0.5, sigmaY=1.0)
    bottom_mtx_blur = cv2.GaussianBlur(bottom_mtx, (7, 3), sigmaX=1.0, sigmaY=0.5)
    #bottom_mtx_blur = cv2.GaussianBlur(bottom_mtx, kernel_size, sigmaX=sigma_x, sigmaY=sigma_y)
    # apply targets on blur
    cut_mtx = overlay_original(cut_mtx_blur, cut_mtx)
    bottom_mtx = overlay_original(bottom_mtx_blur, bottom_mtx)
    # convert back to torch.tensor
    stacked_mtx = np.stack([cut_mtx, bottom_mtx])
    label = torch.from_numpy(stacked_mtx).to(torch.float32)
    return label
