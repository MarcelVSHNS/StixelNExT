import pandas
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import numpy as np
from PIL import Image
from typing import List, Tuple
from collections import defaultdict
import cv2
from utilities.visualization import draw_stixels_on_image
from dataloader.stixel_multicut_interpreter import Stixel
import torch.nn.functional as F

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
    def __init__(self, data_dir, phase, annotation_dir="targets_from_lidar", img_dir="STEREO_LEFT", transform=None, target_transform=None, return_original_image=False):
        self.data_dir: str = os.path.join(data_dir, phase)
        self.img_path: str = os.path.join(self.data_dir, img_dir)
        self.annotation_path: str= os.path.join(self.data_dir, annotation_dir)
        filenames: List[str] = os.listdir(os.path.join(self.data_dir, img_dir))
        self.sample_map: List[str] = [os.path.splitext(filename)[0] for filename in filenames]
        self.return_original_image = return_original_image
        self.transform = transform
        self.target_transform = target_transform
        self.name: str = os.path.basename(data_dir)
        self.img_size = self._determine_image_size()

    # 2. Implement __len()__
    def __len__(self) -> int:
        return len(self.sample_map)

    # 3. Implement __getitem()__
    def __getitem__(self, idx):
        img_path_full: str = os.path.join(self.img_path, self.sample_map[idx] + ".png")
        feature_image: torch.Tensor = read_image(img_path_full, ImageReadMode.RGB).to(torch.float32)
        target_labels: pd.DataFrame = pd.read_csv(os.path.join(self.annotation_path, os.path.basename(self.sample_map[idx]) + ".csv"))
        target_labels = self._preparation_of_target_label(target_labels)
        if self.transform:
            feature_image = self.transform(feature_image)
        if self.target_transform:
            target_labels = self.target_transform(target_labels)
        # data type needs to be like the NN layer like .to(torch.float32)
        if self.return_original_image:
            return feature_image, target_labels, cv2.imread(img_path_full)
        else:
            return feature_image, target_labels

    def _determine_image_size(self):
        test_img_path = os.path.join(self.img_path, self.sample_map[0] + ".png")
        test_feature_image = read_image(test_img_path, ImageReadMode.RGB)
        channels, height, width = test_feature_image.shape
        return {'height': height, 'width': width, 'channels': channels}

    def _preparation_of_target_label(self, y_target: pandas.DataFrame, grid_step: int = 8, max_depth: int = 50) -> torch.tensor:
        y_target['x'] = (y_target['x'] // grid_step).astype(int)
        y_target['yT'] = (y_target['yT'] // grid_step).astype(int)
        y_target['yB'] = (y_target['yB'] // grid_step)
        y_target['depth'] = y_target['depth'].clip(upper=max_depth)
        y_target['depth_x'] = (y_target['depth'] / max_depth * 149).astype(int)
        y_target['depth_y'] = (y_target['depth'] / max_depth * 239).astype(int)
        matrix = np.zeros((3, 150, 240))
        for index, row in y_target.iterrows():
            for y in range(row['yT'], row['yB'] + 1):  # Markieren aller Punkte von yT bis yB in der y-Achse
                matrix[0, int(y), int(row['x'])] = 1

            matrix[1][int(row['depth_x']), int(row['x'])] = 1

            for y in range(row['yT'], row['yB'] + 1):  # Markieren aller Punkte von yT bis yB in der y-Achse
                matrix[2, int(y), int(row['depth_y'])] = 1
        label = torch.from_numpy(matrix).to(torch.float32)
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
    yx_mtx = y_target_numpy[0, :, :]
    zx_mtx = y_target_numpy[1, :, :]
    yz_mtx = y_target_numpy[2, :, :]
    # use (1,5) as kernel_size to refer to columns, or softer with (3,5)
    # be aware: close points lead to higher values, lonely points have lower prob.
    # blur with cv2 filter
    xy_mtx_blur = cv2.GaussianBlur(yx_mtx, (3, 7), sigmaX=1.8, sigmaY=1.2)
    xz_mtx_blur = cv2.GaussianBlur(zx_mtx, (3, 11), sigmaX=1.0, sigmaY=2.0)
    yz_mtx_blur = cv2.GaussianBlur(yz_mtx, (3, 9), sigmaX=1.0, sigmaY=1.5)
    #bottom_mtx_blur = cv2.GaussianBlur(bottom_mtx, kernel_size, sigmaX=sigma_x, sigmaY=sigma_y)
    # apply targets on blur
    yx_mtx = overlay_original(xy_mtx_blur, yx_mtx)
    zx_mtx = overlay_original(xz_mtx_blur, zx_mtx)
    yz_mtx = overlay_original(yz_mtx_blur, yz_mtx)
    # convert back to torch.tensor
    stacked_mtx = np.stack([yx_mtx, zx_mtx, yz_mtx])
    label = torch.from_numpy(stacked_mtx).to(torch.float32)
    return label
