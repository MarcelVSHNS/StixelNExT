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
        y_target['x'] = (y_target['x'] // grid_step).astype(int)
        y_target['yT'] = (y_target['yT'] // grid_step).astype(int)
        y_target['yB'] = (y_target['yB'] // grid_step)
        y_target['depth'] = ((y_target['depth'] * 0.5).round(0).clip(upper=44.5) * 2).astype(int)
        matrix = np.zeros((90, 150, 240))
        for index, row in y_target.iterrows():
            depth_z = int(row['depth'])
            row_x = int(row['x'])
            for y in range(row['yT'], row['yB'] + 1):  # Markieren aller Punkte von yT bis yB in der y-Achse
                matrix[depth_z, y, row_x] = 1
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


def target_transform_gaussian_blur(y_target: torch.Tensor, kernel_size=3, sigma=4.0, iterations=1) -> torch.Tensor:
    def gaussian_kernel(size, sigma):
        # create gaussian kernel
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2

        g = coords ** 2
        g = (-g / (2 * sigma ** 2)).exp()

        g /= g.sum()
        return g.view(1, -1) * g.view(-1, 1)

    def gaussian_kernel_3d(kernel_size, sigma):
        gk = gaussian_kernel(kernel_size, sigma)
        gk3d = torch.zeros((kernel_size, kernel_size, kernel_size))
        for i in range(kernel_size):
            gk3d[i] = gk * gk[i]
        return gk3d

    gauss_kernel = gaussian_kernel_3d(kernel_size, sigma)
    gauss_kernel = gauss_kernel.expand(1, 1, *gauss_kernel.shape)  # Anpassen der Form für conv3d

    # Konvertieren der Matrix in einen Tensor und Anwenden des Gauß'schen Weichzeichners
    label_tensor: torch.Tensor = y_target.unsqueeze(0).unsqueeze(0)
    for _ in range(iterations):
        blurred_label = F.conv3d(label_tensor, gauss_kernel, padding=kernel_size // 2)
    blurred_label: torch.Tensor = blurred_label.squeeze(0).squeeze(0)
    amplified_label = blurred_label * 10
    amplified_label[y_target == 1] = 1
    #test = amplified_label.numpy()
    #max = np.amax(test)
    return torch.clamp(amplified_label, max=1)
