import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np


# 0. Implementation of a Dataset
class MultiCutStixelData(Dataset):
    # 1. Implement __init()__
    def __init__(self, annotations_file, data_dir="data", transform=None, target_transform=None):
        self.data_dir = data_dir
        self.annotations = pd.read_csv(os.path.join(data_dir, annotations_file))
        self.img_map = self.__create_image_reference_map()
        self.transform = transform
        self.target_transform = target_transform

    # 2. Implement __len()__
    def __len__(self):
        return len(self.img_map)

    # 3. Implement __getitem()__
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_map[idx])
        image = Image.open(img_path)
        label = self.annotations.groupby('img_path').get_group(self.img_map[idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # data type needs to be like the NN layer like .to(torch.float32)
        return image, label

    def __create_image_reference_map(self):
        # Separate the path col
        image_map = self.annotations.loc[:, 'img_path'].tolist()
        # drop all duplicates and add a label array
        image_map = list(dict.fromkeys(image_map))
        return image_map


def transforming(x_features):
    return ToTensor()(x_features).unsqueeze_(0)


# TODO: create a tensor of 160 x 240
def target_transforming(y_target, grid_step=8):
    coordinates = y_target.loc[:, ['x', 'y', 'class']].to_numpy()
    # adapt to the label_size
    coordinates = coordinates/(grid_step, grid_step, 1)
    mtx = np.zeros((160, 240))
    for point in coordinates.astype(int):
        mtx[point[1]][point[0]] = point[2]
    y_target_label = torch.from_numpy(mtx)
    return y_target_label
