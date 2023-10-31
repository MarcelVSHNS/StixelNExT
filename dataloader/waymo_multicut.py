import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.io import read_image, ImageReadMode
import numpy as np
import datetime


# 0. Implementation of a Dataset
class MultiCutStixelData(Dataset):
    # 1. Implement __init()__
    def __init__(self, data_dir, annotation_dir="targets", img_dir="", transform=None, target_transform=None):
        self.data_dir = data_dir
        self.img_path = os.path.join(self.data_dir, img_dir)
        self.annotation_path = os.path.join(self.data_dir, annotation_dir)
        filenames = os.listdir(os.path.join(data_dir, img_dir))
        self.sample_map = [os.path.splitext(filename)[0] for filename in filenames]
        self.transform = transform
        self.target_transform = target_transform

    # 2. Implement __len()__
    def __len__(self):
        return len(self.sample_map)

    # 3. Implement __getitem()__
    def __getitem__(self, idx):
        img_path_full = os.path.join(self.img_path, self.sample_map[idx] + ".png")
        image = read_image(img_path_full, ImageReadMode.RGB)
        label = pd.read_csv(os.path.join(self.annotation_path, os.path.basename(self.sample_map[idx]) + ".csv"))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # data type needs to be like the NN layer like .to(torch.float32)
        return image, label


def transforming(x_features):
    return x_features.to(torch.float32)


# TODO: create a tensor of 160 x 240
def target_transforming(y_target, grid_step=8):
    coordinates = y_target.loc[:, ['x', 'y', 'class']].to_numpy()
    # adapt to the label_size
    coordinates = coordinates/(grid_step, grid_step, 1)
    mtx = np.zeros((150, 240))
    for point in coordinates.astype(int):
        mtx[point[1]][point[0]] = 1       # point[2]
    y_target_label = torch.from_numpy(mtx).to(torch.float32)
    label = y_target_label.squeeze()
    return label
