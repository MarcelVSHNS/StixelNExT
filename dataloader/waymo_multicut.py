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
    def __init__(self, annotations_file, data_dir, annotation_dir="targets", transform=None, target_transform=None):
        self.data_dir = data_dir
        self.annotation_dir = os.path.join(data_dir, annotation_dir)
        self.sample_map = pd.read_csv(os.path.join(data_dir.split('/')[0], annotations_file), header=None).values.reshape(-1,).tolist()
        self.transform = transform
        self.target_transform = target_transform

    # 2. Implement __len()__
    def __len__(self):
        return len(self.sample_map)

    # 3. Implement __getitem()__
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.sample_map[idx] + ".png")
        image = read_image(img_path, ImageReadMode.RGB).to(torch.float32)
        label = pd.read_csv(os.path.join(self.annotation_dir, os.path.basename(self.sample_map[idx]) + ".csv"))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # data type needs to be like the NN layer like .to(torch.float32)
        return image, label


def transforming(x_features):
    return ToTensor()(x_features).unsqueeze_(0)


# TODO: create a tensor of 160 x 240
def target_transforming(y_target, grid_step=8):
    coordinates = y_target.loc[:, ['x', 'y', 'class']].to_numpy()
    # adapt to the label_size
    coordinates = coordinates/(grid_step, grid_step, 1)
    mtx = np.zeros((160, 240))
    for point in coordinates.astype(int):
        mtx[point[1]][point[0]] = 1
    y_target_label = torch.from_numpy(mtx)
    return y_target_label
