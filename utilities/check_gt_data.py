import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import io

from utilities.visualization import __draw_stixel_on_image
from dataloader.waymo_multicut import MultiCutStixelData, transforming, target_transforming


def main():
    checking_data_path = "training_data.csv"
    checking_data = MultiCutStixelData(checking_data_path, data_dir="data/training",
                                       transform=transforming,
                                       target_transform=target_transforming)
    checking_dataloader = DataLoader(checking_data, batch_size=1, num_workers=4, pin_memory=True, shuffle=True)

    test_features, test_labels = next(iter(checking_dataloader))
    print(test_features.shape)
    target_mtx = test_labels[-1].numpy()
    check_image = __draw_stixel_on_image(test_features[-1], target_mtx, save_fig=True)
    plt.figure(figsize=(20, 12))
    plt.imshow(check_image)
    plt.show()


if __name__ == '__main__':
    main()
