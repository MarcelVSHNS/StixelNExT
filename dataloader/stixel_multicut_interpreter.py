import numpy as np
from typing import List
import matplotlib.pyplot as plt
import cv2
from PIL import Image


class Stixel:
    def __init__(self, x, y_t, y_b, depth=None):
        self.column = x
        self.top_row = y_t
        self.bottom_row = y_b
        self.depth = depth
        self.scale_by_grid()

    def __repr__(self):
        return f"{self.column},{self.top_row},{self.bottom_row},{self.depth}"

    def scale_by_grid(self, grid_step=8):
        self.column = self.column * grid_step
        self.top_row = self.top_row * grid_step
        self.bottom_row = self.bottom_row * grid_step
        if self.bottom_row > 1200 or self.top_row > 1200 or self.column > 1920:
            print("nooo")


def extract_stixels(prediction, s1, s2):
    num_rows, num_cols = prediction[0].shape
    xy_repr = prediction[0]
    xz_repr = prediction[1]
    zy_repr = prediction[2]
    stixels = []
    for col in range(num_cols):
        in_stixel = False
        stixel_start = 0
        depth = 0
        for row in range(num_rows):
            if in_stixel:
                if xy_repr[row][col] < s2:
                    stixels.append(Stixel(col, stixel_start, row, depth))
                    in_stixel = False
            else:
                if xy_repr[row][col] >= s1:
                    z_row = np.argmax(zy_repr[row, :])
                    print(np.argmax(zy_repr[99, :]))
                    z_xz = xz_repr[z_row, col]
                    depth = z_row / 240 * 50
                    in_stixel = True
                    stixel_start = row
        if in_stixel:
            stixels.append(Stixel(col, stixel_start, num_rows - 1, depth))
    return stixels


def get_color_from_depth(depth, min_depth, max_depth):
    normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    color = plt.cm.RdYlGn(normalized_depth)[:3]
    return tuple(int(c * 255) for c in color)


def draw_stixels_on_image(image, stixels: List[Stixel], stixel_width=8, alpha=0.1):
    image = np.array(image.numpy())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    stixels.sort(key=lambda x: x.depth, reverse=True)
    min_depth, max_depth = 0, 50
    for stixel in stixels:
        top_left_x, top_left_y = stixel.column, stixel.top_row
        bottom_left_x, bottom_left_y = stixel.column, stixel.bottom_row
        color = get_color_from_depth(stixel.depth, min_depth, max_depth)
        bottom_right_x = bottom_left_x + stixel_width
        overlay = image.copy()
        cv2.rectangle(overlay, (top_left_x, top_left_y), (bottom_right_x, bottom_left_y), color, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_left_y), color, 2)
    return Image.fromarray(image)


class StixelNExTInterpreter:
    def __init__(self, detection_threshold=0.4, hysteresis_threshold=0.35):
        self.s1 = detection_threshold
        self.s2 = hysteresis_threshold
        self.stixel_list = None

    def extract_stixel_from_prediction(self, prediction):
        prediction_mtx_numpy: np.array = prediction.numpy()
        self.stixel_list = extract_stixels(prediction_mtx_numpy, s1=self.s1, s2=self.s2)

    def show_stixel(self, pil_image):
        image_with_stixel = draw_stixels_on_image(pil_image, self.stixel_list)
        image_with_stixel.show()



