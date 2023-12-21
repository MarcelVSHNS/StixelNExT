import numpy as np
from typing import List
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class Stixel:
    def __init__(self, x, y_t, y_b, depth=8.0):
        self.column = x
        self.top = y_t
        self.bottom = y_b
        self.depth = depth
        self.scale_by_grid()

    def __repr__(self):
        return f"{self.column},{self.top},{self.bottom},{self.depth}"

    def scale_by_grid(self, grid_step=8):
        self.column = self.column * grid_step
        self.top = self.top * grid_step
        self.bottom = self.bottom * grid_step
        if self.bottom > 1200 or self.top > 1200 or self.column > 1920:
            print("nooo")


def extract_stixels(prediction, s1, s2):
    """

    Extract Stixels

    This method takes a prediction matrix, `prediction`, along with two threshold values, `s1` and `s2`. It calculates and returns a list of stixels.

    Parameters:
    - `prediction` (list of numpy arrays): A list containing two numpy arrays representing the prediction matrix. The shape of each numpy array should be (num_rows, num_cols).
    - `s1` (int): The threshold value for determining if a stixel starts.
    - `s2` (int): The threshold value for determining if a stixel ends.

    Returns:
    - `stixels` (list): A list of Stixel objects. Each Stixel object represents a stixel and contains the following attributes:
        - `col` (int): The column index where the stixel starts or ends.
        - `start` (int): The row index where the stixel starts.
        - `end` (int): The row index where the stixel ends.

    Note:
    - A stixel is defined as a continuous region in the prediction matrix where the value in the `stixel_repr` array is greater than or equal to `s1` and the value in the `bottom_repr` array
    * is less than `s2`.
    - The `Stixel` object is not provided and should be defined separately before using this method.

    Example Usage:
    ```
    prediction = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 0, 1], [0, 1, 0]])]
    s1 = 3
    s2 = 1
    stixels = extract_stixels(prediction, s1, s2)
    ```
    """
    num_rows, num_cols = prediction[0].shape
    stixel_repr = prediction[0]
    bottom_repr = prediction[1]
    stixels = []
    for col in range(num_cols):
        in_stixel = False
        stixel_start = 0
        for row in range(num_rows):
            if in_stixel:
                if stixel_repr[row][col] < s2 or bottom_repr[row][col]  >= s1:
                    stixels.append(Stixel(col, stixel_start, row))
                    in_stixel = False
            else:
                if stixel_repr[row][col] >= s1:
                    in_stixel = True
                    stixel_start = row
        if in_stixel:
            stixels.append(Stixel(col, stixel_start, num_rows - 1))
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
        top_left_x, top_left_y = stixel.column, stixel.top
        bottom_left_x, bottom_left_y = stixel.column, stixel.bottom
        color = get_color_from_depth(stixel.depth, min_depth, max_depth)
        bottom_right_x = bottom_left_x + stixel_width
        overlay = image.copy()
        cv2.rectangle(overlay, (top_left_x, top_left_y), (bottom_right_x, bottom_left_y), color, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_left_y), color, 2)
    return Image.fromarray(image)


def draw_bottom_lines(image, bottom_pts: np.array, threshold, grid_step=8, alpha=0.1):
    image = np.array(image.numpy())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i in range(bottom_pts.shape[0]):
        for j in range(bottom_pts.shape[1]):
            if bottom_pts[i, j] > threshold:
                center_coordinates = (j * grid_step, i * grid_step)
                radius = 1 # define radius of circle here
                color = (255, 0, 0)  # color of the circle
                thickness = -1 # line thickness, use -1 for filled circle
                image = cv2.circle(image, center_coordinates, radius, color, thickness)
    return Image.fromarray(image)


class StixelNExTInterpreter:
    def __init__(self, detection_threshold=0.4, hysteresis_threshold=0.35):
        self.s1 = detection_threshold
        self.s2 = hysteresis_threshold
        self.stixel_list = None
        self.bottom_pts = None

    def extract_stixel_from_prediction(self, prediction, detection_threshold=None, hysteresis_threshold=None):
        prediction_mtx_numpy: np.array = prediction.numpy()
        s1 = detection_threshold if detection_threshold else self.s1
        s2 = hysteresis_threshold if hysteresis_threshold else self.s2
        self.stixel_list = extract_stixels(prediction_mtx_numpy, s1=s1, s2=s2)
        self.bottom_pts = prediction_mtx_numpy[1]
        return self.stixel_list

    def show_stixel(self, pil_image):
        image_with_stixel = draw_stixels_on_image(pil_image, self.stixel_list)
        image_with_stixel.show()

    def show_bottoms(self, image):
        image_with_bottom_pts = draw_bottom_lines(image, self.bottom_pts, self.s1)
        image_with_bottom_pts.show()
