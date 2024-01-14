import numpy as np
from typing import List
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import yaml
# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

class Stixel:
    def __init__(self, x, y_t, y_b, depth=42.0):
        self.column = x
        self.top = y_t
        self.bottom = y_b
        self.seg_class = -1
        self.depth = depth
        self.scale_by_grid()

    def __repr__(self):
        return f"{self.column},{self.top},{self.bottom},{self.depth}"

    def scale_by_grid(self, grid_step=config['grid_step']):
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

def draw_heatmap(image, prediction, stixel_width=config['grid_step']):
    # Wir werden die Matrixwerte benutzen, um eine Heatmap zu erstellen
    # Zuerst skalieren wir die Matrix auf die Dimension des Bildes
    heatmap = cv2.resize(prediction, (image.shape[1] // stixel_width, image.shape[0] // stixel_width))

    # Jetzt erweitern wir die Heatmap auf die vollständige Größe des Bildes, um es zu überlagern
    heatmap_large = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Anwendung eines Farbschemas auf die Heatmap
    heatmap_large = np.uint8(255 * heatmap_large)  # Umwandlung zu einem Bildformat
    color_heatmap = cv2.applyColorMap(heatmap_large, cv2.COLORMAP_JET)

    # Überlagern der Heatmap auf das Originalbild
    overlay_image = cv2.addWeighted(image, 0.7, color_heatmap, 0.3, 0)

    # Visualisierung des Ergebnisses
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))  # Umwandlung in RGB für die Anzeige
    plt.axis('off')  # Keine Achsen für die Anzeige
    plt.show()


class StixelNExTInterpreter:
    def __init__(self, detection_threshold=0.4, hysteresis_threshold=False):
        self.s1 = detection_threshold
        if hysteresis_threshold:
            self.s2 = hysteresis_threshold
        else:
            self.s2 = detection_threshold - 0.05
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
