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
    def __init__(self, x, y_t, y_b, depth=42.0, grid_step=config['grid_step']):
        self.column = int(x)
        self.top = int(y_t)
        self.bottom = int(y_b)
        self.seg_class = -1
        self.depth = depth
        self.grid_step = grid_step
        # self.scale_by_grid()

    def __repr__(self):
        return f"{self.column},{self.top},{self.bottom},{self.depth}"

    def scale_by_grid(self):
        self.column = int(self.column * self.grid_step)
        self.top = int(self.top * self.grid_step)
        self.bottom = int(self.bottom * self.grid_step)
        if self.bottom > 1200 or self.top > 1200 or self.column > 1920:
            print("nooo")

    def cut_is_in_stixel(self, cut_row, tolerance=120):
        cut_row = cut_row * self.grid_step
        if self.top + tolerance < cut_row < self.bottom - tolerance:
            return True
        else:
            return False

    def divide_stixel_into_two(self, cut_row):
        col = self.column / self.grid_step
        y_t = self.top / self.grid_step
        y_b = self.bottom / self.grid_step
        upper_stixel = Stixel(col, y_t, cut_row)
        lower_stixel = Stixel(col, cut_row + 1, y_b)
        return upper_stixel, lower_stixel


def extract_stixels_classic(prediction, s1, s2=0.1):
    num_rows, num_cols = prediction[0].shape
    occupancy = prediction[0]
    cut_mtx = prediction[1]

    stixels = []
    for col in range(num_cols):
        col_stixels = []
        col_cuts = []
        in_stixel = False
        stixel_start = 0
        for row in range(num_rows):
            if in_stixel:
                if occupancy[row][col] < s1 - s2:
                    col_stixels.append(Stixel(col, stixel_start, row))
                    in_stixel = False
            else:
                if occupancy[row][col] >= s1:
                    in_stixel = True
                    stixel_start = row
        if in_stixel:
            col_stixels.append(Stixel(col, stixel_start, num_rows - 1))
        # find cuts
        in_cut = False
        offset = 0.3
        for row in range(num_rows):
            if in_cut:
                if cut_mtx[row][col] < s1 - offset - s2:
                    cut_end = row
                    col_cuts.append((cut_start + cut_end) / 2)
                    in_cut = False
            else:
                if cut_mtx[row][col] >= s1 - offset:
                    in_cut = True
                    cut_start = row
        for cut in col_cuts:
            for idx in range(len(col_stixels)):
                if col_stixels[idx].cut_is_in_stixel(cut):
                    upper, lower = col_stixels[idx].divide_stixel_into_two(cut)
                    col_stixels.pop(idx)
                    col_stixels.append(upper)
                    col_stixels.append(lower)
                    break
        for stixel in col_stixels:
            stixels.append(stixel)
    return stixels


def extract_stixels(prediction, threshold, with_depth=False) -> List[Stixel]:
    stixels = []
    height = config['img_height']
    grid_step = config['grid_step']
    for col in range(prediction.shape[2]):
        objects = prediction[:, :, col]
        for max_objects in range(objects.shape[1]):
            stixel = objects[:, max_objects]
            if stixel[2] >= threshold:
                stixels.append(Stixel(col * grid_step, stixel[0] * height, stixel[1] * height))
    return stixels


def get_color_from_depth(depth, min_depth, max_depth):
    normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    color = plt.cm.RdYlGn(normalized_depth)[:3]
    return tuple(int(c * 255) for c in color)


def draw_stixels_on_image(image, stixels: List[Stixel], color=[255, 0, 0], stixel_width=8, alpha=0.3):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    stixels.sort(key=lambda x: x.depth, reverse=True)
    min_depth, max_depth = 0, 50
    for stixel in stixels:
        top_left_x, top_left_y = stixel.column, stixel.top
        bottom_left_x, bottom_left_y = stixel.column, stixel.bottom
        color = color
        #get_color_from_depth(stixel.depth, min_depth, max_depth)
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


def draw_heatmap(image, prediction, stixel_width=config['grid_step'], map=0):
    image = np.array(image)
    prediction = prediction[map].numpy()
    heatmap = cv2.resize(prediction, (image.shape[1] // stixel_width, image.shape[0] // stixel_width))
    heatmap_large = cv2.resize(heatmap, (int(image.shape[1]-stixel_width/2), int(image.shape[0]-stixel_width/2)))
    heatmap_centered = np.zeros((image.shape[0], image.shape[1]))
    offset = int(stixel_width / 2)
    heatmap_centered[offset:offset + heatmap_large.shape[0], offset:offset + heatmap_large.shape[1]] = heatmap_large
    heatmap = np.uint8(255 * heatmap_centered)
    color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay_image = cv2.addWeighted(image, 0.7, color_heatmap, 0.3, 0)
    overlay_image_pil = Image.fromarray(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    return overlay_image_pil


def show_pred_heatmap(pil_image, output, map=0):
    image_with_heatmap = draw_heatmap(pil_image, output, map=map)
    image_with_heatmap.show()


class StixelNExTInterpreter:
    def __init__(self, detection_threshold=0.4):
        self.detection_threshold = detection_threshold
        self.stixel_list = None
        self.bottom_pts = None

    def extract_stixel_from_prediction(self, prediction, detection_threshold=None):
        prediction_mtx_numpy: np.array = prediction.numpy()
        if detection_threshold is None:
            threshold = self.detection_threshold
        else:
            threshold = detection_threshold
        self.stixel_list = extract_stixels(prediction_mtx_numpy, threshold=threshold)
        return self.stixel_list

    def show_stixel(self, pil_image, stixel_list=None, color=[255, 0, 0]):
        if stixel_list is None:
            stixels = self.stixel_list
        else:
            stixels = stixel_list
        image_with_stixel = draw_stixels_on_image(pil_image, stixels, color=color)
        image_with_stixel.show()

    def show_bottoms(self, image):
        image_with_bottom_pts = draw_bottom_lines(image, self.bottom_pts, self.s1)
        image_with_bottom_pts.show()