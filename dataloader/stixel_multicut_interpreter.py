import numpy as np
from typing import List
import torch


class Stixel:
    def __init__(self, x, y_t, y_b, cls=None):
        self.x = x
        self.y_t = y_t
        self.y_b = y_b
        self.y_t_obj = []
        self.cls = cls

    def __repr__(self):
        return f"{self.x},{self.y_t},{self.y_b},{self.cls}"

    def scale_by_grid(self, grid_step=8):
        self.x = self.x * grid_step
        self.y_t = self.y_t * grid_step
        self.y_b = self.y_b * grid_step
        if self.y_b > 1200 or self.y_t > 1200 or self.x >1920:
            print("nooo")


def extract_stixels(cut_mtx, bottom_mtx):
    stixels = []
    rows, cols = cut_mtx.shape
    for x in range(cols):
        stixel_started = False
        for y in range(rows):
            if bottom_mtx[y, x] == 1 and stixel_started:
                # End of the current stixel and start of a new one
                current_stixel.y_b = y
                stixels.append(current_stixel)
                stixel_started = False
                current_stixel = None
            elif bottom_mtx[y, x] == 1:
                print(f"Missmatch at ({x},{y}), Cut_mtx: {cut_mtx[y, x]}, Bottom_mtx: {bottom_mtx[y, x]}.")
            if cut_mtx[y, x] == 1:
                if not stixel_started:
                    # Start a new stixel
                    stixel_started = True
                    y_t = y
                    current_stixel = Stixel(x, y_t, None)
                # Check for cut points
                elif bottom_mtx[y, x] == 0:
                    current_stixel.y_t_obj.append(y)
        # In case the stixel doesn't end properly
        if stixel_started:
            print("no end found")
            current_stixel.y_b = rows - 1  # Set to the last row
            stixels.append(current_stixel)
    return stixels


def extract_object_stixel_and_scale(stixels: List[Stixel]):
    stixel_list = []
    for stixel in stixels:
        stixel_list.append(Stixel(stixel.x, stixel.y_t, stixel.y_b, cls=1))
        for obj_stixel in stixel.y_t_obj:
            stixel_list.append(Stixel(stixel.x, obj_stixel, stixel.y_b, cls=0))
    for stixel in stixel_list:
        stixel.scale_by_grid(grid_step=8)
    return stixel_list


def threshold_matrix_np(matrix, threshold):
    return np.where(matrix >= threshold, 1, 0)


class StixelNExTInterpreter:
    def __init__(self, prediction_mtx: torch.Tensor, detection_threshold=0.4):
        prediction_mtx_numpy: np.array = prediction_mtx.numpy()
        # possible option is to blur 3-dimensional (like an RGB color img): cut, bottom
        cut_mtx = prediction_mtx_numpy[0, :, :]
        bottom_mtx = prediction_mtx_numpy[1, :, :]
        self.cut_mtx = cut_mtx
        self.bottom_mtx = bottom_mtx
        self.detection_threshold = detection_threshold
        self.detected_cuts_mtx = threshold_matrix_np(self.cut_mtx, threshold=detection_threshold)
        self.detected_bottoms_mtx = threshold_matrix_np(self.bottom_mtx, threshold=detection_threshold)

    def get_stixel(self):
        stixels = extract_stixels(self.detected_cuts_mtx, self.detected_bottoms_mtx)
        full_stixel_list = extract_object_stixel_and_scale(stixels)
        return full_stixel_list
