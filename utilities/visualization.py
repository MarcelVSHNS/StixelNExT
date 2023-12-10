import matplotlib.pyplot as plt
import numpy as np
import torch
import io
from PIL import Image
import random
from typing import List
import cv2
import matplotlib.patches as patches


def show_meta_data(x_feature, y_target, idx=0):
    print(f"Feature batch shape: {x_feature.size()}")
    print(f"Labels batch shape: {y_target.size()}")
    print(x_feature[idx].dtype)


def show_data_pair(img_pair):
    fig = plt.figure(figsize=(8, 11))
    fig.add_axes([0.06, 0.02, 0.9, 0.95])
    plt.imshow(img_pair)
    plt.show()


def create_sample_comparison(x_feature, y_prediction, y_target, idx=-1, t_infer=0.0, threshold=0.3):
    if y_prediction.shape[0] == 160:
        y_prediction = torch.unsqueeze(y_prediction, 0)
    pred_mtx = y_prediction[idx].numpy()
    target_mtx = y_target[idx].numpy()

    prediction_img = __draw_stixel_on_image(x_feature[idx], pred_mtx, threshold=threshold,
                                          title=f'Prediction in {t_infer/1000000} ms & Threshold of ')
    target_img = __draw_stixel_on_image(x_feature[idx], target_mtx, title='Ground Truth ')

    img_pair = __vertical_img_stack(prediction_img, target_img)
    return img_pair


def __horizontal_img_stack(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def __vertical_img_stack(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def __draw_stixel_on_image(img, stxl_mtx, threshold=1.0, grid_step=8, title="", save_fig=False):
    img = img.squeeze().to(torch.uint8).permute(1, 2, 0)
    pts_mtx = stxl_mtx
    xs = []
    ys = []
    cs = []
    # Row
    for i in range(pts_mtx.shape[0]):
        # Col
        for j in range(pts_mtx.shape[1]):
            if pts_mtx[i][j] >= threshold:
                xs.append(j * grid_step)
                ys.append(i * grid_step)
                if threshold != 1:
                    cs.append([255, 139, 254])
                else:
                    cs.append([15, 223, 61])  # WAYMO_SEG_COLOR_MAP[pts_mtx[i][j]]
    plt.figure(figsize=(20, 12))
    plt.title(f'{title} ({int(threshold*100)} %)')
    plt.scatter(xs, ys, c=np.array(cs) / 255, s=8.0, edgecolors="none")
    plt.imshow(img)
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    stxl_img = Image.open(img_buf)
    if save_fig:
        name = "image_export/" + str(random.randint(100000, 999999)) + ".png"
        stxl_img.save(name)
    plt.close()
    plt.clf()
    return stxl_img


def plot_roc_curve(fpr, tpr, thres_idx=None, display=False):
    plt.plot(fpr, tpr)
    if thres_idx:
        plt.plot(fpr[thres_idx], tpr[thres_idx], "or")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if display:
        plt.show()
    else:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight')
        roc_curve_img = Image.open(img_buf)
        plt.close()
        plt.clf()
        return roc_curve_img


def draw_stixels_on_image(image_tensor, stixels, stixel_width=8):
    tensor_np_image = image_tensor.permute(1, 2, 0).numpy()
    np_image_rgb = np.array((tensor_np_image * 255).astype(np.uint8))
    opencv_image = cv2.cvtColor(np_image_rgb, cv2.COLOR_RGB2BGR)
    # black_image = np.zeros((1200, 1920, 3), dtype=np.uint8)
    for stixel in stixels:
        top_left_x, top_left_y = stixel.x, stixel.y_t
        bottom_left_x, bottom_left_y = stixel.x, stixel.y_b
        color = (0, 255, 0)
        bottom_right_x = bottom_left_x + stixel_width
        cv2.rectangle(opencv_image, (top_left_x, top_left_y), (bottom_right_x, bottom_left_y), color, 1)
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    normal_image = 255 - rgb_image
    return Image.fromarray(normal_image)
