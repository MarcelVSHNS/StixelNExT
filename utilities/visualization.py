import matplotlib.pyplot as plt
import numpy as np
import torch
import io
from PIL import Image


def show_meta_data(x_feature, y_target, idx=0):
    print(f"Feature batch shape: {x_feature.size()}")
    print(f"Labels batch shape: {y_target.size()}")
    print(x_feature[idx].dtype)


def show_data_pair(img_pair):
    fig = plt.figure(figsize=(8, 12))
    fig.add_axes([0.06, 0.00, 0.9, 0.95])
    plt.imshow(img_pair)
    plt.show()


def create_sample_comparison(x_feature, y_prediction, y_target, idx=-1, t_infer=0.0, threshold=0.3):
    img = x_feature[idx].cpu().detach()
    pred_mtx = y_prediction[idx].cpu().detach().numpy()
    target_mtx = y_target[idx].cpu().detach().numpy()

    prediction_img = __draw_stixel_on_image(img, pred_mtx, threshold=threshold,
                                          title=f'Prediction in {t_infer/1000000} ms & Threshold of ')
    target_img = __draw_stixel_on_image(img, target_mtx, title='Ground Truth ')

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


def __draw_stixel_on_image(img, stxl_mtx, threshold=1.0, grid_step=8, title=""):
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
    plt.close()
    plt.clf()
    return stxl_img


WAYMO_SEGMENTATION = {
    'TYPE_UNDEFINED': 0,
    'TYPE_EGO_VEHICLE': 1,
    'TYPE_CAR': 2,
    'TYPE_TRUCK': 3,
    'TYPE_BUS': 4,
    'TYPE_OTHER_LARGE_VEHICLE': 5,
    'TYPE_BICYCLE': 6,
    'TYPE_MOTORCYCLE': 7,
    'TYPE_TRAILER': 8,
    'TYPE_PEDESTRIAN': 9,
    'TYPE_CYCLIST': 10,
    'TYPE_MOTORCYCLIST': 11,
    'TYPE_BIRD': 12,
    'TYPE_GROUND_ANIMAL': 13,
    'TYPE_CONSTRUCTION_CONE_POLE': 14,
    'TYPE_POLE': 15,
    'TYPE_PEDESTRIAN_OBJECT': 16,
    'TYPE_SIGN': 17,
    'TYPE_TRAFFIC_LIGHT': 18,
    'TYPE_BUILDING': 19,
    'TYPE_ROAD': 20,
    'TYPE_LANE_MARKER': 21,
    'TYPE_ROAD_MARKER': 22,
    'TYPE_SIDEWALK': 23,
    'TYPE_VEGETATION': 24,
    'TYPE_SKY': 25,
    'TYPE_GROUND': 26,
    'TYPE_DYNAMIC': 27,
    'TYPE_STATIC': 28
}


WAYMO_SEG_COLOR_MAP = {
    WAYMO_SEGMENTATION['TYPE_UNDEFINED']: [0, 0, 0],
    WAYMO_SEGMENTATION['TYPE_EGO_VEHICLE']: [102, 102, 102],
    WAYMO_SEGMENTATION['TYPE_CAR']: [0, 0, 142],
    WAYMO_SEGMENTATION['TYPE_TRUCK']: [0, 0, 70],
    WAYMO_SEGMENTATION['TYPE_BUS']: [0, 60, 100],
    WAYMO_SEGMENTATION['TYPE_OTHER_LARGE_VEHICLE']: [61, 133, 198],
    WAYMO_SEGMENTATION['TYPE_BICYCLE']: [119, 11, 32],
    WAYMO_SEGMENTATION['TYPE_MOTORCYCLE']: [0, 0, 230],
    WAYMO_SEGMENTATION['TYPE_TRAILER']: [111, 168, 220],
    WAYMO_SEGMENTATION['TYPE_PEDESTRIAN']: [220, 20, 60],
    WAYMO_SEGMENTATION['TYPE_CYCLIST']: [255, 0, 0],
    WAYMO_SEGMENTATION['TYPE_MOTORCYCLIST']: [180, 0, 0],
    WAYMO_SEGMENTATION['TYPE_BIRD']: [127, 96, 0],
    WAYMO_SEGMENTATION['TYPE_GROUND_ANIMAL']: [91, 15, 0],
    WAYMO_SEGMENTATION['TYPE_CONSTRUCTION_CONE_POLE']: [230, 145, 56],
    WAYMO_SEGMENTATION['TYPE_POLE']: [153, 153, 153],
    WAYMO_SEGMENTATION['TYPE_PEDESTRIAN_OBJECT']: [234, 153, 153],
    WAYMO_SEGMENTATION['TYPE_SIGN']: [246, 178, 107],
    WAYMO_SEGMENTATION['TYPE_TRAFFIC_LIGHT']: [250, 170, 30],
    WAYMO_SEGMENTATION['TYPE_BUILDING']: [70, 70, 70],
    WAYMO_SEGMENTATION['TYPE_ROAD']: [128, 64, 128],
    WAYMO_SEGMENTATION['TYPE_LANE_MARKER']: [234, 209, 220],
    WAYMO_SEGMENTATION['TYPE_ROAD_MARKER']: [217, 210, 233],
    WAYMO_SEGMENTATION['TYPE_SIDEWALK']: [244, 35, 232],
    WAYMO_SEGMENTATION['TYPE_VEGETATION']: [107, 142, 35],
    WAYMO_SEGMENTATION['TYPE_SKY']: [70, 130, 180],
    WAYMO_SEGMENTATION['TYPE_GROUND']: [102, 102, 102],
    WAYMO_SEGMENTATION['TYPE_DYNAMIC']: [102, 102, 102],
    WAYMO_SEGMENTATION['TYPE_STATIC']: [102, 102, 102]
}
