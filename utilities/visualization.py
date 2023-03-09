import matplotlib.pyplot as plt
import numpy as np
import torch


def show_data(x_feature, y_target, idx=0, grid_step=8):
    print(f"Feature batch shape: {x_feature.size()}")
    print(f"Labels batch shape: {y_target.size()}")
    print(x_feature[idx].dtype)
    img = x_feature[idx].squeeze().to(torch.uint8).permute(1, 2, 0)
    label = y_target[idx]
    # Scatter
    pts_mtx = label.numpy()
    xs = []
    ys = []
    cs = []
    # Row
    for i in range(pts_mtx.shape[0]):
        # Col
        for j in range(pts_mtx.shape[1]):
            if pts_mtx[i][j] != 0:
                xs.append(j*grid_step)
                ys.append(i*grid_step)
                cs.append(WAYMO_SEG_COLOR_MAP[pts_mtx[i][j]])
    plt.figure(figsize=(20, 12))
    plt.scatter(xs, ys, c=np.array(cs)/255, s=8.0, edgecolors="none")
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")


def rgba(r, r_max=50):
    """Generates a color based on range.
  Args:
    r: the range value of a given point.
    r_max:
  Returns:
    The color for a given range
  """
    c = plt.get_cmap('plasma')((r % r_max) / r_max)
    c = list(c)
    c[-1] = 0.7  # alpha
    return c


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
