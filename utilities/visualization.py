import matplotlib.pyplot as plt


def show_data(x_feature, y_target, idx=0, grid_step=8):
    print(f"Feature batch shape: {x_feature.size()}")
    print(f"Labels batch shape: {y_target.size()}")
    img = x_feature[idx].squeeze().permute(1, 2, 0)
    print(x_feature[idx].dtype)
    label = y_target[idx]
    # Scatter
    pts_mtx = label.numpy()
    xs = []
    ys = []
    # Row
    for i in range(pts_mtx.shape[0]):
        # Col
        for j in range(pts_mtx.shape[1]):
            if pts_mtx[i][j] != 0:
                xs.append(j*grid_step)
                ys.append(i*grid_step)
    plt.figure(figsize=(20, 12))
    plt.scatter(xs, ys, c='r', s=8.0, edgecolors="none")
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")
