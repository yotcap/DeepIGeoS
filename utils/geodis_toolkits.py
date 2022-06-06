import cv2
import random
import numpy as np
from multiprocessing import Pool

import torch
import torchio as tio
import GeodisTK


def focusregion_index(pred_array):

    """ 找到具有最大和值的轴的索引 """

    # pred_array (H,W,D)

    h, w, d = None, None, None

    thres = 0
    for i in range(pred_array.shape[0]):
        if np.sum(pred_array[i]) > thres:
            h = i
            thres = np.sum(pred_array[i])

    thres = 0
    for i in range(pred_array.shape[1]):
        if np.sum(pred_array[:, i]) > thres:
            w = i
            thres = np.sum(pred_array[:, i])

    thres = 0
    for i in range(pred_array.shape[2]):
        if np.sum(pred_array[:, :, i]) > thres:
            d = i
            thres = np.sum(pred_array[:, :, i])

    return h, w, d


def randompoint(seg):

    # 通过组件分析随机选择

    """ 
        在错误的分割区域中随机采样 n 个像素来模拟用户的交互。
        如果欠分割/过分割的区域像素为 Nm，如果 Nm < 30 则 n=0，否则 n=Nm/100
    """

    seg_shape = seg.shape
    seg_array = np.array(seg, dtype=np.uint8)
    focus_h, focus_w, focus_d = focusregion_index(seg_array)
    output = np.zeros(shape=seg_shape)

    if None not in [focus_h, focus_w, focus_d]:
        # h
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            seg_array[focus_h, :, :]
        )
        for i in range(1, len(stats)):
            region_size = stats[i][4]
            if region_size >= 30:
                number_n = int(np.ceil(region_size / 100))
                index_list = np.random.choice(region_size, number_n, replace=False)
                output[
                    focus_h,
                    np.where(labels == i)[0][index_list],
                    np.where(labels == i)[1][index_list],
                ] = 1

        # w
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            seg_array[:, focus_w, :]
        )
        for i in range(1, len(stats)):
            region_size = stats[i][4]
            if region_size >= 30:
                number_n = int(np.ceil(region_size / 100))
                index_list = np.random.choice(region_size, number_n, replace=False)
                output[
                    np.where(labels == i)[0][index_list],
                    focus_w,
                    np.where(labels == i)[1][index_list],
                ] = 1

        # d
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            seg_array[:, :, focus_d]
        )
        for i in range(1, len(stats)):
            region_size = stats[i][4]
            if region_size >= 30:
                number_n = int(np.ceil(region_size / 100))
                index_list = np.random.choice(region_size, number_n, replace=False)
                output[
                    np.where(labels == i)[0][index_list],
                    np.where(labels == i)[1][index_list],
                    focus_d,
                ] = 1

    return output


def randominteraction(pred_array, label_array):
    # 过分割区域
    overseg = np.where(pred_array - label_array == 1, 1, 0)
    sb = randompoint(overseg)  # background

    # 欠分割区域
    underseg = np.where(pred_array - label_array == -1, 1, 0)
    sf = randompoint(underseg)  # foreground
    return sb, sf


def geodismap(sf, sb, input_np):

    # shape 需要一致
    # 原始图像尺寸为：h, w, d

    # sf: 前景交互（欠分割）
    # sb: 后景交互（过分割）

    """ 
        使用栅格扫描获得 3D 测地线距离。
        I: 输入图像 array，能够得到多通道，shape [D, H, W] 或 [D, H, W, C]，类型应为 np.float32。
        S: 二进制图像非零像素用作种子，shape [D, H, W]，类型应为 np.uint8。
        spacing: 分别沿 D、H 和 W 维度的像素间距的浮点数元组。
        lamb: 0.0-1.0之间的权重
            如果 lamb==0.0，返回空间欧几里得距离而不考虑梯度
            如果 lamb==1.0，距离仅基于梯度，而无需使用空间距离
        iter: 栅格扫描的枚举数
    """

    I = np.squeeze(input_np, axis=0).transpose(2, 0, 1)
    sf = np.array(sf, dtype=np.uint8).transpose(2, 0, 1)
    sb = np.array(sb, dtype=np.uint8).transpose(2, 0, 1)
    spacing = tio.ScalarImage(tensor=np.expand_dims(I, axis=0)).spacing

    with Pool(2) as p:
        fore_dist_map, back_dist_map = p.starmap(GeodisTK.geodesic3d_raster_scan, 
                                                 [(I, sf, spacing, 1, 2), (I, sb, spacing, 1, 2)])

    if fore_dist_map.all():
        fore_dist_map = I

    if back_dist_map.all():
        back_dist_map = I

    return fore_dist_map.transpose(1, 2, 0), back_dist_map.transpose(1, 2, 0)


def get_geodismaps(inputs_np, true_labels_np, pred_labels_np):
    fore_dist_map_batch = np.empty(inputs_np.shape, dtype=np.float32)
    back_dist_map_batch = np.empty(inputs_np.shape, dtype=np.float32)

    for i, (input_np, pred_label_np, true_label_np) in enumerate(zip(inputs_np, 
                                                                       pred_labels_np, 
                                                                       true_labels_np)):
        sb, sf = randominteraction(pred_label_np, true_label_np)
        fore_dist_map, back_dist_map = geodismap(sf, sb, input_np)

        fore_dist_map_batch[i] = np.expand_dims(fore_dist_map, axis=0)
        back_dist_map_batch[i] = np.expand_dims(back_dist_map, axis=0)

    return fore_dist_map_batch, back_dist_map_batch