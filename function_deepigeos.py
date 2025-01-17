from PyQt5.QtWidgets import*
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5 import *
from PyQt5 import uic
from PyQt5.QtGui import QImage, qRgb
import numpy as np
import nibabel as nib
import cv2, os
import GeodisTK

import glob
import ipywidgets as ipyw
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torchio as tio

from utils.geodis_toolkits import geodismap
from models.networks import P_RNet3D
from data_loaders.transforms import get_transform


def clk_seg(usrId, count, path, int_pos, int_neg, axis, img, pn, clk):
    pos = (int_pos==0)
    neg = (int_neg==0)
    
    clk = (clk.x(), clk.y())
    
    if pn == 1:

        if not f'pos_{count}.npy' in os.listdir(path):
            np.save(f'../res/{usrId}/seg/{axis}/pos_{count}.npy', pos)

        img = cv2.circle(img, (clk[0], clk[1]), 8, (0, 0, 0), 3)

        cv2.imwrite(f'../res/{usrId}/seg/{axis}/{count}.png', img)

        pos = np.load(f'../res/{usrId}/seg/{axis}/pos_{count}.npy')
        pos[clk[1], clk[0]] = 1
        np.save(f'../res/{usrId}/seg/{axis}/pos_{count}.npy', pos)

        return img

    else:

        if not f'neg_{count}.npy' in os.listdir(path):
            np.save(f'../res/{usrId}/seg/{axis}/neg_{count}.npy', neg)

        img = cv2.rectangle(img, (clk[0]-8, clk[1]-8), (clk[0]+8, clk[1]+8), (0, 0, 0), 3)

        cv2.imwrite(f'../res/{usrId}/seg/{axis}/{count}.png', img)

        neg = np.load(f'../res/{usrId}/seg/{axis}/neg_{count}.npy')
        neg[clk[1], clk[0]] = 1
        np.save(f'../res/{usrId}/seg/{axis}/neg_{count}.npy', neg)

        return img


def nextImage( usrId, imgs, segs, ax, count, pn, clk=(0,0)):

    if ax==0 : axis= 'X'
    elif ax==1 : axis= 'Y'
    elif ax==2 : axis= 'Z'

    path = f'../res/{usrId}/seg/{axis}/'

    if count >= imgs.shape[0]:
        count = imgs.shape[0]
    elif count < 0:
        count = 0


    if ax == 0:
        seg = segs[count,:,:]
        iH, iW = imgs[count,:,:].shape
    elif ax == 1: 
        seg = segs[:,count,:]
        iH, iW = imgs[:,count,:].shape
    elif ax == 2: 
        seg = segs[:,:,count]
        iH, iW = imgs[:,:,count].shape


    if not f'{count}.png' in os.listdir(path):
        int_pos = np.uint8(255*np.ones([iW*2, iH*2]))
        int_neg = np.uint8(255*np.ones([iW*2, iH*2]))
        if ax == 0:
            img = imgs[count,:,:]          
        elif ax == 1: 
            img = imgs[:,count,:]          
        elif ax == 2: 
            img = imgs[:,:,count]   
            
        img = np.rot90(img, 1)
        img = np.flip(img, 1)
        img = cv2.divide(img, img.max())
        img = cv2.resize(img, (iH*2, iW*2))
        img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        pos = (int_pos==0)
        neg = (int_neg==0)

    else:
        int_pos = np.uint8(255*np.ones([iW*2, iH*2]))
        int_neg = np.uint8(255*np.ones([iW*2, iH*2]))

        img = cv2.imread(f'../res/{usrId}/seg/{axis}/{count}.png')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    seg = np.rot90(seg, 1)
    seg = np.flip(seg, 1)
    seg = cv2.divide(seg, seg.max())
    seg[np.where(seg!=0)]=1
    seg = cv2.normalize(src=seg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    seg = cv2.resize(seg, (iH*2, iW*2))

    if clk == (0,0):
        return img, seg

    img = clk_seg(usrId, count, path, int_pos, int_neg, axis, img, pn, clk)

    return img, seg


def seg_reduction(int_seg):
    h, w = int_seg.shape
    idx = np.where(int_seg==1)
    
    for i in range(idx[0].shape[0]):
        int_seg[idx[0][i], idx[1][i] +1] =1

        int_seg[idx[0][i]+1, idx[1][i] +0] =1
        int_seg[idx[0][i]+1, idx[1][i] +1] =1
        
    int_seg = np.rot90(int_seg, 1)
    int_seg = np.flip(int_seg, 1)
    int_seg = cv2.resize(int_seg, (int(h/2), int(w/2)), interpolation = cv2.INTER_NEAREST)
    
    return int_seg


def save_func(imgs, path, usrId):
    file_path = []
    for (root, directories, files) in os.walk(path):
        for file in files:
            if '.npy' in file:
                file_path.append(os.path.join(root, file))
                
    int_pos_result = np.uint8(255*np.ones(imgs.shape))
    int_neg_result = np.uint8(255*np.ones(imgs.shape))

    for i in file_path:
        axis = i.split('/')[-2]
        int_side = i.split('/')[-1].split('.')[0].split('_')[0]
        count = int(i.split('/')[-1].split('.')[0].split('_')[1])
        
        int_seg = np.load(i)
        int_seg = int_seg.astype('uint8')

        if int_side == 'pos':
            if axis == 'X':
                int_pos_result[count,:,:] = seg_reduction(int_seg)
            elif axis == 'Y':
                int_pos_result[:,count,:] = seg_reduction(int_seg)
            elif axis == 'Z':
                int_pos_result[:,:,count] = seg_reduction(int_seg)
                
        elif int_side == 'neg':
            if axis == 'X':
                int_neg_result[count,:,:] = seg_reduction(int_seg)
            elif axis == 'Y':
                int_neg_result[:,count,:] = seg_reduction(int_seg)
            elif axis == 'Z':
                int_neg_result[:,:,count] = seg_reduction(int_seg)
                
    int_pos_result = (int_pos_result==1)
    int_neg_result = (int_neg_result==1)

    np.save(f'../res/{usrId}/result/int_pos_result.npy', int_pos_result)
    np.save(f'../res/{usrId}/result/int_neg_result.npy', int_neg_result)

    return int_pos_result, int_neg_result


def pnet_inference(
    image_path,
    save_path,
    pnet, 
    transform, 
    norm_transform, 
    device
):
    """
        P-Net inference function

        参数：
            image_path:     输入图像的文件路径（例：image_path.nii.gz）
            save_path:      保存结果的路径（例：pnet_pred.nii.gz）
            pnet:           训练好的 pnet 模型（torch.nn.Module）
            transform:      预处理 transforms（torchio.Compose）
            norm_transform: 预处理 transforms（normalization）
            device:         torch device (torch.device)
    """

    # 读取图片并使其应用转换
    subject = tio.Subject(
        image = tio.ScalarImage(image_path),
    )
    subject = transform(subject)
    subject = norm_transform(subject)

    # 使 numpy array 到 torch tensor
    input_image = subject.image.data
    input_tensor = input_image.unsqueeze(dim=0).to(device)

    # inference
    with torch.no_grad():
        pred_logits = pnet(input_tensor)
    
    # logits 到 labels
    pred_labels = torch.argmax(pred_logits, dim=1)

    # labels 到一个 hot labels （例：[1, 2, 3] -> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]）
    pred_onehot = torch.nn.functional.one_hot(pred_labels, 2).permute(0, 4, 1, 2, 3)
    pred_onehot_target = pred_onehot[:, 1, ...]

    # 保存结果
    pred_labelmap = tio.LabelMap(tensor=pred_onehot_target.cpu())
    pred_labelmap.save(save_path)    
    
    
def rnet_inference(
    image_path, 
    pnet_pred_path,
    fg_point_path, 
    bg_point_path, 
    save_path,
    rnet, 
    transform, 
    norm_transform, 
    device
):
    """
        R-Net inference function

        参数：
            image_path:
            pnet_pred_path:     pnet 预测标签文件路径（例：pnet_pred.nii.gz）
            fg_point_path:      用户交互的前景点文件地址（例：fg_points.npy）
            bg_point_path:      用户交互的后景点文件地址（例：bg_points.npy）
            save_path:          保存结果的路径（例：pnet_pred.nii.gz）
            rnet:               训练好的 rnet 模型（torch.nn.Module）
            transform:
            norm_transform:
            device:
    """

    # 读取图片并使其应用转换
    subject = tio.Subject(
        image = tio.ScalarImage(image_path),
        pnet_pred = tio.LabelMap(pnet_pred_path)
    )
    subject = transform(subject)
    subject_norm = norm_transform(subject)

    input_image = subject.image.data
    input_image_norm = subject_norm.image.data
    input_tensor_norm = input_image_norm.unsqueeze(dim=0).to(device)

    pnet_pred = tio.LabelMap(pnet_pred_path)
    pnet_pred_label = pnet_pred.data
    pnet_pred_tensor = pnet_pred_label.unsqueeze(dim=0).to(device)

    sf, sb = np.load(fg_point_path), np.load(bg_point_path)

    # 从随机点中过去测地距离图，并应用 transform
    sf, sb = sf.astype(np.float32), sb.astype(np.float32)
    fore_dist_map, back_dist_map = geodismap(sf, sb, input_image.numpy())
    fore_dist_map = torch.Tensor(norm_transform(np.expand_dims(fore_dist_map, axis=0)))
    back_dist_map = torch.Tensor(norm_transform(np.expand_dims(back_dist_map, axis=0)))

    # 使 rnet 输入到 tensor 中
    rnet_inputs = torch.cat([
        input_tensor_norm,
        pnet_pred_tensor, 
        fore_dist_map.unsqueeze(dim=1).to(device), 
        back_dist_map.unsqueeze(dim=1).to(device)
    ], dim=1)

    # inference
    with torch.no_grad():
        pred_logits = rnet(rnet_inputs)
    
    pred_labels = torch.argmax(pred_logits, dim=1)

    pred_onehot = torch.nn.functional.one_hot(pred_labels, 2).permute(0, 4, 1, 2, 3)
    pred_onehot_target = pred_onehot[:, 1, ...]

    # 保存结果
    pred_labelmap = tio.LabelMap(tensor=pred_onehot_target.cpu())
    pred_labelmap.save(save_path)
