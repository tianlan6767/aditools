#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：adi-datatool 
@File    ：filters.py
@Author  ：LvYong
@Date    ：2022/4/11 14:29 
"""
import os
import shutil

import cv2
import pywt
import numpy as np
from typing import Union, Tuple
from comm import create_dir, get_file_infos
from glob import glob
from tqdm import tqdm
from time import perf_counter


def nothing(x):
    pass


def dual_kernel_filter(image_folder,
                       s_kernel: Union[int, Tuple[int, int]] = 3,
                       b_kernel: Union[int, Tuple[int, int]] = 11):

    bin_folder = create_dir(os.path.join(image_folder, 'bin'))
    if isinstance(s_kernel, int):
        s_kernel = (s_kernel, s_kernel)
    if isinstance(b_kernel, int):
        b_kernel = (b_kernel, b_kernel)
    # pro_bar = tqdm(sorted(glob(os.path.join(image_folder, '*.bmp'))))
    for idx, img_file in enumerate(sorted(glob(os.path.join(image_folder, '*.bmp')))):
        img_name = get_file_infos(img_file)[1]
        img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        # img = cv2.equalizeHist(img)
        # _, bin_img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

        # cv2.imencode('.bmp', cv2.equalizeHist(img))[1].tofile(os.path.join(bin_folder, img_name))
        # continue

        s_img = cv2.blur(img, s_kernel)
        b_img = cv2.blur(img, b_kernel)
        r_img = cv2.subtract(b_img, s_img)

        _, bin_img = cv2.threshold(r_img, 10, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # cv2.namedWindow("image")
        # cv2.createTrackbar("d", "image", 0, 255, nothing)
        # cv2.createTrackbar("sigmaColor", "image", 0, 255, nothing)
        # cv2.createTrackbar("sigmaSpace", "image", 0, 255, nothing)
        # while True:
        #     d = cv2.getTrackbarPos("d", "image")
        #     sigmaColor = cv2.getTrackbarPos("sigmaColor", "image")
        #     sigmaSpace = cv2.getTrackbarPos("sigmaSpace", "image")
        #     out_img = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
        #     cv2.imshow("result", out_img)
        #     k = cv2.waitKey(1) & 0xFF
        #     if k == 27:
        #         break
        # cv2.destroyAllWindows()
        #
        # temp = dyn_threshold(img, cv2.blur(img, (30, 30)), 20, 'dark')
        # # temp = cv2.blur(img, (9, 9))
        # cv2.imencode('.bmp', temp)[1].tofile(os.path.join(bin_folder, img_name))

        print(f'\n{img_name}')
        filtered_contours = []
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            # 面积过滤
            if 50 < area:
                rect = cv2.boundingRect(contours[i])
                shape = (rect[3], rect[2])
                mask = np.zeros(shape, dtype=np.uint8)
                cv2.fillPoly(mask, [contours[i] - (rect[0], rect[1])], color=255)
                roi = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                mean, stddev = cv2.meanStdDev(roi, mask=mask)
                # 均值和方差过滤
                # 长宽比过滤
                print(f'---mean:{mean[0][0]:^7.3f} stddev:{stddev[0][0]:^7.3f} '
                      f'center:({rect[0] + rect[2] // 2}, {rect[1] + rect[3] // 2})')
                if 7 < mean[0][0] < 18 and 2 < stddev[0][0] < 10 and 0.33 < rect[3] / rect[2] < 3:
                    print(f'---mean:{mean[0][0]:^7.3f} stddev:{stddev[0][0]:^7.3f} '
                          f'center:({rect[0] + rect[2] // 2}, {rect[1] + rect[3] // 2})')
                    filtered_contours.append(contours[i])

        empty_img = np.zeros(img.shape, dtype=np.uint8)
        for filtered_contour in filtered_contours:
            cv2.drawContours(empty_img, [filtered_contour], -1, 255, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        empty_img2 = cv2.dilate(empty_img, kernel)
        empty_img2 = cv2.subtract(empty_img2, empty_img)
        contours, hierarchy = cv2.findContours(empty_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        empty_img = img.copy()

        for i in range(len(contours)):
            rect = cv2.boundingRect(contours[i])
            roi = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            mask = empty_img2[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            mean, stddev = cv2.meanStdDev(roi, mask=mask)

            if 10 <= mean[0][0] < 41 and 2 < stddev[0][0] < 23:
                print(f'+++mean:{mean[0][0]:^7.3f} stddev:{stddev[0][0]:^7.3f} '
                      f'center:({rect[0] + rect[2] // 2}, {rect[1] + rect[3] // 2})')
                cv2.drawContours(empty_img, [contours[i]], -1, 255, 1)

        cv2.imencode('.bmp', empty_img)[1].tofile(os.path.join(bin_folder, img_name))


def dyn_threshold(src, pre, offset, thresh_type):
    temp = src - pre
    dst = np.zeros(src.shape, dtype=np.uint8)
    if thresh_type == 'light':
        dst = np.where(temp >= offset, 255, 0)
    elif thresh_type == 'dark':
        dst = np.where(pre - src >= offset, 255, 0)
    elif thresh_type == 'equal':
        for r in range(temp.shape[0]):
            for c in range(temp.shape[1]):
                if -offset <= temp[r, c] <= offset:
                    dst[r, c] = 255
                else:
                    dst[r, c] = 0
    else:
        for r in range(temp.shape[0]):
            for c in range(temp.shape[1]):
                if temp[r, c] < -offset or temp[r, c] > offset:
                    dst[r, c] = 255
                else:
                    dst[r, c] = 0
    return dst


def test_wave(img_path, save_path):
    # for f in pywt.families():
    #     print(pywt.wavelist(family=f, kind='discrete'))
    create_dir(save_path)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    src = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    cv2.imencode('.bmp', src)[1].tofile(os.path.join(save_path, f'{img_name}_gray.bmp'))

    cA, (cH, cV, cD) = pywt.dwt2(src, 'haar')

    AH = np.concatenate([cA, cH + 255], axis=1)
    VD = np.concatenate([cV + 255, cD + 255], axis=1)
    img = np.concatenate([AH, VD], axis=0)
    img0 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img)
    # img0 = (img - minVal) / (maxVal - minVal) * 255
    cv2.imencode('.bmp', img0)[1].tofile(os.path.join(save_path, f'{img_name}_1.bmp'))

    # 将每个子图的像素范围都归一化到与CA2一致  CA2 [0, 255 * 2**level]
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(src, 'haar', level=2)
    AH2 = np.concatenate([cA2, cH2 + 510], axis=1)
    VD2 = np.concatenate([cV2 + 510, cD2 + 510], axis=1)
    cA1 = np.concatenate([AH2, VD2], axis=0)

    AH = np.concatenate([cA1, (cH1 + 255) * 2], axis=1)
    VD = np.concatenate([(cV1 + 255) * 2, (cD1 + 255) * 2], axis=1)
    img = np.concatenate([AH, VD], axis=0)
    img0 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imencode('.bmp', img0)[1].tofile(os.path.join(save_path, f'{img_name}_2.bmp'))

    cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(src, 'haar', level=3)
    AH3 = np.concatenate([cA3, cH3 + 1020], axis=1)
    VD3 = np.concatenate([cV3 + 1020, cD3 + 1020], axis=1)
    cA2 = np.concatenate([AH3, VD3], axis=0)

    AH2 = np.concatenate([cA2, (cH2 + 510) * 2], axis=1)
    VD2 = np.concatenate([(cV2 + 510) * 2, (cD2 + 510) * 2], axis=1)
    cA1 = np.concatenate([AH2, VD2], axis=0)

    AH = np.concatenate([cA1, (cH1 + 255) * 4], axis=1)
    VD = np.concatenate([(cV1 + 255) * 4, (cD1 + 255) * 4], axis=1)
    img = np.concatenate([AH, VD], axis=0)
    img0 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imencode('.bmp', img0)[1].tofile(os.path.join(save_path, f'{img_name}_3.bmp'))

    # import matplotlib.pyplot as plt
    # plt.imshow(img, 'gray')
    # plt.title('2D WT')
    # plt.savefig(os.path.join(save_path, f'{img_name}.jpg'))
    # plt.show()


def get_image_content(img_path, save_path):
    create_dir(save_path)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    src = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # src = cv2.resize(src, (2048, 1504))
    # cv2.imencode('.bmp', src)[1].tofile(os.path.join(save_path, f'{img_name}.bmp'))
    # return

    # wavelets = ['haar', 'db1', 'db4', 'sym2', 'sym4', 'coif1', 'bior1.1', 'rbio1.1']
    wavelets = ['haar']
    level = 3
    # save_path = os.path.join(save_path, f'{img_name}_wavelets_{level}')
    # create_dir(save_path)
    for wl in wavelets:
        start_t = perf_counter()
        coeffs = pywt.wavedec2(src, wl, level=level)
        img0 = cv2.normalize(coeffs[0], None, 0, 255, cv2.NORM_MINMAX)
        img0 = cv2.resize(img0, (512, 384))
        print(f'time: {(perf_counter() - start_t) * 1000:.3f} ms')
        cv2.imencode('.bmp', img0)[1].tofile(os.path.join(save_path, f'{img_name}_{wl}_{level}.bmp'))


def generate_basis(img_path, save_path):
    create_dir(save_path)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    src = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)

    coeffs = pywt.wavedec2(src, 'haar', level=3)
    img = cv2.normalize(coeffs[0], None, 0, 255, cv2.NORM_MINMAX)
    cv2.imencode('.bmp', img)[1].tofile(os.path.join(save_path, f'{img_name}_haar_base.bmp'))

    # 屏蔽高频信息
    for i in range(1, len(coeffs)):
        coeffs[i] = list(coeffs[i])
        for j in range(len(coeffs[i])):
            coeffs[i][j] = np.zeros_like(coeffs[i][j])
    basis = pywt.waverec2(coeffs, 'haar')

    cv2.imencode('.bmp', basis)[1].tofile(os.path.join(save_path, f'{img_name}_haar.bmp'))


def getVarianceMean(scr, winSize):
    if scr is None or winSize is None:
        print("The input parameters of getVarianceMean Function error")
        return -1

    if winSize % 2 == 0:
        print("The window size should be singular")
        return -1

    copyBorder_map = cv2.copyMakeBorder(scr, winSize // 2, winSize // 2, winSize // 2, winSize // 2,
                                        cv2.BORDER_REPLICATE)
    row, col = np.shape(scr)[:2]

    local_mean = np.zeros_like(scr)
    local_std = np.zeros_like(scr)

    for i in range(row):
        for j in range(col):
            temp = copyBorder_map[i:i + winSize, j:j + winSize]
            local_mean[i, j], local_std[i, j] = cv2.meanStdDev(temp)
            if local_std[i, j] <= 0:
                local_std[i, j] = 1e-8

    return local_mean, local_std


def adaptContrastEnhancement(img_path, save_path, winSize, maxCg, k=''):
    if img_path is None or winSize is None or maxCg is None:
        print("The input parameters of ACE Function error")
        return -1

    src = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    create_dir(save_path)

    # 转换通道
    YUV_img = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
    Y_Channel = YUV_img[:, :, 0]
    shape = np.shape(Y_Channel)
    meansGlobal = cv2.mean(Y_Channel)[0]
    localMean_map, localStd_map = getVarianceMean(Y_Channel, winSize)

    for i in range(shape[0]):
        for j in range(shape[1]):
            cg = 0.2 * meansGlobal / localStd_map[i, j]
            if cg > maxCg:
                cg = maxCg
            elif cg < 1:
                cg = 1

            temp = Y_Channel[i, j].astype(float)
            temp = max(0, min(localMean_map[i, j] + cg * (temp - localMean_map[i, j]), 255))

            # Y_Channel[i,j]=max(0,min(localMean_map[i,j]+cg*(Y_Channel[i,j]-localMean_map[i,j]),255))
            Y_Channel[i, j] = temp

    YUV_img[:, :, 0] = Y_Channel

    dst = cv2.cvtColor(YUV_img, cv2.COLOR_YUV2BGR)
    cv2.imencode('.bmp', dst)[1].tofile(os.path.join(save_path, f'{img_name}_ace{k}.bmp'))

    return dst


def draw_segment(src_folder, pre_folder, min_pro=0.5, min_area=10):
    src_images = glob(os.path.join(src_folder, '*.bmp'))
    save_folder = create_dir(os.path.join(src_folder, 'result'))
    pro_bar = tqdm(src_images, desc='分割处理')
    for img_file in pro_bar:
        img_name, name = get_file_infos(img_file)[1:3]
        img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)

        # img = cv2.resize(img, (512, 384))
        # cv2.imencode('.bmp', img)[1].tofile(os.path.join(save_folder, img_name))
        # continue

        pre_file = os.path.join(pre_folder, f'{name}.png')
        if not os.path.exists(pre_file):
            continue
        pre_img = cv2.imdecode(np.fromfile(pre_file, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        _, bin_img = cv2.threshold(pre_img, int(min_pro * 255), 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        filtered_contours = []
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if min_area < area:
                filtered_contours.append(contours[i])

        for filtered_contour in filtered_contours:
            cv2.drawContours(img, [filtered_contour], -1, (0, 0, 255), 3)

        cv2.imencode('.bmp', img)[1].tofile(os.path.join(save_folder, img_name))


if __name__ == "__main__":
    # test_wave(r"D:\Projects\819XC\4-2_2_1.bmp",
    #           r'D:\Projects\819XC\out')

    # generate_basis(r"D:\Projects\819XC\out\4-2_2_1.bmp", r'D:\Projects\819XC\out')

    for _, f in enumerate(sorted(glob(os.path.join(r'D:\Projects\819XC\100', '*.bmp')))):
        get_image_content(f,
                          r'D:\Projects\819XC\100\wet')

    # dual_kernel_filter(r'D:\out\9-2_2_1_wavelets_4', 3, 20)

    # img_path = r"D:\out\4-2_2_1_wavelets_4\4-2_2_1_haar_4.bmp"
    # save_path = os.path.join(os.path.dirname(img_path), 'ace')
    # img_name = os.path.splitext(os.path.basename(img_path))[0]
    # adaptContrastEnhancement(img_path,
    #                          save_path,
    #                          winSize=9, maxCg=20)
    # adaptContrastEnhancement(img_path,
    #                          save_path,
    #                          winSize=15, maxCg=3, k='1')
    #
    # img1 = cv2.imdecode(np.fromfile(os.path.join(save_path, f'{img_name}_ace1.bmp'), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imdecode(np.fromfile(os.path.join(save_path, f'{img_name}_ace.bmp'), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    # res1 = dyn_threshold(img1, img2, 15, 'equal')
    # cv2.imencode('.bmp', res1)[1].tofile(os.path.join(save_path, 'aa.bmp'))
    #
    # contours, hierarchy = cv2.findContours(res1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # filtered_contours = []
    # for i in range(len(contours)):
    #     area = cv2.contourArea(contours[i])
    #     # 面积过滤
    #     if 50 < area:
    #         filtered_contours.append(contours[i])
    #
    # empty_img = np.zeros(img1.shape, dtype=np.uint8)
    # for filtered_contour in filtered_contours:
    #     cv2.drawContours(empty_img, [filtered_contour], -1, 255, -1)
    #
    # cv2.imencode('.bmp', empty_img)[1].tofile(os.path.join(save_path, 'bb.bmp'))

    # draw_segment(r'D:\2工位\wet',
    #              r'D:\2工位\wet', min_area=500)

    # src = r'D:\1-50'
    # dst = r'D:\3工位'
    # create_dir(dst)
    # names = ['2_1_3.bmp', '2_2_3.bmp']
    # folders = ["ORIG"]
    # for folder in os.listdir(src):
    #     files = [os.path.join(src, folder, fd, n) for n in names for fd in folders]
    #     [shutil.move(f, os.path.join(dst, f'{folder}_{os.path.basename(f)}')) for f in files if os.path.exists(f)]

    # src = r'D:\Projects\819XC\比较浅显\2工位\0408\wet'
    # for f in glob(os.path.join(src, '*.bmp')):
    #     shutil.copy(f, os.path.join(src, f'0408_{os.path.basename(f)}'))

    print('done')
