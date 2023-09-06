#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：adi-classifier 
@File    ：decorator.py
@Author  ：LvYong
@Date    ：2022/3/1 21:17 
"""
from typing import Optional, List, Union
import inspect
import sys
from skimage.feature import greycomatrix, greycoprops
import numpy as np
import pandas as pd
import cv2
from scipy.fftpack import fft


__all__ = ['get_feature_map', 'get_feature']


class BaseFeature:
    def __init__(self, name: Union[str, List[str]], key: Union[int, List[int]],
                 selected_key: Optional[List[int]] = None):
        self._name = name
        self._key = key
        self.enable = True
        self.selected_key = selected_key

    @property
    def idx(self):
        """排序使用"""
        return self.key if isinstance(self.key, int) else max(self.key)

    @property
    def selected_key(self):
        return self._selected_key

    @selected_key.setter
    def selected_key(self, selected_key: Optional[List[int]]):
        self.enable = True
        self._selected_key = []
        if selected_key is None:
            self._selected_key = [self.key] if isinstance(self.key, int) else self.key
        elif isinstance(self.key, int) and self.key in selected_key:
            self._selected_key = [self.key]
        elif isinstance(self.key, list) and len(set(self.key).intersection(set(selected_key))):
            self._selected_key = list(set(self.key).intersection(set(selected_key)))
        else:
            self.enable = False

    @property
    def name(self):
        return self._name

    @property
    def key(self):
        return self._key

    def _values(self, img, contour, org_contour, *args, **kwargs):
        return []

    def __repr__(self):
        return type(self).__name__

    def __str__(self):
        # 对应特征编号和特征名称
        if isinstance(self.key, int):
            feature_info = {self.key: self.name}
        else:
            feature_info = {self.key[i]: self.name[i] for i in range(len(self.key))}
        return f'{feature_info}'


class FeatureDecorator(BaseFeature):
    def __init__(self, name='', key=0, selected_key=None):
        super(FeatureDecorator, self).__init__(name, key, selected_key)
        self._feature: Optional[FeatureDecorator] = None

    def decorate(self, feature):
        self._feature = feature

    def _values(self, img, contour, org_contour, *args, **kwargs):
        """
            内部调用，计算特征信息

        :param org_contour:
        :param img:
        :param contour:
        :param args:
        :param kwargs:
        :return:
        """
        result = []
        if self._feature is not None:
            result.extend(self._feature._values(img, contour, org_contour, *args, **kwargs))
        return result

    @property
    def selected_all_key(self):
        result = list(self.selected_key)
        if self._feature is not None:
            result.extend(self._feature.selected_all_key)
        elif self.key == 0:
            result = []
        return result

    @property
    def default_feature(self):
        """默认特征"""
        return [0 for _ in range(len(self.selected_all_key))]

    def get_image_feature(self, image, regions, *args, **kwargs):
        """
            获取图像缺陷的特征

        :param image: 图像
        :param regions: 缺陷的区域信息, 包含shape_attributes.all_points_x等字段信息或模型推理结果
        :param args: 其他参数
        :param kwargs: 其他参数
        :return: 特征值
        """
        feature_data = []
        for n, region in enumerate(regions):
            if isinstance(region, dict):
                # 解析json文件
                region_x = np.array(region['shape_attributes']["all_points_x"])
                region_y = np.array(region['shape_attributes']["all_points_y"])
            else:
                # 解析模型推理信息
                polygons = mask_to_polygons(region)[0]
                polygon = polygons[0].astype("int32").tolist()
                for i in range(1, len(polygons)):
                    polygon.extend(polygons[i].astype("int32").tolist())
                region_x = np.array(polygon[::2])
                region_y = np.array(polygon[1::2])

            roi_img, roi_contour = preprocess_region_image(image, region_x, region_y)

            if roi_img is None or roi_contour is None or len(roi_contour[0]) < 3:
                feature_data.append(self.default_feature)
            else:
                feature_data.append(self._values(roi_img, roi_contour, region, *args, **kwargs))
        return feature_data


class GLCMFeature(FeatureDecorator):
    def __init__(self):
        super(GLCMFeature, self).__init__(
            ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'],
            [1, 2, 3, 4, 5, 6]
        )

    def _values(self, img, contour, org_contour, *args, **kwargs):
        glcm = greycomatrix(img,
                            [1],  # 步长
                            [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],  # 方向角度
                            levels=256,  # 灰度共生矩阵阶数
                            normed=False, symmetric=False)

        result = []
        if 1 in self.selected_key:
            result.append(np.mean(greycoprops(glcm, 'contrast')[0]))
        if 2 in self.selected_key:
            result.append(np.mean(greycoprops(glcm, 'dissimilarity')[0]))
        if 3 in self.selected_key:
            result.append(np.mean(greycoprops(glcm, 'homogeneity')[0]))
        if 4 in self.selected_key:
            result.append(np.mean(greycoprops(glcm, 'energy')[0]))
        if 5 in self.selected_key:
            result.append(np.mean(greycoprops(glcm, 'correlation')[0]))
        if 6 in self.selected_key:
            result.append(np.mean(greycoprops(glcm, 'ASM')[0]))

        result.extend(super(GLCMFeature, self)._values(img, contour, org_contour, *args, **kwargs))
        return result


class OSTUFeature(FeatureDecorator):
    """大津阈值"""
    def __init__(self):
        super(OSTUFeature, self).__init__('ostu_ret', 7)

    def _values(self, img, contour, org_contour, *args, **kwargs):
        result = [cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[0]]

        result.extend(super(OSTUFeature, self)._values(img, contour, org_contour, *args, **kwargs))
        return result


class GrayFeature(FeatureDecorator):
    """灰度均值、方差、偏度、峰度 """
    def __init__(self):
        super(GrayFeature, self).__init__(['mean', 'var', 'skewness', 'kurtosis'], [8, 9, 10, 11])

    def _values(self, img, contour, org_contour, *args, **kwargs):
        mask0 = np.zeros_like(img)
        mask0 = cv2.fillPoly(mask0, contour, color=255)

        gray_value = img[mask0 != 0]
        gray_value = pd.Series(gray_value, dtype=np.int64)
        result = []
        if 8 in self.selected_key:
            result.append(np.mean(gray_value))
        if 9 in self.selected_key:
            result.append(np.var(gray_value))
        if 10 in self.selected_key:
            result.append(gray_value.skew())
        if 11 in self.selected_key:
            result.append(gray_value.kurt())

        result.extend(super(GrayFeature, self)._values(img, contour, org_contour, *args, **kwargs))
        return result


class CentripetalMomentRatioFeature(FeatureDecorator):
    """面积、周长、圆形度、向心矩比"""
    def __init__(self):
        super(CentripetalMomentRatioFeature, self).__init__(
            ['area', 'perimeter', 'circularity', 'centripetal_moment_ratio'],
            [12, 13, 14, 15]
        )

    def _values(self, img, contour, org_contour, *args, **kwargs):
        result = []
        area, perimeter = 0, 1
        if 12 in self.selected_key or 14 in self.selected_key or 15 in self.selected_key:
            area = cv2.contourArea(contour, oriented=False)  # ROI面积
            if 12 in self.selected_key:
                result.append(area)

        if 13 in self.selected_key or 14 in self.selected_key:
            perimeter = cv2.arcLength(contour, True)
            if 13 in self.selected_key:
                result.append(perimeter)

        if 14 in self.selected_key:
            result.append(4 * np.pi * area / (perimeter ** 2))

        if 15 in self.selected_key:
            if area <= 1:
                result.append(1)
            else:
                center, size, angle = cv2.minAreaRect(contour)  # 【最小外接矩形】中心点坐标，边长，旋转角度
                vertices = cv2.boxPoints((center, size, angle))  # 最小外接矩形的顶点
                vertices = np.array(vertices, np.int32)
                area_ver = cv2.contourArea(vertices, oriented=False)  # 最小外接矩形面积
                result.append(area / area_ver)

        result.extend(super(CentripetalMomentRatioFeature, self)._values(img, contour, org_contour, *args, **kwargs))
        return result


class GuptaFeature(FeatureDecorator):
    def __init__(self):
        super(GuptaFeature, self).__init__(['F1', 'F2', 'F3', 'F3-F1'], [16, 17, 18, 19])

    def _values(self, img, contour, org_contour, *args, **kwargs):
        M = cv2.moments(contour)
        # 重心
        barycenter = np.array((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        z = np.sqrt(np.sum(np.square(contour[0] - barycenter), axis=1))

        result = []
        # 一阶边界矩
        m1 = sum(z) / len(z)
        # 二阶中心矩
        F1, F2, F3 = 0, 0, 0
        if 16 in self.selected_key or 19 in self.selected_key:
            M2 = sum((z - m1) ** 2) / len(z)
            F1 = M2 ** (1 / 2) / m1
            if 16 in self.selected_key:
                result.append(F1)
        # 三阶中心矩
        if 17 in self.selected_key:
            M3 = sum((z - m1) ** 3) / len(z)
            F2 = M3 ** (1 / 3) / m1 if M3 >= 0 else -(-M3) ** (1 / 3) / m1
            result.append(F2)
        # 四阶中心矩
        if 18 in self.selected_key or 19 in self.selected_key:
            M4 = sum((z - m1) ** 4) / len(z)
            F3 = M4 ** (1 / 4) / m1
            if 18 in self.selected_key:
                result.append(F3)
        if 19 in self.selected_key:
            result.append(F3 - F1)

        result.extend(super(GuptaFeature, self)._values(img, contour, org_contour, *args, **kwargs))
        return result


class FourierFeature(FeatureDecorator):
    def __init__(self):
        super(FourierFeature, self).__init__('FF', 20)

    def _values(self, img, contour, org_contour, *args, **kwargs):
        y = [i[0] + 1j * i[1] for i in contour[0]]
        fft_contour = fft(y)
        N = len(fft_contour)
        if N % 2 != 0:
            fft_contour = fft_contour[:-1]
        fft_contour = abs(fft_contour / abs(fft_contour[1]))
        fft_contour1 = fft_contour[1:int(N / 2) + 1]
        fft_contour1 = [fft_contour1[i] / (i + 1) for i in range(len(fft_contour1))]
        fft_contour2 = fft_contour[int(N / 2) + 1:]
        fft_contour2 = [fft_contour1[i] / abs(i + 1 - N / 2) for i in range(len(fft_contour2))]
        result = [(sum(fft_contour1) + sum(fft_contour2)) / sum(fft_contour[1:])]

        result.extend(super(FourierFeature, self)._values(img, contour, org_contour, *args, **kwargs))
        return result


class InertiaRatioFeature(FeatureDecorator):
    """惯性比"""
    def __init__(self):
        super(InertiaRatioFeature, self).__init__('inertia_ratio', 21)

    def _values(self, img, contour, org_contour, *args, **kwargs):
        mask0 = np.zeros_like(img, dtype=np.uint8)
        mask0 = cv2.fillPoly(mask0, contour, color=255)
        gray_level = np.array(range(256))
        gray_level_2 = gray_level ** 2
        N = cv2.countNonZero(mask0)
        if not N:
            return [0]
        inertia_num = cv2.calcHist([img], [0], mask0, [256], [0, 256])
        inertia_r = inertia_num.ravel() / N
        p_max = max(inertia_r)
        # 直方图的级惯性矩
        I0 = sum(inertia_r * (gray_level_2 + inertia_r ** 2 / 4))
        # 矩形的级惯性矩
        Ir0 = sum(p_max * (gray_level_2 + p_max ** 2 / 4))
        result = [I0 / Ir0]

        result.extend(super(InertiaRatioFeature, self)._values(img, contour, org_contour, *args, **kwargs))
        return result


class ConnectedDomainFeature(FeatureDecorator):
    """计算一个缺陷中包含多少个连通域"""
    def __init__(self):
        super(ConnectedDomainFeature, self).__init__('conDom', 22)

    def _values(self, img, contour, org_contour, *args, **kwargs):
        num = 1
        if isinstance(org_contour, dict):
            # 解析json文件
            region_x = np.array(org_contour['shape_attributes']["all_points_x"])
            region_y = np.array(org_contour['shape_attributes']["all_points_y"])
            for i in range(1, len(region_x)):
                if region_x[i] - region_x[i - 1] > 1 or region_y[i] - region_y[i - 1] > 1:
                    num += 1
        else:
            # 解析模型推理信息
            polygons = mask_to_polygons(org_contour)[0]
            num = len(polygons)

        result = [num]
        # 调用其他特征
        result.extend(super(ConnectedDomainFeature, self)._values(img, contour, org_contour, *args, **kwargs))
        return result


class DemoFeature(FeatureDecorator):
    """特征对象示例"""
    def __init__(self):
        super(DemoFeature, self).__init__('demo', 9999)

    def _values(self, img, contour, org_contour, *args, **kwargs):
        # 计算并添加特征
        result = [9999]
        # 调用其他特征
        result.extend(super(DemoFeature, self)._values(img, contour, org_contour, *args, **kwargs))
        return result


def get_feature(wanted_feature: Optional[List[int]] = None):
    """
        根据特征序列, 动态构建特征计算方法

        通过get_feature_map可以查看所有的特征映射！

    :param wanted_feature: 特征序列，默认为计算所有特征
    :return: 特征计算方法
    """
    features = [obj()
                for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
                if issubclass(obj, FeatureDecorator) and name not in ['FeatureDecorator', 'DemoFeature']]
    features = sorted(features, key=lambda u: u.idx)
    feature_keys = []
    for f in features:
        f.selected_key = wanted_feature
        if isinstance(f.key, int):
            feature_keys.append(f.key)
        else:
            feature_keys.extend(f.key)

    if len(set(feature_keys)) != len(feature_keys):
        raise IOError(f'特征对象定义错误，存在相同的索引. \r {feature_keys}')

    # 特征方法的装饰
    feature = FeatureDecorator()
    for idx in range(len(features) - 1, -1, -1):
        if features[idx].enable:
            features[idx].decorate(feature)
            feature = features[idx]
        else:
            del features[idx]

    return feature


def get_feature_map(feature: Optional[FeatureDecorator] = None):
    """
        获取特征方法的映射信息

    :param feature: 查询的特征，默认查询所有特征的映射
    :return: 特征序列和特征名称的映射信息
    """
    features = [obj()
                for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
                if issubclass(obj, FeatureDecorator) and name != 'FeatureDecorator']
    features = sorted(features, key=lambda u: u.idx)
    feature_map = {}
    feature_key = feature.selected_all_key if feature else []
    for feature in features:
        for f_key, f_name in eval(f'{feature}').items():
            if not feature_key or f_key in feature_key:
                feature_map[f_key] = f_name
    del features
    return feature_map


def preprocess_region_image(image, region_contour_x, region_contour_y):

    if image is None or len(region_contour_x) < 3:
        return None, None

    x_min, x_max = min(region_contour_x), max(region_contour_x)
    y_min, y_max = min(region_contour_y), max(region_contour_y)
    img_height, img_width = image.shape

    if len(region_contour_x) > 200:
        tmp = range(0, len(region_contour_x), 2)
        region_contour_x = region_contour_x[tmp]
        region_contour_y = region_contour_y[tmp]

    # 截取缺陷，外扩0个像素
    box0 = [y_min, x_min, y_max, x_max]  # y是行，x是列
    """
    特殊情况1：标注的坐标呈一行或一列，如
    [501, 736, 501, 738]
    将其改为
    [500, 736, 502, 738]
    """
    if box0[0] == box0[2] or box0[1] == box0[3]:
        if box0[0] == box0[2]:
            region_contour_x = np.hstack((region_contour_x, region_contour_x[::-1]))
            if box0[0] != 0:
                box0[0] -= 1
                box0[2] += 1
                if box0[2] != img_height:
                    region_contour_y = np.hstack((region_contour_y - 1, region_contour_y + 1))
                else:
                    region_contour_y = np.hstack((region_contour_y - 1, region_contour_y))
            else:
                box0[2] += 2
                region_contour_y = np.hstack((region_contour_y, region_contour_y + 1))

        if box0[1] == box0[3]:
            region_contour_y = np.hstack((region_contour_y, region_contour_y[::-1]))
            if box0[1] != 0:
                box0[1] -= 1
                box0[3] += 1
                if box0[3] != img_width:
                    region_contour_x = np.hstack((region_contour_x - 1, region_contour_x + 1))
                else:
                    region_contour_x = np.hstack((region_contour_x - 1, region_contour_x))
            else:
                box0[3] += 2
                region_contour_x = np.hstack((region_contour_x, region_contour_x + 1))

    """
    特殊情况2：
    标注的坐标成两行或两列，如
    [500, 736, 501, 738]
    将其改为
    [500, 736, 502, 738]
    """
    if box0[2] - box0[0] == 1 or box0[3] - box0[1] == 1:
        if box0[2] - box0[0] == 1:
            box0[2] += 1
        if box0[3] - box0[1] == 1:
            box0[3] += 1

    # 计算轮廓的凸包
    points = np.dstack((region_contour_x, region_contour_y))
    hull = cv2.convexHull(points)
    region_contour_x = np.array([i[0][0] for i in hull])
    region_contour_y = np.array([i[0][1] for i in hull])
    # 轮廓的点集
    crop_contour = np.dstack((region_contour_x - box0[1], region_contour_y - box0[0]))
    crop_img = image[box0[0]:box0[2], box0[1]:box0[3]]

    return crop_img, crop_contour


def mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support in-contiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [x + 0.5 for x in res if len(x) >= 6]
    return res, has_holes


if __name__ == '__main__':
    print(get_feature_map())

    # import json
    # import os
    # from tqdm import tqdm
    # jf = r"C:\Users\lvyong\Desktop\030817\batch_341.json"
    # img_folder = r"C:\Users\lvyong\Desktop\030817\image\341"
    # wf = [3, 15, 1, 14, 3, 2, 17, 99]
    #
    # feature_method = get_feature(wf)
    # print(get_feature_map(feature_method))
    #
    # ann = json.load(open(jf))
    # feature_pro = tqdm(ann.items(), desc='提取特征')
    # for k, v in feature_pro:
    #     src_img = cv2.imdecode(np.fromfile(os.path.join(img_folder, k), dtype=np.uint8), 0)
    #     print(feature_method.get_image_feature(src_img, v['regions']))
    #     break
