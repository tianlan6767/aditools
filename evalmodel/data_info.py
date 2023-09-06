import os
import logging
import warnings

import cv2
import numpy as np

import time
from copy import deepcopy
from warnings import warn
from typing import List, Union, Callable, Tuple, Dict, Any
from PIL import Image as PImage
from skimage.draw import polygon
from collections import defaultdict
from evalmodel.utils.comm import create_dir, sort_dict
from evalmodel.utils.visualizer import draw_via_region, get_via_region_boundary

__all__ = ["RegionInfo", "ImageInfo", 'ModelInfo']
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class RegionInfo(object):
    """区域描述信息"""

    def __init__(self, xs: List[int], ys: List[int], area: int = 0, iou: float = 0.0, mask_zone=False):
        """
            通过点集构造对象

        :param xs: 点集x坐标
        :param ys: 点集y坐标
        :param area: 面积
        :param iou: IOU
        :param mask_zone: 区域的排斥性，True表示掩模区，False表示检测区
        """

        self.pts_x = xs
        self.pts_y = ys
        assert len(xs) == len(ys) and len(xs) >= 3, f'输入轮廓点信息异常，轮廓点个数{len(ys)}'
        self.score = 0.0
        self._label = "1"
        self.__is_valid: bool = False
        self.__len = len(self.pts_x)
        self.__area = area
        self.iou: float = iou
        self.json_info = {}
        self.IR = None  # 最大相交区域
        self.is_mark_region = True
        self.idx = 0  # 标识，根据标注/推理顺序赋值，由区域性质决定

        self._left = min(self.pts_x)
        self._top = min(self.pts_y)
        self._right = max(self.pts_x)
        self._bottom = max(self.pts_y)
        self.is_mask = mask_zone  # 表示区域是掩模区域还是检测区域
        self.tag = None

    @classmethod
    def from_json(cls, info: dict, is_mark: bool, area: int = 0, iou: float = 0.0) -> 'RegionInfo':
        """
            通过json数据构造对象

        :param info: 缺陷信息
        :param is_mark: 标注信息还是推理信息
        :param area: 缺陷面积，缺省时自动计算
        :param iou: 相交IOU
        :return: RegionInfo
        """

        pts_x = info['shape_attributes']['all_points_x']
        pts_y = info['shape_attributes']['all_points_y']
        if len(pts_y) == 0:
            reg = cls.empty()
        else:
            reg = cls(pts_x, pts_y, area, iou)
            if 'regions' in info['region_attributes']:
                reg.label = info['region_attributes']['regions']
            if 'score' in info['region_attributes']:
                reg.score = float(info['region_attributes']['score'])
            reg.json_info = info
        reg.is_mark_region = is_mark
        return reg

    @classmethod
    def from_LTRB(cls, left, top, right, bottom, mask_zone=False) -> 'RegionInfo':
        """
            通过左上角和右下角点构建对象

        :param left: 左上角x坐标
        :param top: 左上角y坐标
        :param right: 右下角x坐标
        :param bottom: 右下角y坐标
        :param mask_zone: 区域的排斥性，True表示掩模区，False表示检测区
        :return: RegionInfo
        """

        xs = [left, right, right, left]
        ys = [top, top, bottom, bottom]
        return cls(xs, ys, mask_zone=mask_zone)

    @property
    def label(self) -> str:
        """区域标签"""
        return self._label

    @label.setter
    def label(self, label: str):
        """区域标签"""
        self._label = label
        if len(self.json_info):
            self.json_info['region_attributes']['regions'] = label

    @classmethod
    def empty(cls) -> 'RegionInfo':
        """构造空区域"""
        return cls.from_LTRB(0, 0, 0, 0)

    @property
    def area(self) -> int:
        """区域面积"""
        if self.__area == 0:
            if len(self.pts_x) >= 3 and not self.is_empty:
                region_points = [(int(self.pts_x[i]), int(self.pts_y[i])) for i in range(self.__len)]
                mark_rect = cv2.boundingRect(np.array(region_points))
                shape = (mark_rect[3], mark_rect[2])
                img = np.zeros(shape, dtype=np.uint8)
                cv2.fillPoly(img, [np.array(region_points) - (mark_rect[0], mark_rect[1])], color=255)
                self.__area = cv2.countNonZero(img)
            else:
                self.__area = 0
        return self.__area

    @property
    def box(self) -> Tuple[int, int, int, int]:
        """
            区域外接正矩形

        :return: (left, top, right, bottom)
        """

        return self.left, self.top, self.right, self.bottom

    @property
    def left(self):
        return self._left

    @property
    def top(self):
        return self._top

    @property
    def right(self):
        return self._right

    @property
    def bottom(self):
        return self._bottom

    @property
    def is_empty(self):
        """是否是空框"""
        return self.width <= 0 or self.height <= 0

    @property
    def width(self):
        """宽度"""
        return self.right - self.left

    @property
    def height(self):
        """高度"""
        return self.bottom - self.top

    def get_feature(self, key: str) -> Any:
        """
            获取指定特征信息

        :param key: 特征名称
        :return: 特征值，当特征不存在时，返回''
        """

        if len(self.json_info) > 0 and 'region_attributes' in self.json_info:
            if key in self.json_info['region_attributes']:
                return self.json_info['region_attributes'][key]
            else:
                warn(f'There is no feature {key}', UserWarning)
        return ''

    def is_valid(self, min_iou: float = 0.1) -> bool:
        """
            是否有效

            对于标注区域，当与推理的IOU小于设定值是时，表示区域为漏检区域\r\n
            对于推理区域，当与标注的IOU小于设定值是时，表示区域为过检区域
        :param min_iou: 最小有效IOU阈值
        :return: True 有效区域
        """

        return self.iou >= min_iou

    def reset_iou(self) -> None:
        """重置区域的IOU信息"""

        self.iou = 0
        self.IR = None

    def update_iou(self, other: 'RegionInfo') -> None:
        """
            更新计算IOU

            更新前请使用 reset_iou() 进行属性重置！！！
            仅标注区域调用，更新标注区域的IOU时，同时更新推理区域的IOU
        :param other: 其他区域
        :return:
        """

        if not self.is_mark_region:
            return

        # 计算IOU
        iou = self.cal_iou2(other)
        # iou = self.cal_iou1(other)

        # 更新IOU
        if iou > 0:
            other.reset_iou()
            other.iou = iou
            other.IR = self
            # if iou > self.iou:
            if other.score > self.score:
                self.iou = iou
                self.IR = other
                self.score = other.score

    def cal_iou1(self, other: 'RegionInfo') -> float:
        """
            计算输入区域间的IOU，不推荐使用，耗时久

        :param other: 推理区域
        :return: IOU
        """

        if not isinstance(other, RegionInfo):
            raise ValueError('other must be RegionInfo object!')

        if not self.intersect_with(other, simple=True):
            return 0.0

        start = time.perf_counter()
        # 计算IOU
        r1, c1 = polygon(np.asarray(self.pts_y), np.asarray(self.pts_x))
        r2, c2 = polygon(np.asarray(other.pts_y), np.asarray(other.pts_x))
        try:
            r_max = max(r1.max(), r2.max()) + 1
            c_max = max(c1.max(), c2.max()) + 1
        except ValueError:
            return 0.0

        canvas = np.zeros((r_max, c_max))
        canvas[r1, c1] += 1
        canvas[r2, c2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0.0
        intersection = np.sum(canvas == 2)
        iou = intersection / union
        coast = time.perf_counter()
        print(f'Elapsed Time1 of Calculating IOU-[{iou:.4f}]: {coast - start:^7.3f} s.')

        return iou

    def cal_iou2(self, other: 'RegionInfo') -> float:
        """
            计算输入区域间的IOU

        :param other: 推理区域
        :return: IOU
        """

        if not isinstance(other, RegionInfo):
            raise ValueError('other must be RegionInfo object!')

        if not self.intersect_with(other, simple=True):
            return 0.0

        start = time.perf_counter()
        mark_region = [(int(self.pts_x[i]), int(self.pts_y[i])) for i in range(self.__len)]
        inf_region = [(int(other.pts_x[i]), int(other.pts_y[i])) for i in range(len(other.pts_x))]
        all_region = []
        all_region.extend(mark_region)
        all_region.extend(inf_region)

        # 计算合适的背景图大小
        mark_rect = cv2.boundingRect(np.array(all_region))
        shape = (mark_rect[3], mark_rect[2])
        mark_img = np.zeros(shape, dtype=np.uint8)  # shape = (row, col)
        inf_img = np.zeros(shape, dtype=np.uint8)

        # 生成推理和标注图
        if isinstance(mark_region[0], list):
            for region in mark_region:
                cv2.fillPoly(mark_img, [np.array(region) - (mark_rect[0], mark_rect[1])], color=255)
        else:
            cv2.fillPoly(mark_img, [np.array(mark_region) - (mark_rect[0], mark_rect[1])], color=255)
        if isinstance(inf_region[0], list):
            for region in inf_region:
                cv2.fillPoly(inf_img, [np.array(region) - (mark_rect[0], mark_rect[1])], color=255)
        else:
            cv2.fillPoly(inf_img, [np.array(inf_region) - (mark_rect[0], mark_rect[1])], color=255)

        # 计算交集和并集
        dst_img1 = np.zeros(shape, dtype=np.uint8)
        dst_img2 = np.zeros(shape, dtype=np.uint8)
        and_img = cv2.bitwise_and(mark_img, inf_img, dst_img1)
        or_img = cv2.bitwise_or(mark_img, inf_img, dst_img2)
        iou = (cv2.countNonZero(and_img)) / ((cv2.countNonZero(or_img)) + 1)

        # 计算面积
        if self.__area == 0:
            self.__area = cv2.countNonZero(mark_img)

        coast = time.perf_counter()
        logging.debug(f'Elapsed Time2 of Calculating IOU-[{iou:.4f}]: {coast - start:^7.3f} s.')

        return iou

    def cal_intersection(self, other: 'RegionInfo') -> float:
        """
            计算与输入区域间的交集占比

        :param other: 其他区域
        :return: 交集区域占自身区域的比值
        """

        if not isinstance(other, RegionInfo):
            raise ValueError('other must be RegionInfo object!')

        if not self.intersect_with(other, simple=True):
            return 0.0

        start = time.perf_counter()
        mark_region = [(int(self.pts_x[i]), int(self.pts_y[i])) for i in range(self.__len)]
        inf_region = [(int(other.pts_x[i]), int(other.pts_y[i])) for i in range(len(other.pts_x))]
        all_region = []
        all_region.extend(mark_region)
        all_region.extend(inf_region)

        # 计算合适的背景图大小
        mark_rect = cv2.boundingRect(np.array(all_region))
        shape = (mark_rect[3], mark_rect[2])
        mark_img = np.zeros(shape, dtype=np.uint8)  # shape = (row, col)
        inf_img = np.zeros(shape, dtype=np.uint8)

        # 生成推理和标注图
        if isinstance(mark_region[0], list):
            for region in mark_region:
                cv2.fillPoly(mark_img, [np.array(region) - (mark_rect[0], mark_rect[1])], color=255)
        else:
            cv2.fillPoly(mark_img, [np.array(mark_region) - (mark_rect[0], mark_rect[1])], color=255)
        if isinstance(inf_region[0], list):
            for region in inf_region:
                cv2.fillPoly(inf_img, [np.array(region) - (mark_rect[0], mark_rect[1])], color=255)
        else:
            cv2.fillPoly(inf_img, [np.array(inf_region) - (mark_rect[0], mark_rect[1])], color=255)

        # 计算交集
        dst_img1 = np.zeros(shape, dtype=np.uint8)
        and_img = cv2.bitwise_and(mark_img, inf_img, dst_img1)
        i = (cv2.countNonZero(and_img)) / ((cv2.countNonZero(mark_img)) + 1)

        # 计算面积
        if self.__area == 0:
            self.__area = cv2.countNonZero(mark_img)

        coast = time.perf_counter()
        logging.debug(f'Elapsed Time2 of Calculating I-[{i:.4f}]: {coast - start:^7.3f} s.')

        return i

    def contains(self, x, y) -> bool:
        """判断是否包含点"""
        if self.left < x < self.right and self.top < y < self.bottom:

            def is_cross(poi, s_poi, e_poi):
                # 排除不相交场景
                if s_poi[1] == e_poi[1]:  # 与射线平行、重合，线段首尾端点重合的情况
                    return False
                if s_poi[1] > poi[1] and e_poi[1] > poi[1]:  # 线段在射线下边
                    return False
                if s_poi[1] < poi[1] and e_poi[1] < poi[1]:  # 线段在射线上边
                    return False
                if s_poi[1] == poi[1] and e_poi[1] > poi[1]:  # 交点为上端点，对应start_point
                    return False
                if e_poi[1] == poi[1] and s_poi[1] > poi[1]:  # 交点为上端点，对应end_point
                    return False
                if s_poi[0] < poi[0] and e_poi[1] < poi[1]:  # 线段在射线左边
                    return False

                # 求交
                x_cross = e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1])
                # 交点在射线起点的右侧
                return x_cross >= poi[0]

            cross_num = 0
            for i in range(self.__len - 1):
                start_poi = (self.pts_x[i], self.pts_y[i])
                end_poi = (self.pts_x[i + 1], self.pts_y[i + 1])
                if is_cross((x, y), start_poi, end_poi):
                    cross_num += 1
            return True if cross_num % 2 == 1 else False
        return False

    def intersect_with(self, other: Union[List['RegionInfo'], 'RegionInfo'], simple=False) -> bool:
        """
            与其他区域是否相交

        :param other: 其他框
        :param simple: 是否仅利用外接矩形判断即可
        :return: True: 相交
        """

        if other is None or self.is_empty:
            return False

        if not isinstance(other, list):
            other = [other]
        for box in other:
            if not box.is_empty and \
                    max(self.left, box.left) < min(self.right, box.right) and \
                    max(self.top, box.top) < min(self.bottom, box.bottom):
                if simple:
                    return True
                for idx, x in enumerate(self.pts_x):
                    if box.contains(x, self.pts_y[idx]):
                        return True
        return False

    def is_retained(self, boxes: List['RegionInfo']) -> bool:
        """
            根据过滤区域，判断是否需要保留该区域

        :param boxes: 过滤框
        :return: True: 保留
        """

        if boxes is None:
            return True

        in_boxes = [box for box in boxes if not box.is_mask]
        ex_boxes = [box for box in boxes if box.is_mask]

        # if ex_boxes and self.intersect_with(ex_boxes):
        #     return False
        # if in_boxes and not self.intersect_with(in_boxes):
        #     return False

        if ex_boxes:
            for box in ex_boxes:
                if self.cal_intersection(box) >= 0.4:
                    return False
        if in_boxes:
            for box in in_boxes:
                if self.cal_intersection(box) >= 0.4:
                    return True
            return False

        return True


class ImageInfo(object):
    """图像描述信息"""

    def __init__(self, image_folder, model_key: str = '') -> None:
        """
            构造函数

        :param image_folder: 图像所在文件夹
        :param model_key: 关联的模型标识
        """

        self.full_name: str = ''
        self.img_path: str = ''
        self.img_folder = image_folder
        self.model_key = model_key if model_key else 'Unknown'
        self.mark_regions: List[RegionInfo] = []
        self.inf_regions: List[RegionInfo] = []
        self.is_tested = False

    def update_region_info(self, infos: dict,
                           gen_box_func: Callable[["ImageInfo", Dict], List[RegionInfo]] = None, **kwargs) -> None:
        """
            通过json信息，更新区域信息

        :param infos: 区域的json数据
        :param gen_box_func: 生成目标框函数, (ImageInfo, Dict) -> List[RegionInfo]
        :param kwargs: 生成目标框函数的参数列表
        :return: None
        """

        if gen_box_func and not callable(gen_box_func):
            raise ValueError(f'gen_box_func must be callable')
        else:
            boxes = gen_box_func(self, **kwargs) if gen_box_func else None

        file_name = infos['filename']
        regions = infos['regions']
        region_type = infos['type'] if 'type' in infos else 'mark'

        if not self.full_name:
            self.full_name = file_name
            self.img_path = os.path.join(self.img_folder, file_name)
        elif self.full_name != file_name:
            return

        self.is_tested = region_type == 'inf'

        # 初始化区域信息
        is_first = True
        for idx, reg in enumerate(regions):
            reg_info = RegionInfo.from_json(reg, is_mark=region_type != 'inf')
            if not reg_info.is_retained(boxes):
                continue

            reg_info.idx = idx
            if reg_info.is_mark_region:
                if is_first:
                    self.mark_regions.clear()
                self.mark_regions.append(reg_info)
            else:
                if is_first:
                    self.inf_regions.clear()
                self.inf_regions.append(reg_info)
            is_first = False

        # 更新区域IOU
        self.update_iou()

    def update_iou(self):
        """更新图像所有区域的IOU"""
        for inf_region in self.inf_regions:
            inf_region.reset_iou()
        for region in self.mark_regions:
            region.reset_iou()
            region.score = 0
            for inf_region in self.inf_regions:
                region.update_iou(inf_region)

    @property
    def name(self) -> str:
        """图像名称"""

        return os.path.splitext(self.full_name)[0] if self.full_name else ''

    @property
    def exists(self) -> bool:
        """
            图像是否存在

        :return: True 存在
        """

        return os.path.exists(self.img_path)

    @property
    def is_ok_image(self) -> bool:
        """
            是否是OK图

        :return: True OK图
        """

        return len(self.mark_regions) == 0

    @property
    def is_over_image(self) -> bool:
        """
            是否是过检图，仅OK图的过检情况

        :return: True 过检图
        """

        return self.is_ok_image and len(self.inf_regions) != 0

    def is_miss_image(self, min_iou=0.0) -> bool:
        """
            是否是漏检图像

            只要检出标注的缺陷就不算是漏检图
        :param min_iou: 标注与推理的最小IOU阈值，当设定为0时，不排除蒙的情况
        :return: True 漏检图
        """

        if len(self.mark_regions) != 0 and len(self.inf_regions) == 0:
            return True
        for region in self.inf_regions:
            if region.iou >= min_iou:
                return False
        return len(self.mark_regions) != 0

    def is_good_image(self, min_iou=0.0) -> bool:
        """
            是否是检出图像

            只要检出标注的缺陷就算是检出图
        :param min_iou: 标注与推理的最小IOU阈值，当设定为0时，不排除蒙的情况
        :return: True 漏检图
        """

        if self.is_miss_image(min_iou):
            return False
        for region in self.inf_regions:
            if region.iou >= min_iou:
                return True
        return len(self.inf_regions) == 0

    def get_miss_regions(self, min_iou) -> List[RegionInfo]:
        """
            获取标注漏检区域信息

        :param min_iou: 标注与推理的最小IOU阈值
        :return: 标注漏检区域信息
        """

        return [region for region in self.mark_regions if not region.is_valid(min_iou)]

    def get_over_regions(self, min_iou) -> List[RegionInfo]:
        """
            获取推理过检区域信息

        :param min_iou: 推理与标注的最小IOU阈值
        :return: 推理过检区域信息
        """

        return [region for region in self.inf_regions if not region.is_valid(min_iou)]

    def get_check_regions(self, min_iou, region_type='mark') -> List[RegionInfo]:
        """
            获取检出区域的信息

        :param min_iou: 推理与标注的最小IOU阈值
        :param region_type: 获取区域类型，['mark', 'inf']
        :return: 检出区域信息
        """

        assert region_type in ['mark', 'inf'], "region_type can set 'mark' or 'inf'"
        check_regions = []
        for region in self.mark_regions if region_type == 'mark' else self.inf_regions:
            if region.is_valid(min_iou):
                check_regions.append(region)
        return check_regions

    def get_feature_regions(self, feature_name, min_iou, min_feature, max_feature) -> List[RegionInfo]:
        """
            获取特征(min_feature, max_feature)区间内的区域

            注意：特征值必须是数值类型，才有意义

        :param feature_name: 特征名称
        :param min_iou: iou阈值
        :param min_feature: 最小特征值
        :param max_feature: 最大特征值
        :return: 得分满足设定的推理区域
        """

        assert min_feature < max_feature, f'min:{min_feature}, max:{max_feature}'
        if feature_name == 'score' or feature_name == 'inf-score':
            return [region for region in self.inf_regions if min_feature < region.score < max_feature]
        elif feature_name == 'check-score':
            return [region for region in self.get_check_regions(min_iou, 'inf')
                    if min_feature < region.score < max_feature]
        elif feature_name == 'over-score':
            return [region for region in self.get_over_regions(min_iou) if min_feature < region.score < max_feature]
        elif feature_name == 'iou':
            return [region for region in self.inf_regions if min_feature < region.iou < max_feature]
        else:
            return [region for region in self.inf_regions
                    if min_feature < region.get_feature(feature_name) < max_feature]

    def draw_regions(self, save_folder, region_type='mark', compared=False, draw_label=True, draw_single=False,
                     color=(255, 0, 0), thickness=1, save_format='.bmp', min_iou: float = 0.1, **kwargs) -> str:
        """
            绘制区域


        :param save_folder: 保存路径
        :param region_type: 绘制区域类型，['miss', 'over', 'check', 'mark', 'inf', 'all',
                                        'iou', 'score', 'check-score', 'over-score', 'inf-score']
        :param compared: 输出时，是否左侧添加对比图像
        :param draw_label: 是否绘制标签
        :param draw_single: 是否将每个缺陷作为独立个体绘制在图像上, False:将所有区域绘制在同一张图上, True:将每个区域绘制在独自图像上
        :param color: 区域绘制颜色, region_type='all' 或 'check' 时，指的推理的区域颜色
        :param thickness: 绘制区域边界线条宽度，≥1
        :param save_format: 保存格式，查看局部图时，建议为.bmp格式；查看整图时，建议.jpg格式
        :param min_iou: 推理与标注的最小IOU阈值
        :param kwargs: 其他控制参数
        :return: 绘制图像保存路径
        """

        assert region_type in ['miss', 'over', 'check', 'mark', 'inf', 'all',
                               'iou', 'score', 'check-score', 'over-score', 'inf-score'], region_type
        thickness = thickness if thickness >= 1 else 1
        save_folder = create_dir(os.path.join(save_folder, region_type))
        color1 = (255 - color[0], 255 - color[1], 255 - color[2])
        # 输入变量处理
        if region_type == 'all':
            draw_single = False
            if len(self.inf_regions) == 0:
                region_type = 'mark'
                color = color1
        save_folder = create_dir(os.path.join(save_folder, 'thumb' if draw_single else 'large'))

        if not self.exists:
            return save_folder

        if region_type in ['mark', 'inf', 'all']:
            regions = self.mark_regions if region_type == 'mark' else self.inf_regions
        elif region_type == 'miss':
            regions = self.get_miss_regions(min_iou)
        elif region_type == 'over':
            regions = self.get_over_regions(min_iou)
        elif region_type == 'check':
            regions = self.get_check_regions(min_iou, 'inf')
        else:
            assert 'min_feature' in kwargs and 'max_feature' in kwargs, kwargs
            regions = self.get_feature_regions(region_type, min_iou, kwargs['min_feature'], kwargs['max_feature'])

        self.draw(save_folder, regions, region_type, compared, draw_label, draw_single, color, color1, thickness,
                  save_format)

        return save_folder

    def draw(self, save_folder, regions: List[RegionInfo], region_type, compared=False, draw_label=True, draw_single=False,
             color=(0, 0, 255), color1=None, thickness=1, save_format='.bmp'):
        """

        :param regions:
        :param region_type: ['score', 'iou', ...]
        :param save_folder:
        :param compared:
        :param draw_label:
        :param draw_single:
        :param color:
        :param color1:
        :param thickness:
        :param save_format:
        :return:
        """

        src_img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        comparison = src_img.copy() if compared and not draw_single else None
        img_size = src_img.shape[:2]
        color1 = (255 - color[0], 255 - color[1], 255 - color[2]) if color1 is None else color1
        draw_score = region_type in ['check', 'iou', 'inf', 'over', 'score',
                                     'check-score', 'over-score', 'inf-score', 'all']

        for idx, region in enumerate(regions):
            if len(region.json_info) == 0:
                continue

            if draw_single:
                drawn_img = draw_via_region(src_img, region.json_info, color, thickness, draw_label,
                                            draw_score, inplace=False)
                if region_type in ['check', 'iou', 'score', 'check-score', 'over-score', 'inf-score'] and region.IR:
                    draw_via_region(drawn_img, region.IR.json_info, color1, thickness, draw_label)

                # 每个缺陷作为独立结果输出
                box, anchor = get_via_region_boundary(region.json_info, 50, img_size)
                df_mrk_img = drawn_img[box[1]:box[3], box[0]:box[2]]
                save_temp = create_dir(os.path.join(save_folder, region.label))
                save_file = os.path.join(save_temp, f'{os.path.splitext(self.full_name)[0]}_{idx}{save_format}')
                if compared:
                    border = 2
                    bg_img = PImage.new('RGB',
                                        ((box[2] - box[0]) * 2 + border * 3, (box[3] - box[1]) + border * 2),
                                        "white")
                    df_cmp_img = src_img[box[1]:box[3], box[0]:box[2]]
                    bg_img.paste(PImage.fromarray(df_cmp_img), (border, border))
                    bg_img.paste(PImage.fromarray(df_mrk_img), (box[2] - box[0] + border * 2, border))
                    bg_img = PImage.fromarray(cv2.cvtColor(np.asarray(bg_img), cv2.COLOR_RGB2BGR))
                    bg_img.save(save_file)
                else:
                    cv2.imencode(save_format, df_mrk_img)[1].tofile(save_file)
            else:
                if region_type == 'all' and idx == 0:
                    for reg in self.mark_regions:
                        draw_via_region(src_img, reg.json_info, color1, thickness, draw_label)

                draw_via_region(src_img, region.json_info, color, thickness, draw_label, draw_score=draw_score)

                # 绘制所有缺陷作为输出
                if idx + 1 == len(regions):
                    save_file = os.path.join(save_folder, os.path.splitext(self.full_name)[0] + save_format)
                    if compared:
                        border = 2
                        bg_img = PImage.new('RGB', (img_size[1] * 2 + border * 3, img_size[0] + border * 2),
                                            "#FFFFFF")
                        bg_img.paste(PImage.fromarray(comparison), (border, border))
                        bg_img.paste(PImage.fromarray(src_img), (img_size[1] + border * 2, border))
                        bg_img = PImage.fromarray(cv2.cvtColor(np.asarray(bg_img), cv2.COLOR_RGB2BGR))
                        bg_img.save(save_file)
                    else:
                        cv2.imencode(save_format, src_img)[1].tofile(save_file)

    def __str__(self):
        return self.full_name

    def get_json(self, info_type: str, min_iou: float = 0.0001) -> dict:
        """
            获取指定类型的区域信息

        :param info_type: 标注信息类型 ['mrk_check', 'inf_check', 'miss', 'over', 'inf_all', 'mark_all']
        :param min_iou: 标注与推理的最小IOU阈值
        :return: dict
        """
        assert info_type in ['mrk_check', 'inf_check', 'miss', 'over', 'inf_all', 'mark_all'], \
            f"info_type: {info_type} is wrong!"
        if info_type == 'miss':
            regions = self.get_miss_regions(min_iou)
        elif info_type == 'over':
            regions = self.get_over_regions(min_iou)
        elif info_type == 'inf_all':
            regions = self.inf_regions
        elif info_type == 'mark_all':
            regions = self.mark_regions
        else:
            regions = self.get_check_regions(min_iou, 'mark' if info_type == 'mrk_check' else 'inf')
        region_type = 'inf 'if info_type in ['inf_check', 'over', 'inf_all'] else 'mark'
        json_info = dict(filename=self.full_name, regions=[], type=region_type)
        for region in regions:
            json_info['regions'].append(region.json_info)
        return json_info


class ModelInfo(object):
    """模型表述信息"""

    def __init__(self, model_key: str, image_infos: List[ImageInfo]):
        self.model_key = model_key
        self.total_img_infos: List[ImageInfo] = image_infos
        self.img_infos: List[ImageInfo] = []
        self.score = 0.0
        self.min_iou = 0.0
        self.__inf_num = 0
        self.__dft_num = 0
        self.__dft_chk_num = 0
        self.__dft_over_num = 0
        self.__img_chk_num = 0
        self.__img_over_num = 0
        self.__img_miss_num = 0
        self.__img_chk_rate = 0.0

        self.filter(0.4, 10)
        self.update(0.1)

    def filter(self, min_score, min_area, filter_boxes: List[RegionInfo] = None,
               special_scores: Dict[str, float] = None) -> None:
        """
            过滤推理结果

        :param min_score: 最小得分
        :param min_area: 最小面积
        :param filter_boxes: 过滤框。包含框，推理结果在框外就会过滤掉；排除框，推理结果在框内就会过滤掉
        :param special_scores: 特定类别的得分过滤阈值
        :return: None
        """

        self.img_infos.clear()
        for img_info in self.total_img_infos:
            # ignore which is undetected
            if not img_info.is_tested:
                warn('There are some Undetected images', UserWarning)
                continue

            new_image = deepcopy(img_info)
            remove_idxes = []
            for idx, inf_region in enumerate(new_image.inf_regions):
                if inf_region.area < min_area or inf_region.score < min_score or \
                        not inf_region.is_retained(filter_boxes):
                    remove_idxes.append(idx)
                elif special_scores is not None and inf_region.label in special_scores and \
                        inf_region.score < special_scores[inf_region.label]:
                    remove_idxes.append(idx)
            if len(remove_idxes):
                remove_idxes = sorted(remove_idxes, reverse=True)
                for idx in remove_idxes:
                    new_image.inf_regions.pop(idx)
                # 更新图像区域的IOU
                new_image.update_iou()

            self.img_infos.append(new_image)

    def update(self, min_iou) -> None:
        """
            更新模型评价指标

        :param min_iou: 标注与推理的最小IOU阈值
        :return: None
        """

        # 重置指标初始值
        for attr in list(self.__dict__):
            if attr.replace('_ModelInfo', '').startswith('__') and isinstance(getattr(self, attr), (int, float)):
                setattr(self, attr, 0)
        self.min_iou = min_iou
        self.score = 0.0

        for img_info in self.img_infos:
            self.__inf_num += len(img_info.inf_regions)
            self.__dft_num += len(img_info.mark_regions)
            self.__dft_chk_num += len(img_info.get_check_regions(min_iou))
            self.__dft_over_num += len(img_info.get_over_regions(min_iou))
            if img_info.is_over_image:
                self.__img_over_num += 1
            elif img_info.is_miss_image(min_iou):
                self.__img_miss_num += 1
            else:
                self.__img_chk_num += 1
        if len(self.img_infos):
            self.__img_chk_rate = self.__img_chk_num / len(self.img_infos)

    @classmethod
    def empty(cls) -> 'ModelInfo':
        """构建空模型"""
        return cls('empty_model', [])

    @property
    def is_empty(self):
        """判断是否是空模型"""
        return len(self.img_infos) == 0

    @property
    def inf_num(self) -> int:
        """推理缺陷数目"""
        return self.__inf_num

    @property
    def dft_num(self) -> int:
        """标注缺陷数目"""
        return self.__dft_num

    @property
    def dft_chk_num(self) -> int:
        """标注缺陷检出数目"""
        return self.__dft_chk_num

    @property
    def dft_chk_rate(self) -> int:
        """标注缺陷检出率"""
        return self.dft_chk_num / self.dft_num if self.dft_num else 0

    @property
    def dft_over_num(self) -> int:
        """推理缺陷过检数目"""
        return self.__dft_over_num

    @property
    def img_chk_num(self) -> float:
        """图像严格检出数目"""
        return self.__img_chk_num

    @property
    def img_chk_rate(self) -> float:
        """图像严格检出率"""
        return self.__img_chk_rate

    def get_img_num(self, img_type) -> int:
        """
            获取图像的数量

        :param img_type: 图像类型 ['ok', 'ng', 'both']
        :return:
        """

        assert img_type in ['ok', 'ng', 'both'], img_type
        if img_type == 'both':
            return len(self.img_infos)
        elif img_type == 'ok':
            return len([img for img in self.img_infos if img.is_ok_image])
        return len([img for img in self.img_infos if not img.is_ok_image])

    def __str__(self) -> str:
        return f'{self.model_key} ' \
               f'min_iou: {self.min_iou} ' \
               f'defect_num: {self.dft_num} ' \
               f'defect_check_num: {self.dft_chk_num} ' \
               f'defect_over_num: {self.dft_over_num} ' \
               f'image_check_rate: {self.img_chk_rate:.2%}' \
               f'score: {self.score:.3f}'

    def get_performance(self) -> dict:
        """获取模型的性能指标"""

        scores = self.get_scores('inf_check')
        chk_inf_mean_score = sum(scores) / len(scores) if len(scores) else 0
        scores = self.get_scores('mrk_check')
        chk_mrk_mean_score = sum(scores) / len(scores) if len(scores) else 0
        scores = self.get_scores('over')
        ovr_mean_score = sum(scores) / len(scores) if len(scores) else 0

        return dict(
            model_key=self.model_key,
            min_iou=self.min_iou,
            defect_num=self.dft_num,
            defect_check_num=self.dft_chk_num,
            defect_check_rate=round(self.dft_chk_rate, 4),
            defect_over_num=self.dft_over_num,
            mean_inf_chk_score=round(chk_inf_mean_score, 4),
            mean_mrk_chk_score=round(chk_mrk_mean_score, 4),
            mean_ovr_score=round(ovr_mean_score, 4),
            image_num=len(self.img_infos),
            image_check_num=self.__img_chk_num,
            image_check_rate=round(self.img_chk_rate, 4),
        )

    def get_scores(self, score_type: str):
        """
            获取模型得分

        :param score_type: 得分类型, ['check', 'mrk_check', 'inf_check', 'over', 'inf'], check和mrk_check等效
        :return:
        """

        assert score_type in ['inf_check', 'check', 'mrk_check', 'over', 'inf'], score_type
        scores = []
        for image_info in self.img_infos:
            if score_type in ['inf_check', 'inf']:
                for region_info in image_info.get_check_regions(self.min_iou, 'inf'):
                    scores.append(region_info.score)
            if score_type in ['over', 'inf']:
                for region_info in image_info.get_over_regions(self.min_iou):
                    scores.append(region_info.score)
            if score_type in ['check', 'mrk_check']:
                for region_info in image_info.get_check_regions(self.min_iou, 'mark'):
                    scores.append(region_info.score)
        return scores

    def get_images(self, image_type='miss') -> List[ImageInfo]:
        """
            获取图像

        :param image_type: 图像类型，['miss', 'over', 'check', 'all', 'abs-over']
        :return: 满足条件的图像
        """

        assert image_type in ['miss', 'over', 'check', 'all', 'abs-over'], image_type

        if image_type == 'all':
            return self.img_infos
        elif image_type == 'miss':
            return [img for img in self.img_infos if img.is_miss_image(self.min_iou)]
        elif image_type == 'check':
            return [img for img in self.img_infos if img.is_good_image(self.min_iou)]
        elif image_type == 'over':
            return [img for img in self.img_infos if img.is_over_image]
        else:
            return [img for img in self.img_infos if img.get_over_regions(self.min_iou)]

    def get_image_desc(self) -> dict:
        """获取模型图像检出的详情"""

        img_count = len(self.img_infos)
        ok_img_count = len([img for img in self.img_infos if img.is_ok_image])

        return dict(
            image_num=img_count,
            image_check_num=self.__img_chk_num,
            image_miss_num=self.__img_miss_num,
            image_over_num=self.__img_over_num,
            ok_image_num=ok_img_count,
            ok_image_check_num=ok_img_count - self.__img_over_num,
            ng_image_num=img_count - ok_img_count,
            ng_image_check_num=img_count - ok_img_count - self.__img_miss_num,
        )

    def get_defect_desc(self) -> dict:
        """获取缺陷检出的详情"""

        checked_detail, miss_detail, mark_detail = defaultdict(int), defaultdict(int), defaultdict(int)
        checked_inf_detail, over_inf_detail = defaultdict(int), defaultdict(int)
        over_ok_dft_num, over_ng_dft_num, over_ok_img_num, over_ng_img_num = 0, 0, 0, 0
        for image_info in self.img_infos:
            for region_info in image_info.get_check_regions(self.min_iou, 'mark'):
                label = int(region_info.label)
                checked_detail[label] += 1
                mark_detail[label] += 1
            for region_info in image_info.get_check_regions(self.min_iou, 'inf'):
                label = int(region_info.label)
                checked_inf_detail[label] += 1
            for region_info in image_info.get_miss_regions(self.min_iou):
                label = int(region_info.label)
                miss_detail[label] += 1
                mark_detail[label] += 1
            is_first = True
            for region_info in image_info.get_over_regions(self.min_iou):
                if image_info.is_ok_image:
                    over_ok_dft_num += 1
                    over_ok_img_num += 1 if is_first else 0
                else:
                    over_ng_dft_num += 1
                    over_ng_img_num += 1 if is_first else 0
                is_first = False
                label = int(region_info.label)
                over_inf_detail[label] += 1

        for k in mark_detail.keys():
            if k not in checked_detail:
                checked_detail[k] = 0
            if k not in miss_detail:
                miss_detail[k] = 0

        return dict(
            check_mrk=sort_dict(checked_detail),
            check_inf=sort_dict(checked_inf_detail),
            over_inf=sort_dict(over_inf_detail),
            miss=sort_dict(miss_detail),
            total=sort_dict(mark_detail),
            dft_over_ok_num=over_ok_dft_num,
            dft_over_ng_num=over_ng_dft_num,
            img_over_ok_num=over_ok_img_num,
            img_over_ng_num=over_ng_img_num,
        )
