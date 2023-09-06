import cv2
import os
import json
import warnings
import numpy as np
from typing import List, Callable
from glob import glob
from evalmodel.data_info import ImageInfo, RegionInfo
from evalmodel.utils.comm import create_dir


__all__ = ['common_filter', 'common_view', 'hg_product_filter']


def hg_product_filter(image_info: ImageInfo, gray=10, area=20) -> List[RegionInfo]:
    """
        测试代码

    :param image_info:
    :param gray:
    :param area:
    :return:
    """

    print(f'{image_info.full_name}, gray: {gray}, area: {area}')
    return [RegionInfo.from_LTRB(0, 0, 4096, 4096)]


def common_filter(image_info: ImageInfo, cfg: str, key_parse: Callable[[str], str] = None) -> List[RegionInfo]:
    """
        通用过滤接口，通过与测试图匹配的过滤配置解析进行过滤

    :param image_info: 当前图像信息
    :param cfg: 过滤配置文件路径
    :param key_parse: 图像与过滤信息匹配的映射关系
    :return: 过滤信息
    """

    if not os.path.exists(cfg):
        raise ValueError('过滤配置文件不存在')

    if key_parse and not callable(key_parse):
        raise ValueError(f'key_parse must be callable')

    cfg_data = json.load(open(cfg))
    if key_parse:
        img_key = key_parse(image_info.name)
    else:
        img_key = image_info.name

    # 判断图像是否与配置文件匹配
    filter_boxes = []
    if img_key in cfg_data:
        for region in cfg_data[img_key]['regions']:
            mask_type = int(region['region_attributes']['regions']) == 2
            ptx = region['shape_attributes']['all_points_x']
            pty = region['shape_attributes']['all_points_y']
            filter_boxes.append(RegionInfo(ptx, pty, mask_zone=mask_type))

    if not len(filter_boxes):
        warnings.warn(UserWarning("未在软件过滤配置文件中匹配到图像信息"))
    return filter_boxes


def common_view(image_folder, cfg: str, key_parse: Callable[[str], str] = None):
    """
        查看过滤框

    :param image_folder: 图像文件夹
    :param cfg: 过滤配置信息
    :param key_parse: 图像与过滤信息匹配的映射关系
    :return: None
    """

    if key_parse and not callable(key_parse):
        raise ValueError(f'key_parse must be callable')

    cfg_data = json.load(open(cfg))
    save_folder = create_dir(os.path.join(image_folder, 'box_view'))
    for img_file in glob(os.path.join(image_folder, '*.[bj][mp][pg]')):
        img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
        img_name = os.path.basename(img_file)
        name = os.path.splitext(img_name)[0]
        if key_parse:
            img_key = key_parse(name)
        else:
            img_key = img_name

        if img_key in cfg_data:
            for region in cfg_data[img_key]['regions']:
                mask_type = int(region['region_attributes']['regions']) == 2
                ptx = region['shape_attributes']['all_points_x']
                pty = region['shape_attributes']['all_points_y']
                if mask_type:
                    color = (0, 0, 255)  # 屏蔽区域，红色
                else:
                    color = (255, 0, 0)  # 检测区域，蓝色

                points = np.dstack((ptx, pty))
                cv2.polylines(img, pts=[points], isClosed=True, color=color, thickness=1)

            cv2.imencode('.bmp', img)[1].tofile(os.path.join(save_folder, f'{name}.bmp'))
