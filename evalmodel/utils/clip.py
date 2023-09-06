#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：adi-evalmodel 
@File    ：clip.py
@Author  ：LvYong
@Date    ：2022/5/12 11:21 
"""
import shutil
import cv2
import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm
from glob import glob
from typing import Any, Callable, Dict, Tuple
from evalmodel.utils.comm import create_dir, save_json, get_file_infos, get_points_from_json
from PIL import Image as PImage



__all__ = ['clipping', 'splice_clipping','cliping_reverse','splice_clipping_reverse']


def __parse_software_roi(cfg: str):
    """
        解析软件屏蔽检测区的信息

    :param cfg: json配置文件路径
    :return: 解析信息
    """

    if not os.path.exists(cfg):
        raise ValueError('过滤配置文件不存在')

    new_data = {}
    cfg_data = json.load(open(cfg))

    # 判断是否为软件配置的json
    if 'all_image' not in cfg_data:
        return cfg_data

    # 解析软件ROI配置信息
    for img_cfg in cfg_data['all_image']:
        for key in img_cfg.keys():
            new_key = f'{key}.bmp'
            new_data[new_key] = {"filename": new_key, "regions": []}

            # 解析过滤信息
            for filter_info in img_cfg[key]:
                for info in filter_info.values():
                    mask_type, shield = False, {}
                    for f_info in info:
                        if 'regiontype' in f_info:
                            mask_type = f_info['regiontype'] == 0
                        elif 'shield_roi' in f_info:
                            shield = f_info['shield_roi']

                    if not shield:
                        continue

                    if shield['shape'] == 0:
                        ptx = [int(shield['c1']), int(shield['c2']), int(shield['c2']), int(shield['c1'])]
                        pty = [int(shield['r1']), int(shield['r1']), int(shield['r2']), int(shield['r2'])]
                    else:
                        ptx = [int(x) for x in shield['c1'].split(',')]
                        pty = [int(y) for y in shield['r1'].split(',')]

                    new_data[new_key]['regions'].append(
                        {"shape_attributes": {
                            "all_points_x": ptx,
                            "all_points_y": pty,
                        },
                            "region_attributes": {"regions": "2" if mask_type else "1"}
                        })

    return new_data


def __json2images(mrk_data, width, height):
    """
        将输入图像对应的标注json数据转换为掩模图（灰度值对应缺陷标签）

    :param mrk_data: 标注json数据
    :param width: 输入图像宽
    :param height: 输入图像高
    :return: 掩模图
    """

    mrk_images = []
    for region in mrk_data['regions']:
        label = int(region['region_attributes']['regions']) if 'regions' in region['region_attributes'] else 1
        pt = get_points_from_json(region)
        black_img = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(black_img, [np.array(pt)], color=label)
        mrk_images.append(black_img)

    return mrk_images if len(mrk_images) else [np.zeros((height, width), dtype=np.uint8)]


def __images2json(images, image_name, label_count=25, min_region=5) -> dict:
    """
        将掩模图转换为标注信息，掩模图的灰度值对应缺陷标签

    :param image: 原图
    :param image_name: 标注图
    :param label_count: 标签类别数目
    :param min_region: 最小标注面积
    :return: 标注json信息
    """

    images = [images] if not isinstance(images, list) else images
    mrk_data = {}
    content = dict(filename=image_name, regions=list())
    content["type"] = "inf"
    mrk_data[image_name] = content

    for image in images:
        if cv2.countNonZero(image) != 0:
            for label in range(1, label_count + 1):
                ret, binary = cv2.threshold(image, label, 255, cv2.THRESH_TOZERO_INV)
                ret, bin_image = cv2.threshold(binary, label - 1, 255, cv2.THRESH_BINARY)

                contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    region_area = cv2.contourArea(contour)
                    if region_area >= min_region:
                        region = dict(region_attributes={'regions': str(label), "score":0.5}) ###################################
                        new_contour = contour.squeeze()
                        region['shape_attributes'] = {
                            'all_points_x': [int(pt[0]) for pt in new_contour],
                            'all_points_y': [int(pt[1]) for pt in new_contour]
                        }
                        content['regions'].append(region)
            

    return mrk_data


def __sequential_clipping(src_img, mrk_images, mrk_data, dst_size) -> Any:
    """
        顺序均匀裁切图像

    :param src_img: 原图
    :param mrk_images: 标注图
    :param mrk_data: 标注信息
    :param dst_size: 裁切尺寸(row, col)
    :return: 裁切结果
    """

    img_height, img_width = src_img.shape[:2]
    image_name, image_ext = mrk_data['filename'].split('.')
    rows = img_height // dst_size[0]
    cols = img_width // dst_size[1]

    clip_images, clip_data = [], {}
    for row in range(rows):
        for col in range(cols):
            left, top = col * dst_size[1], row * dst_size[0]
            left = img_width - dst_size[1] if left + dst_size[1] > img_width else left
            top = img_height - dst_size[0] if top + dst_size[0] > img_height else top

            src_crop_img = src_img[top:top + dst_size[0], left:left + dst_size[1]]
            clip_images.append(src_crop_img)

            crop_img_name = f'{image_name}_{row * cols + col}.{image_ext}'
            mrk_crop_img = [mrk_img[top:top + dst_size[0], left:left + dst_size[1]] for mrk_img in mrk_images]
            clip_data.update(__images2json(mrk_crop_img, crop_img_name))

    return clip_images, clip_data


def splice_clipping(src_img, mrk_images, mrk_data, dst_size):
    """
        自定义图像裁剪，仅支持裁剪生成2048*2048的图像

    :param src_img:
    :param mrk_images:
    :param mrk_data:
    :param dst_size:
    :return:
    """
    assert dst_size == (2048, 2048)

    offset = 0
    ch_size = 2048 + offset
    image_name, image_ext = mrk_data['filename'].split('.')

    # 原图裁切
    img_1 = src_img[:ch_size, :ch_size, :]
    img_2 = src_img[:ch_size, (ch_size - 2 * offset):, :]

    img_3 = src_img[2048:, :ch_size, :]
    img_4 = src_img[2048:, (ch_size - 2 * offset):, :]

    bg_3 = np.zeros((ch_size, ch_size, 3), dtype=np.uint8)
    bg_3[:img_3.shape[0], :, :] = img_3
    bg_3[img_3.shape[0] + 72:img_3.shape[0] + 72 + img_4.shape[0], :, :] = img_4
    crop_images = [img_1, img_2, bg_3]

    # 掩模图裁切
    crop_img_masks = [[] for _ in range(len(crop_images))]
    for masks_bg in mrk_images:
        mask_1 = masks_bg[:ch_size, :ch_size]
        crop_img_masks[0].append(mask_1)

        mask_2 = masks_bg[:ch_size, (ch_size - 2 * offset):]
        crop_img_masks[1].append(mask_2)

        mask_3 = masks_bg[2048:, :ch_size]
        mask_4 = masks_bg[2048:, (ch_size - 2 * offset):]
        bg_mask_3 = np.zeros((ch_size, ch_size), dtype=np.uint8)
        bg_mask_3[:img_3.shape[0], :] = mask_3
        bg_mask_3[img_3.shape[0] + 72:img_3.shape[0] + 72 + img_4.shape[0], :] = mask_4
        crop_img_masks[2].append(bg_mask_3)

    # 掩模图转json
    crop_json = {}
    for idx, crop_img_mask in enumerate(crop_img_masks):
        image_key = f"{image_name}_{idx}.{image_ext}"
        crop_json.update(__images2json(crop_img_mask, image_key))

    return crop_images, crop_json



def splice_clipping_reverse(src_img, src_imn,  mrk_data, json_mark_size) -> Any:
    """
        自定义图像裁剪，仅支持裁剪生成2048*2048的图像

    :param src_img:
    :param mrk_images:
    :param mrk_data:
    :param dst_size:
    :return:
    """
    assert json_mark_size == (2048, 2048)
    
    height, width = src_img.shape[:2]
    split_ratio_x = int(width // json_mark_size[1])   # 2
    split_ratio_y = int(height // json_mark_size[0])  # 1
    if width % json_mark_size[1] == 0:
        size = width
    else:
        size = height

    split_ratio = split_ratio_x * split_ratio_y + 1
    
    mrk_img = PImage.fromarray(np.zeros((height, width), dtype=np.uint8))
    
    for key_split in range(split_ratio):
        
        crop_imn = src_imn[:-4] + f"_{key_split}" + osp.splitext(src_imn)[-1]
        if crop_imn in mrk_data:
            regions = mrk_data[crop_imn]["regions"]
        else:
            regions = []
        
        black_img = np.zeros((json_mark_size[1], json_mark_size[0]), dtype=np.uint8)
        for region in regions:
            label = int(region['region_attributes']['regions'])
            pt = get_points_from_json(region)
            cv2.fillPoly(black_img, [np.array(pt)], color=label)

        if key_split != split_ratio - 1:
            
            left =  int(int(key_split) % split_ratio_x * size / split_ratio_x)
            top = int(int(key_split) // split_ratio_x * size / split_ratio_y)
            mrk_img.paste(PImage.fromarray(black_img), (left, top))
            
            # print("3k第{}张裁剪", key_split, left, top)
        else:
            
            left =  int(int(key_split) % split_ratio_x * size / split_ratio_x)
            top = int(int(key_split) // split_ratio_x * size / (split_ratio_y + 1))
            
            
            mrk_img.paste(PImage.fromarray(black_img).crop((0, 0, json_mark_size[0], height - json_mark_size[1])), (left, top))
            # print("3k第三张裁剪", key_split, left, top)
            
            left =  int(int(key_split+1) % split_ratio_x * size / split_ratio_x)
            top = int(int(key_split+1) // split_ratio_x * size / (split_ratio_y+1))
            mrk_img.paste(PImage.fromarray(black_img).crop((0,  height - json_mark_size[1] + 72, json_mark_size[0], json_mark_size[1] - 1)), (left, top))
            # print("3k第三张裁剪", key_split, left, top)
    return np.array(mrk_img)


def __sequential_clipping_reverse(src_img, src_imn,  mrk_data, json_mark_size) -> Any:
    """
        顺序還原图像

    :param src_img: 原图
    :param mrk_images: 标注图
    :param mrk_data: 标注信息
    :param dst_size: 裁切尺寸(row, col)
    :return: 裁切结果
    """
    height, width = src_img.shape[:2]
    split_ratio_x = width // json_mark_size[0]
    split_ratio_y = height // json_mark_size[1]
    
    mrk_img = PImage.fromarray(np.zeros((height, width), dtype=np.uint8))
    # mrk_img = np.zeros(np.zeros((height, width),dtype=np.uint8))
    
    for key_split in range(split_ratio_x * split_ratio_y):
        # 还原位置信息计算
        crop_imn = src_imn[:-4] + f"_{key_split}" + osp.splitext(src_imn)[-1]
        if crop_imn in mrk_data:
            regions = mrk_data[crop_imn]['regions']
        else:
            regions = []
        # loc = str(row * cols + col)
        # left = col * json_mark_size[0]
        # top = row * json_mark_size[1]
        # left = width - json_mark_size[0] if left + json_mark_size[0] > raw_image.width else left
        # top = height - json_mark_size[1] if top + json_mark_size[1] > raw_image.height else top
        
        top = int(int(key_split) // split_ratio_x * height / split_ratio_y)
        left =  int(int(key_split) % split_ratio_x * width / split_ratio_x)
        # print("-----", key_split, left, top)
        
        black_img = np.zeros((json_mark_size[1], json_mark_size[0]), dtype=np.uint8)
        for region in regions:
            label = int(region['region_attributes']['regions'])
            pt = get_points_from_json(region)
            cv2.fillPoly(black_img, [np.array(pt)], color=label)
        mrk_img.paste(PImage.fromarray(black_img), (left, top))

    return np.array(mrk_img)



def clipping(org_images, org_json, dst_size=(2048, 2048), dst_channel=1, clip_type="ALL", save_image=True,
             special_clip: Callable[[Any, Any, Dict, Tuple[int, int]], Any] = None):
    """
        裁切图像

    :param org_images: 原图文件夹
    :param org_json: 原图标注信息
    :param dst_size: 裁切尺寸, (rows, cols)
    :param dst_channel: 裁切图通道数
    :param clip_type: 裁切类型 ['ALL', 'NG']
    :param save_image: 是否保存裁切图像
    :param special_clip: 自定义裁切操作。返回裁切小图和对应的标注信息，标注信息可能是标注图，也可能是标注json
    :return:
    """

    # 输入标注信息兼容性处理，解析来自软件的标注信息
    org_data = __parse_software_roi(org_json)

    result = {}
    save_folder = create_dir(os.path.join(org_images, 'clips'))

    # 图像裁剪处理
    image_files = glob(os.path.join(org_images, '*.bmp'))
    read_type = cv2.IMREAD_GRAYSCALE if dst_channel == 1 else cv2.IMREAD_COLOR
    pro_bar = tqdm(image_files, desc='☛clipping')
    for image_file in pro_bar:
        full_name, image_name, image_ext = get_file_infos(image_file)[1:]
        if full_name not in org_data:
            continue

        mrk_data = org_data[full_name]
        src_img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), read_type)
        src_height, src_width = src_img.shape[:2]
        mrk_images = __json2images(mrk_data, src_width, src_height)

        # 裁剪
        clip_method = __sequential_clipping
        if src_width % dst_size[1] or src_height % dst_size[0]:
            # Clip Size is Bad
            if special_clip is None or not callable(special_clip):
                raise ValueError('请自定义裁切函数')
            clip_method = special_clip

        clip_images, clip_data = clip_method(src_img, mrk_images, mrk_data, dst_size)

        # 结果处理
        result.update(clip_data)
        for idx, clip_name in enumerate(clip_data.keys()):
            if clip_type.upper() == 'NG' and len(result[clip_name]['regions']) == 0:
                result.pop(clip_name)
                continue
            if save_image:
                save_path = os.path.join(save_folder, clip_name)
                cv2.imencode(image_ext, clip_images[idx])[1].tofile(save_path)

    save_json(result, save_folder, 'data.json', True)


def cliping_reverse(osrc, crop_jf, crop_size=(2048, 2048), special_clip_reverse: Callable[[Any, Any, Dict, Tuple[int, int]], Any] = None):

    """
        裁切还原大图

    :param osrc: 原图文件夹
    :param org_json: 裁剪json信息
    :param crop_size: 裁切尺寸, (rows, cols)
    :param dst_channel: 裁切图通道数
    :param save_image: 是否保存裁切图像
    :param special_clip: 自定义裁切操作。返回裁切小图和对应的标注信息，标注信息可能是标注图，也可能是标注json
    :return:
    """
    
    
    crop_jd = json.load(open(crop_jf))
    oimps = glob(osrc + "/*.[bj][mp][pg]")
    # raw_mark_folder = create_dir(os.path.join(osrc, 'mark'), )
    pro_bar = tqdm(oimps)
    pro_bar.set_description(f"☞生成原始标注图")
    njsd = {}
    for oimp in pro_bar:
        oimn = osp.basename(oimp)
        oimg = cv2.imdecode(np.fromfile(oimp, dtype=np.uint8), -1)
        
        height, width = oimg.shape[:2]
        clip_reverse_method = __sequential_clipping_reverse
        if height % crop_size[0] != 0 or width % crop_size[1] != 0:
            if special_clip_reverse == None or not callable(special_clip_reverse):
                raise ValueError("请自定义裁剪函数")
            clip_reverse_method = special_clip_reverse
        mark_img = clip_reverse_method(oimg, oimn, crop_jd, crop_size)
        njsd.update(__images2json(mark_img, oimn))
    
    save_json(njsd, osrc, 'data_reverse.json', True)
        
        
        
if __name__ == '__main__':
    # # 还原标注
    # cliping_reverse(r"L:\code\finalshell-download\855G\LJ-tmp",
    #                 r"L:\code\finalshell-download\855G\test-LJ-tmp-p2-040-2048\model_0063999.json",
    #                 special_clip_reverse = splice_clipping_reverse)
    
    
    # # 裁剪标注
    clipping(r'/media/ps/244e88e1-d2e1-477f-9e37-7b9cb43b842a/LB/test/22v19/ok',
            r"/media/ps/244e88e1-d2e1-477f-9e37-7b9cb43b842a/LB/test/22v19/ok/ok.json",
            dst_size=(2048, 2048),
            dst_channel=3, special_clip=splice_clipping)