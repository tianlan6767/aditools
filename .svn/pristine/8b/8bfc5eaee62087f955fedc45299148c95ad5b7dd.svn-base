#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：adi-datatool 
@File    ：utils.py
@Author  ：LvYong
@Date    ：2022/3/9 17:54 
"""
import os

import cv2
import math
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from typing import Tuple, List, Union
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from comm import *


def feature_similarity(feature1: list, feature2: list, sim_method: str) -> float:
    """
        计算两个向量间的相似度

    :param feature1: 特征向量1
    :param feature2: 特征向量2
    :param sim_method: 相似度计算方法['manhattan', 'euclidean', 'chebyshev', 'pearsonr', 'cosine']
    :return:
    """

    if len(feature1) != len(feature2):
        raise ValueError('feature1 and feature2 must have the same length.')
    if len(feature1) < 2:
        raise ValueError('feature1 and feature2 must have length at least 2.')

    feature1 = np.asarray(feature1)
    feature2 = np.asarray(feature2)

    if sim_method == 'manhattan':
        return np.linalg.norm(feature1 - feature2, ord=1)
    elif sim_method == 'euclidean':
        return np.linalg.norm(feature1 - feature2)
    elif sim_method == 'chebyshev ':
        return np.linalg.norm(feature1 - feature2, ord=np.inf)
    elif sim_method == 'pearsonr':
        return stats.pearsonr(feature1, feature2)[0]
    elif sim_method == 'cosine':
        # 向量点乘
        num = float(np.dot(feature1, feature2))
        # 求模长的乘积
        den = np.linalg.norm(feature1) * np.linalg.norm(feature2)
        return 0.5 + 0.5 * (num / den) if den != 0 else 0
    return 0.0


def calc_similarity(features1, features2, sim_method, sim_threshold):
    """
        计算两组数据间的相似性

    :param features1: 数据1，对比对象
    :param features2: 数据2，参考对象
    :param sim_method: 相似性算法
    :param sim_threshold: 相似性阈值，(min, max)
    :return: 数据1中与数据2相似的索引和数据2与数据1的相似对应关系
    """

    sim_idxes, sim_infos = [], defaultdict(list)
    with tqdm(features1, desc='相似度计算', ncols=100) as process_bar:
        for src_idx, src_feature in enumerate(process_bar):
            for ref_idx, ref_feature in enumerate(features2):
                sim = feature_similarity(src_feature, ref_feature, sim_method)
                process_bar.set_postfix(sim=f'{sim:.6f}', ref_num=len(features2))
                if sim_threshold[1] > sim > sim_threshold[0]:
                    sim_idxes.append(src_idx)
                    sim_infos[ref_idx].append(src_idx)
                    break
    print(f'相似数据量:{len(sim_idxes)}, 不相似数据量:{len(features1) - len(sim_idxes)}')
    return sim_idxes, sim_infos


def normalize_data(data1: pd.DataFrame, data2: pd.DataFrame, irrelevant=True):
    """
        标准化数据处理

    :param data1: 数据1
    :param data2: 数据2
    :param irrelevant: 输入数据是否不相关，True:数据1,2不相干、False:数据1包含数据2
    :return: 处理后数据
    """

    if irrelevant:
        all_data_frame = pd.concat((data1, data2))
        all_data = StandardScaler().fit_transform(all_data_frame)
        return all_data[:len(data1)], all_data[len(data1):]
    else:
        ss = StandardScaler()
        data = ss.fit_transform(data1)
        return data, ss.transform(data2)


def compare_feature(src_file, ref_file, feature_idxes, sim_method, sim_threshold: Tuple[float, float],
                    save_name, add_file=None, normalized=True, irrelevant=True):
    """
        特征对比，根据ref_file特征对src_file进行筛选并保存对比结果

    :param src_file: 待处理特征文件
    :param ref_file: 对比参考特征文件
    :param feature_idxes: 对比参考特征序列，参考ref_file的列名索引
    :param sim_method: 特征对比方法
    :param sim_threshold: 特征相似阈值，(min, max)
    :param save_name: 保存文件名称，保存路径与src_file一致
    :param add_file: 附件特征文件，用于src_file筛选后和add_file内容相加再保存
    :param normalized: 是否对输入数据进行标准化处理
    :param irrelevant: 数据数据是否不相关(不存在包含与被包含的关系)
    :return: None
    """

    src_data_frame = pd.read_excel(src_file)
    ref_data_frame = pd.read_excel(ref_file)

    feature_keys = ref_data_frame.columns.tolist()
    feature_keys = [feature_keys[i] for i in feature_idxes]
    if normalized:
        src_data, ref_data = normalize_data(src_data_frame[feature_keys],
                                            ref_data_frame[feature_keys],
                                            irrelevant)
    else:
        src_data = src_data_frame[feature_keys].values
        ref_data = ref_data_frame[feature_keys].values

    sim_idxes = calc_similarity(src_data, ref_data, sim_method, sim_threshold)[0]

    sim_idxes = sorted(sim_idxes, reverse=True)
    new_data = src_data_frame.values.tolist()
    for remove_idx in sim_idxes:
        new_data.pop(remove_idx)
    new_data_frame = pd.DataFrame(new_data, columns=src_data_frame.columns)
    if add_file and os.path.exists(add_file):
        add_data_frame = pd.read_excel(add_file)
        new_data_frame = pd.concat((new_data_frame, add_data_frame)).drop(
            ['qcJdg(Random Forest)'], axis=1, errors='ignore')
    new_data_frame.to_excel(os.path.join(os.path.split(src_file)[0], f'{save_name}.xlsx'), index=False)
    print(f'生成{save_name}.xlsx成功')


def get_similar_image(src_file, ref_file, feature_idxes, image_folder, sim_method, sim_threshold: Tuple[float, float],
                      save_name, normalized=True, irrelevant=True):
    """
        在src_file中获取与ref_file相似的图像

    :param src_file: 对比特征文件
    :param ref_file: 参考特征文件
    :param feature_idxes: 对比参考特征序列，参考ref_file的列名索引
    :param image_folder: src_file与ref_file对应图像所在文件夹
    :param sim_method: 特征对比方法
    :param sim_threshold: 特征相似阈值，(min, max)
    :param save_name: 相似图像的保存文件夹名称，保存路径与src_file一致
    :param normalized: 是否对输入数据进行标准化处理
    :param irrelevant: 数据数据是否不相关(不存在包含与被包含的关系)
    :return:
    """

    src_data_frame = pd.read_excel(src_file)
    ref_data_frame = pd.read_excel(ref_file)

    feature_keys = ref_data_frame.columns.tolist()
    feature_keys = [feature_keys[i] for i in feature_idxes]
    if normalized:
        src_data, ref_data = normalize_data(src_data_frame[feature_keys],
                                            ref_data_frame[feature_keys],
                                            irrelevant)
    else:
        src_data = src_data_frame[feature_keys].values
        ref_data = ref_data_frame[feature_keys].values

    sim_infos = calc_similarity(src_data, ref_data, sim_method, sim_threshold)[1]

    def _get_thumb_name(data_frame, row_idx):
        thumb_name = data_frame['filename'][row_idx]
        if thumb_name == np.nan:
            print(f'row_idx: {row_idx + 2}')
        thumb_name = os.path.splitext(thumb_name)[0]
        return f"{thumb_name}_{data_frame['rIdx'][row_idx]}"

    image_files = get_files(image_folder, '.bmp')
    save_folder = create_dir(os.path.join(os.path.split(src_file)[0], save_name), True)
    with tqdm(sim_infos.items(), desc='提取相似图像', ncols=100) as process_bar:
        for ref_idx, src_idxes in process_bar:
            ref_name = _get_thumb_name(ref_data_frame, ref_idx)
            ref_img_dir = create_dir(os.path.join(save_folder, f'{ref_idx}'))
            copy_file(image_files[ref_name], ref_img_dir)
            files = [image_files[ref_name]]
            src_img_dir = create_dir(os.path.join(save_folder, f'{ref_idx}', 'images'))
            for src_idx in src_idxes:
                src_name = _get_thumb_name(src_data_frame, src_idx)
                copy_file(image_files[src_name], src_img_dir)
                files.append(image_files[src_name])
            new_img = stitching_images(files)
            cv2.imencode('.bmp', new_img)[1].tofile(os.path.join(save_folder, f'{ref_idx}_{ref_name}.bmp'))


def stitching_images(images: List[Union[str, np.array]], interval=5):
    """
        拼接图像

    :param images: 图像集
    :param interval: 图像最小间隔
    :return: 新图像
    """

    assert isinstance(images, list), '输入错误'
    if not len(images):
        return None

    if isinstance(images[0], str):
        images = [cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_ANYCOLOR)
                  for file in images if os.path.exists(file)]

    # 创建背景图，根据图像数目自适应计算背景图的大小
    max_height = max([img.shape[0] for img in images])
    max_width = max([img.shape[1] for img in images])
    img_num = len(images)
    img_cols = math.sqrt(img_num)
    img_cols = int(img_cols) if is_integer(img_cols) else int(img_cols) + 1
    img_rows = img_num / img_cols
    img_rows = int(img_rows) if is_integer(img_rows) else int(img_rows) + 1
    bg_image = np.zeros((img_rows * max_height + (interval * img_rows + 1),
                         img_cols * max_width + (interval * img_cols + 1), 3),
                        dtype=np.uint8) + 255

    # 拼接图像
    offset_row = interval
    for img_row in range(img_rows):
        offset_col = interval
        for img_col in range(img_cols):
            img_idx = img_row * img_cols + img_col
            if img_idx >= img_num:
                break

            if len(images[img_idx].shape) == 2 or images[img_idx].shape[2] == 1:
                images[img_idx] = cv2.cvtColor(images[img_idx], cv2.COLOR_GRAY2BGR)

            img_height, img_width = images[img_idx].shape[:2]
            x = offset_col + (max_width - img_width) // 2
            y = offset_row + (max_height - img_height) // 2
            bg_image[y:(y + img_height), x:(x + img_width), ::] = images[img_idx]

            offset_col += (interval + max_width)
        offset_row += (interval + max_height)

    return bg_image


def calc_product_yield(defect_file, combined=False):
    """
        根据softJdg字段，统计软件数据产品信息的良率!

    :param defect_file: 缺陷信息
    :param combined: 模型和软件检测是否结合使用，结合规则：模型检测OK+软件检测NG/OK
    :return: None
    """

    defect_data = json.load(open(defect_file))
    product_infos = defaultdict(list)    # 每个产品的图像检出结果， True表示OK, False表示NG
    dft_num, dft_qc_num, dft_st_ng_num, dft_md_ng_num, dft_st_ng, dft_md_ng = 0, 0, 0, 0, 0, 0
    st_pdt_name, md_pdt_name = [], []
    chk_qc_regions, miss_qc_regions = [], []
    for img_name, v in defect_data.items():
        # 1.统计产品
        pdt_id = v['sample']['pdtNo']
        pdt_id = f'{img_name.split("_")[0]}{pdt_id}'
        # 判断当前图像是否为质检OK图
        qc_ok = sum([r['region_attributes']['qcJdg'] == 1 for r in v['regions']]) == 0
        product_infos[f'{pdt_id}_qc'].append(qc_ok)
        # 判断当前图像是否为检测OK图
        soft_ok = sum([r['region_attributes']['softJdg'] == 2 for r in v['regions']]) == 0
        product_infos[f'{pdt_id}_soft'].append(soft_ok)
        product_infos[f'{pdt_id}_st_real'].append(qc_ok and soft_ok)
        # 判断当前图像是否为模型OK图
        if combined:
            model_ok = sum([r['region_attributes']['mdJdg'] == 1 and r['region_attributes']['softJdg'] == 2
                            for r in v['regions']]) == 0
            product_infos[f'{pdt_id}_model'].append(model_ok)
            product_infos[f'{pdt_id}_md_real'].append(qc_ok and model_ok)
        else:
            model_ok = sum([r['region_attributes']['mdJdg'] == 1 for r in v['regions']]) == 0
            product_infos[f'{pdt_id}_model'].append(model_ok)
            product_infos[f'{pdt_id}_md_real'].append(qc_ok and model_ok)

        # 2.统计缺陷
        for idx, r in enumerate(v['regions']):
            dft_num += 1
            is_qc_ng = int(r['region_attributes']['qcJdg'] == 1)
            dft_qc_num += is_qc_ng

            if r['region_attributes']['softJdg'] == 2:
                dft_st_ng_num += 1
                dft_st_ng += is_qc_ng
                if is_qc_ng:
                    pdt_name = "_".join(img_name.split("_")[:2])
                    if pdt_name not in st_pdt_name:
                        st_pdt_name.append(pdt_name)

            r['region_attributes']['filename'] = v['filename']
            if r['region_attributes']['mdJdg'] == 1 and (not combined or r['region_attributes']['softJdg'] == 2):
                dft_md_ng_num += 1
                dft_md_ng += is_qc_ng
                if is_qc_ng:
                    chk_qc_regions.append(r)
                    pdt_name = "_".join(img_name.split("_")[:2])
                    if pdt_name not in md_pdt_name:
                        md_pdt_name.append(pdt_name)
            elif is_qc_ng:
                miss_qc_regions.append(r)

    soft_ok_num = sum([sum(i) == len(i) for k, i in product_infos.items() if 'soft' in k])
    soft_ng_num, soft_qc_ng_num = 0, 0
    for k, i in product_infos.items():
        if 'soft' in k:
            if sum(i) != len(i):
                soft_ng_num += 1
                pdt_id = k.split('_')[0]
                if sum(product_infos[f'{pdt_id}_qc']) != len(product_infos[f'{pdt_id}_qc']):
                    soft_qc_ng_num += 1
    qc_ok_num = sum([sum(i) == len(i) for k, i in product_infos.items() if 'qc' in k])
    qc_ng_num = sum([sum(i) != len(i) for k, i in product_infos.items() if 'qc' in k])
    real_ok_num = sum([sum(i) == len(i) for k, i in product_infos.items() if 'st_real' in k])
    model_ok_num = sum([sum(i) == len(i) for k, i in product_infos.items() if 'model' in k])
    model_ng_num, model_qc_ng_num = 0, 0
    for k, product_info in product_infos.items():
        if 'model' in k:
            if False in product_info:
                model_ng_num += 1
                pdt_id = k.split('_')[0]
                if False in product_infos[f'{pdt_id}_qc']:
                    model_qc_ng_num += 1
    md_real_ok_num = sum([sum(i) == len(i) for k, i in product_infos.items() if 'md_real' in k])
    pdt_num = len([k for k in product_infos.keys() if 'soft' in k])
    result = f'产品数量:{pdt_num}, ' \
             f'产品良率:{(qc_ok_num / pdt_num):.2%}, ' \
             f'不良品:{qc_ng_num}, ' \
             f'产品缺陷[QC|Total]: {dft_qc_num} | {dft_num} \n' \
             f'软件良率:{(soft_ok_num / pdt_num):.2%}, ' \
             f'软件真实良率:{(real_ok_num / pdt_num):.2%}, ' \
             f'软件检出[Valid|QC|Total]: {len(st_pdt_name)} | {soft_qc_ng_num} | {soft_ng_num}, ' \
             f'软件缺陷[Valid|Total]: {dft_st_ng} | {dft_st_ng_num} \n' \
             f'模型良率:{(model_ok_num / pdt_num):.2%}, ' \
             f'模型真实良率:{(md_real_ok_num / pdt_num):.2%}, ' \
             f'模型检出[Valid|QC|Total]: {len(md_pdt_name)} | {model_qc_ng_num} | {model_ng_num}, ' \
             f'模型缺陷[Valid|Total]: {dft_md_ng} | {dft_md_ng_num}'
    print(result)
    return chk_qc_regions, miss_qc_regions


def update_data(defect_file, excel_file, json_name, judge=1):
    """

    :param defect_file:
    :param excel_file: 来源于orange3, 主要是模型检出数据，检出数据类型通过judge设定
    :param json_name:
    :param judge: excel_file描述的是哪类信息
    :return:
    """
    data_frame = pd.read_excel(excel_file)
    defect_data = json.load(open(defect_file))
    # 初始化模型检出
    for v in defect_data.values():
        for region in v['regions']:
            if 'mdJdg' not in region['region_attributes']:
                region['region_attributes']['mdJdg'] = 1 - judge
    # 更新模型检出
    for row, file_name in enumerate(data_frame['filename'].values):
        region_id = data_frame['rIdx'][row]
        defect_data[file_name]['regions'][region_id]['region_attributes']['mdJdg'] = judge

    save_json(defect_data, os.path.split(defect_file)[0], json_name, True)


if __name__ == '__main__':
    # compare_feature(src_file=r'D:\Share\users\LY\eval\batch_352_0.xlsx',
    #                 ref_file=r'D:\Share\users\LY\eval\batch_352_1.xlsx',
    #                 feature_idxes=[0, 1, 2, 3, 4, 5, 8, 15, 16, 19],
    #                 sim_method='pearsonr', sim_threshold=(0.98, 0.99999),
    #                 save_name='data222',
    #                 add_file=r'D:\Share\users\LY\eval\batch_352_1.xlsx')

    # get_similar_image(src_file=r"D:\Share\users\LY\eval\batch_352_0_1.xlsx",
    #                   ref_file=r"D:\Share\users\LY\eval\batch_352_1_1.xlsx",
    #                   feature_idxes=[0, 1, 2, 3, 4, 5, 8, 15, 16, 19, 23],
    #                   image_folder=r'D:\Share\users\LY\eval\thumb',
    #                   sim_method='pearsonr', sim_threshold=(0.95, 0.99999),
    #                   save_name='sim01')

    # update_data(defect_file=r'D:\Projects\分类\HG827\batch_2225.json',
    #             excel_file=r'D:\Projects\分类\HG827\batch_2225_ok.xlsx',
    #             json_name=r'batch_2225_update.json',
    #             judge=0)

    # print(">>> 单模型")
    # calc_product_yield(r'D:\Projects\分类\HG827\batch_2225_update.json',
    #                    combined=False)
    print(">>> 单模型+限度")
    calc_product_yield(r'D:\Projects\分类\HG827\backup\batch_2225_update.json',
                       combined=True)

    # pd_data = pd.read_excel(r"D:\Projects\分类\HG827\test\batch_2297.xlsx")
    # js_file = r'D:\Projects\分类\HG827\test\batch_2297.json'
    # js_data = json.load(open(js_file))
    # for r, d in enumerate(pd_data['circularity']):
    #     filename = pd_data['filename'][r]
    #     rIdx = pd_data['rIdx'][r]
    #     js_data[filename]['regions'][rIdx]['region_attributes']['circularity'] = round(d, 4)
    # 
    # save_json(js_data, os.path.dirname(js_file), os.path.basename(js_file), True)

