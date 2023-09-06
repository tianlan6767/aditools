#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：adi-datatool 
@File    ：comm.py
@Author  ：LvYong
@Date    ：2022/3/15 14:25 
"""
import json
import os
import shutil
import time


def create_dir(folder, del_existence=False):
    """
        创建指定路径并返回创建的路径

    :param folder: 需创建的路径
    :param del_existence: 是否删除已存在的文件夹
    :return: 输入的路径
    """

    if not isinstance(del_existence, bool):
        raise ValueError('del_existence is bool')

    try:
        if del_existence and os.path.exists(folder):
            shutil.rmtree(folder)
    except Exception:
        pass

    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except FileExistsError:
        # TODO: 多进程或多线程调用时，需要优化
        pass

    return folder


def copy_file(src, dst):
    if os.path.exists(src):
        shutil.copy(src, dst)


def box_contains_point(box_x, box_y, point) -> int:
    return int(min(box_x) < point[0] < max(box_x) and min(box_y) < point[1] < max(box_y))


def is_integer(float_val: float):
    return float_val * 10 % 10 == 0


def get_cur_time(time_format: str = '%H%M%S') -> str:
    """获取当前时间戳"""
    return time.strftime(time_format, time.localtime(time.time()))


def get_files(folder, suffix):
    """
        迭代获取文件夹下指定格式文件,注意不同文件夹下的同名问题

    :param folder:
    :param suffix:
    :return: dict key:filename, value:filepath
    """
    res = {}
    for root, directory, files in os.walk(folder):
        for filename in files:
            name, suf = os.path.splitext(filename)
            if suf == suffix:
                res[name] = os.path.join(root, filename)
    return res


def get_file_infos(file_path):
    """
        解析路径信息

    :param file_path: 文件路径
    :return: 父文件夹，文件全称（带后缀），文件名，文件后缀
    """

    root, file_full_name = os.path.split(file_path)
    file_name, fill_ext = os.path.splitext(file_full_name)
    return root, file_full_name, file_name, fill_ext


def save_json(data, path, name, removed=False):
    """
        将字典数据保存成json文件

    :param data: 字典数据
    :param path: json文件保存文件夹
    :param name: json文件保存名称，当名称重复时会自动增加时间后缀
    :param removed: 是否移除已存在的文件
    :return: 保存的路径
    """

    if data is None or len(data) == 0:
        return

    create_dir(path)
    name = name.replace('.json', '') if name.endswith('.json') else name
    save_path = os.path.join(path, '{}.json'.format(name))
    if os.path.exists(save_path):
        if not removed:
            import time
            cur_time = time.strftime('%m%d%H%M', time.localtime(time.time()))
            save_path = os.path.join(path, '{}_{}.json'.format(name, cur_time))
        else:
            os.remove(save_path)
    print(f'{name}.json is saving...')
    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print('save successfully! ->PATH: {}'.format(save_path))
    return save_path
