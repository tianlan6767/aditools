import numpy as np
import shutil
import os
import json
import time
from tqdm import tqdm
from glob import glob
from typing import Any


__all__ = ['create_dir', 'get_file_infos', 'get_json_type', 'get_ok_json', 'load_via_json', 'sort_dict', 'dict_to_str',
           'get_cur_time', 'save_json', 'get_ellipse_box', 'get_points_from_json', 'mul_process', 'mul_thread']


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


def get_file_infos(file_path):
    """
        解析路径信息

    :param file_path: 文件路径
    :return: 父文件夹，文件全称（带后缀），文件名，文件后缀
    """

    root, file_full_name = os.path.split(file_path)
    file_name, fill_ext = os.path.splitext(file_full_name)
    return root, file_full_name, file_name, fill_ext


def get_json_type(json_file):
    """
        获取json文件的类型

    :param json_file: json文件路径
    :return: 'mark': 标注文件  'inf': 推理文件  'unknown': 未知
    """

    if os.path.exists(json_file):
        try:
            data = json.load(open(json_file))
            for k, v in data.items():
                if 'filename' not in v or 'regions' not in v:
                    break
                return v['type'] if 'type' in v else 'mark'
        except Exception:
            pass
    return 'unknown'


def get_ok_json(img_folder) -> dict:
    """
        根据图像名称，生成OK标注信息

    :param img_folder: 图像文件
    :return: dict, ok标注信息
    """

    result = {}
    img_files = glob(os.path.join(img_folder, '*.[bj][mp][pg]'))
    for img_file in img_files:
        img_name = get_file_infos(img_file)[1]
        result[img_name] = dict(filename=img_name, regions=[], type='mark')
    return result


def load_via_json(json_file) -> dict:
    """
        解析区域信息

    :param json_file: via标注信息
    :return: 标注数据格式
    """

    json_data = {}
    if not os.path.exists(json_file):
        return json_data

    data = json.load(open(json_file))
    for v in data.values():
        key = v['filename']
        val = {'filename': key, 'regions': [], 'type': 'mark'}
        for reg in v['regions']:
            region = {
                'shape_attributes': __get_shape_box(reg['shape_attributes']),
                'region_attributes': {"regions": 1},
            }
            val['regions'].append(region)

        json_data[key] = val
    return json_data


def __get_shape_box(region: dict) -> dict:
    """
        解析区域信息

    :param region: 区域信息
    :return: 区域点集 {'all_points_x': [...], 'all_points_y': [...]}
    """

    box = {'all_points_x': [], 'all_points_y': []}
    region_shape = region['name']
    if region_shape == "circle":
        cx = region['cx']
        cy = region['cy']
        r = region['r']
        box['all_points_x'] = [cx - r, cx + r, cx + r, cx - r]
        box['all_points_y'] = [cy - r, cy - r, cy + r, cy + r]
    elif region_shape == "rect":
        left = region['x']
        top = region['y']
        right = region['x'] + region['width']
        bottom = region['y'] + region['height']
        box['all_points_x'] = [left, right, right, left]
        box['all_points_y'] = [top, top, bottom, bottom]
    elif region_shape == "polygon":
        box['all_points_x'] = region['all_points_x']
        box['all_points_y'] = region['all_points_y']
    elif region_shape == "ellipse":
        cx = int(region['cx'])
        cy = int(region['cy'])
        major_r = int(region['rx'])
        minor_r = int(region['ry'])
        theta = int(region['theta'])
        left, top, right, bottom = get_ellipse_box(major_r, minor_r, theta, cx, cy)
        box['all_points_x'] = [left, right, right, left]
        box['all_points_y'] = [top, top, bottom, bottom]
    else:
        raise ValueError(f'不支持解析[{region_shape}]区域!!!')

    return box


def get_cur_time(time_format: str = '%H%M%S') -> str:
    """获取当前时间戳"""
    return time.strftime(time_format, time.localtime(time.time()))


def sort_dict(val: dict, reverse=False, sort_by_key=True):
    """
        根据自定的键值进行排序

    :param val: 字典变量
    :param reverse: 降序排列
    :param sort_by_key: 是否按照键进行排序，False：按照值进行排序
    :return:
    """

    result = {}
    if sort_by_key:
        for k in sorted(val, reverse=reverse):
            result[k] = val[k]
    else:
        for k in sorted(val, key=val.__getitem__, reverse=reverse):
            result[k] = val[k]
    return result


def dict_to_str(val: dict):
    result_str = ''
    for k, v in val.items():
        result_str += f'{k:^2}: {v}\r\n'
    return result_str


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


def get_points_from_json(region_data):
    """
        将标注的region信息转换为点集

    :param region_data:
    :return: [(x1,y1),(x2,y2),...]
    """

    if 'all_points_x' in region_data['shape_attributes']:
        pt_x = region_data['shape_attributes']['all_points_x']
        pt_y = region_data['shape_attributes']['all_points_y']
    elif region_data['shape_attributes']['name'] == 'rect':
        x = region_data['shape_attributes']['x']
        y = region_data['shape_attributes']['y']
        w = region_data['shape_attributes']['width']
        h = region_data['shape_attributes']['height']
        pt_x = [x, x + w, x + w, x]
        pt_y = [y, y, y + h, y + h]
    elif region_data['shape_attributes']['name'] in ['circle', 'ellipse']:
        cx = region_data['shape_attributes']['cx']
        cy = region_data['shape_attributes']['cy']
        if region_data['shape_attributes']['name'] == 'circle':
            r = region_data['shape_attributes']['r']
        else:
            rx = region_data['shape_attributes']['rx']
            ry = region_data['shape_attributes']['ry']
            r = max(rx, ry)
        pt_x = [cx - r, cx + r, cx + r, cx - r]
        pt_y = [cy - r, cy - r, cy + r, cy + r]
    else:
        raise Exception('不支持解析')

    return [(int(pt_x[i]), int(pt_y[i])) for i in range(len(pt_x))]


def get_ellipse_box(major_radius, minor_radius, angle, center_x, center_y):
    """
        计算椭圆形外接矩形

    :param major_radius: 主轴的半径
    :param minor_radius: 短轴半径
    :param angle: (顺时针)旋转角度
    :param center_x: 中心点X坐标
    :param center_y: 中心点Y坐标
    :return: 椭圆外接矩形信息 (left, top, right, bottom)
    """

    # 根据椭圆的主轴和次轴半径以及旋转角度(默认圆心在原点)，得到椭圆参数方程的参数。
    # 椭圆参数方程为：A * x^2 + B * x * y + C * y^2 + F = 0
    a, b = major_radius, minor_radius
    sin_theta = np.sin(-angle)
    cos_theta = np.cos(-angle)
    A = a**2 * sin_theta**2 + b**2 * cos_theta**2
    B = 2 * (a**2 - b**2) * sin_theta * cos_theta
    C = a**2 * cos_theta**2 + b**2 * sin_theta**2
    F = -a**2 * b**2

    # 椭圆上下外接点的纵坐标值
    y = np.sqrt(4 * A * F / (B ** 2 - 4 * A * C))
    y1, y2 = -np.abs(y), np.abs(y)

    # 椭圆左右外接点的横坐标值
    x = np.sqrt(4 * C * F / (B ** 2 - 4 * C * A))
    x1, x2 = -np.abs(x), np.abs(x)

    return center_x + x1, center_y + y1, center_x + x2, center_y + y2


def __data_progress_show(data_len, data_queue, desc) -> None:
    """
        数据处理进度封装函数

    :param data_len: 数据总长度
    :param data_queue: 数据队列
    :param desc: 进度描述
    :return: None
    """

    pro_bar = tqdm(total=data_len)
    temp = 0
    while True:
        num = data_len - data_queue.qsize()
        if temp != num:
            pro_bar.set_description(desc)
            pro_bar.update(num - temp)
            temp = num
        time.sleep(0.1)

        if data_queue.empty():
            pro_bar.close()
            break


def __data_progress(data_queue, process, **kwargs):
    """
        数据处理封装

    :param data_queue: 数据队列
    :param process: 处理函数
    :param kwargs: 处理参数
    :return: Any
    """

    while True:
        if data_queue.empty():
            break

        d = data_queue.get()
        process(d, **kwargs)


def mul_process(data: list, pro, pro_num=3, desc='', **kwargs):
    """
        多进程处理封装函数

    :param data: 处理数据
    :param pro: 处理函数，函数第一个参数必须是处理数据类型
    :param pro_num: 进程数目
    :param desc: 处理进度描述
    :param kwargs: 处理函数参数
    :return:
    """
    from multiprocessing import Process, Queue

    d_queue = Queue(len(data))
    [d_queue.put(d) for d in data]
    pro_num = pro_num if pro_num > 0 else 1
    pros = [Process(target=__data_progress, args=(d_queue, pro), kwargs=kwargs) for _ in range(pro_num)]
    pros.insert(0, Process(target=__data_progress_show, args=(len(data), d_queue, desc)))
    [p.start() for p in pros]
    [p.join() for p in pros]


def mul_thread(data: list, pro, td_num=5, desc='', **kwargs) -> Any:
    """
        多线程处理封装函数

    :param data: 处理数据
    :param pro: 处理函数，函数第一个参数必须是处理数据类型
    :param td_num: 线程数目
    :param desc: 处理进度的描述信息
    :param kwargs: 处理函数其他参数
    :return: Any
    """
    from concurrent.futures import ThreadPoolExecutor

    pro_bar = tqdm(range(len(data)), desc=desc)
    result = None
    td_num = td_num if td_num > 0 else 1
    with ThreadPoolExecutor(max_workers=td_num) as t:
        for i in range(0, len(data), td_num):
            ds = data[i:i + td_num]
            tasks = [t.submit(pro, d, **kwargs) for d in ds]
            is_done = [task.done() for task in tasks]
            while False in is_done:
                time.sleep(0.001)
                is_done = [task.done() for task in tasks]

            result = tasks[0].result()
            pro_bar.update(td_num)
    pro_bar.close()
    time.sleep(0.01)

    return result
