import os
import cv2
import numpy as np
from tqdm import tqdm
from dpp.common.dpp_json import DppJson


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

def via_to_json(json_data):
    jx = []
    jy = []
    new_json = {}
    for _, value in json_data.items():
        for region in value['regions']:
            shape_name = region['shape_attributes']['name']
            # region['shape_attributes']['name'] = 'polygon'
            if 'size' in value:
                value.pop("size")
            if 'file_attributes' in value:
                value.pop("file_attributes")
            if shape_name == "rect":
                l_ptx = region['shape_attributes']['x']
                l_pty = region['shape_attributes']['y']
                R_ptx = region['shape_attributes']['x'] + \
                    region['shape_attributes']['width']
                R_pty = region['shape_attributes']['y'] + \
                    region['shape_attributes']['height']
                jx = [l_ptx] + [R_ptx] + [R_ptx] + [l_ptx]
                jy = [l_pty] + [l_pty] + [R_pty] + [R_pty]
                region['shape_attributes']['x'] = jx
                region['shape_attributes']['y'] = jy
                region['shape_attributes'].pop("width")
                region['shape_attributes'].pop("height")
                region['shape_attributes'].pop('name')
                tmp_all_x = "all_points_x"
                tmp_all_y = "all_points_y"
                region['shape_attributes'][tmp_all_x] = region['shape_attributes'].pop(
                    "x")
                region['shape_attributes'][tmp_all_y] = region['shape_attributes'].pop(
                    "y")
                region['region_attributes'].setdefault('regions', '1')
            elif shape_name == "circle":
                l_ptx = region['shape_attributes']['cx'] - \
                    region['shape_attributes']['r']
                l_pty = region['shape_attributes']['cy'] - \
                    region['shape_attributes']['r']
                R_ptx = region['shape_attributes']['cx'] + \
                    region['shape_attributes']['r']
                R_pty = region['shape_attributes']['cy'] + \
                    region['shape_attributes']['r']
                jx = [l_ptx] + [R_ptx] + [R_ptx] + [l_ptx]
                jy = [l_pty] + [l_pty] + [R_pty] + [R_pty]
                region['shape_attributes']["all_points_x"] = jx
                region['shape_attributes']["all_points_y"] = jy
                region['region_attributes'].setdefault('regions', '1')
                region['shape_attributes'].pop("cx")
                region['shape_attributes'].pop("cy")
                region['shape_attributes'].pop("r")
                region['shape_attributes'].pop('name')
            elif shape_name == 'ellipse':
                cx = int(region['shape_attributes']['cx'])
                cy = int(region['shape_attributes']['cy'])
                major_r = int(region['shape_attributes']['rx'])
                minor_r = int(region['shape_attributes']['ry'])
                theta = int(region['shape_attributes']['theta'])
                box = get_ellipse_box(major_r, minor_r, theta, cx, cy)
                region['shape_attributes']["all_points_x"] = [
                    box[0], box[2], box[2], box[0]]
                region['shape_attributes']["all_points_y"] = [
                    box[1], box[1], box[3], box[3]]
                region['region_attributes'].setdefault('regions', '1')
                region['shape_attributes'].pop("cx")
                region['shape_attributes'].pop("cy")
                region['shape_attributes'].pop("rx")
                region['shape_attributes'].pop("ry")
                region['shape_attributes'].pop("theta")
                region['shape_attributes'].pop('name')
            elif shape_name == "polygon" or shape_name == "polyline":
                region['shape_attributes']['all_points_x'] = region['shape_attributes']['all_points_x']
                region['shape_attributes']['all_points_y'] = region['shape_attributes']['all_points_y']
                region['shape_attributes'].pop('name')
            else:
                print(f'不支持解析{shape_name}')
                return
        new_json[value['filename']] = value
    return new_json


def json_to_via(imgs, json_data):
    img_path = os.path.dirname(imgs[0])
    new_json = {}
    for key, value in json_data.items():
        size = os.path.getsize(os.path.join(img_path, key))
        filename = str(key+str(size))
        new_json[filename] = {}
        new_json[filename]["filename"] = key
        new_json[filename]["size"] = size
        regions = value["regions"]
        for region in regions:
            region["shape_attributes"]["name"] = "polygon"
            region["file_attributes"] = {}
        new_json[filename]["regions"] = regions
    return new_json


def json_to_yolo(img_path, json_data,dst,seg=False):
    dj = DppJson(json_data)
    if dj.json_format == 'VIA':
        json_data = via_to_json(json_data)
    for key, value in tqdm(json_data.items()):
        filename = value['filename']
        im = cv2.imread(os.path.join(img_path, filename), 0)
        h, w = im.shape[:2]
        regions = value['regions']
        if len(regions) == 0:
            with open(os.path.join(dst, filename.replace(".bmp", ".txt")), "a", encoding="utf-8") as f:
                pass
        else:
            if seg:
                for region in regions:
                    xs = region["shape_attributes"]['all_points_x']
                    ys = region["shape_attributes"]['all_points_y']
                    label = int(region['region_attributes']['regions'])-1
                    xy = np.array(list(zip([x/w for x in xs],[y/h for y in ys]))).flatten().tolist()
                    value = str(label)
                    for item in xy:
                        value += " "+str(item)
                    with open(os.path.join(dst, filename.replace(".bmp", ".txt").replace(".jpg", ".txt")), "a", encoding="utf-8") as f:
                        f.write("{}\n".format(value))
            else:
                for region in regions:
                    xs = region["shape_attributes"]['all_points_x']
                    ys = region["shape_attributes"]['all_points_y']
                    label = int(region['region_attributes']['regions'])-1
                    center_x = (min(xs)+max(xs))/2/w
                    center_y = (min(ys)+max(ys))/2/h
                    bbox_w = (max(xs)-min(xs))/w
                    bbox_h = (max(ys)-min(ys))/h
                    with open(os.path.join(dst, filename.replace(".bmp", ".txt").replace(".jpg", ".txt")), "a", encoding="utf-8") as f:
                        f.write("{} {} {} {} {}\n".format(
                            label, center_x, center_y, bbox_w, bbox_h))