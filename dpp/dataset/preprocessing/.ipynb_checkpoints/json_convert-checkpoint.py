import os
import cv2
from dpp.common.dpp_json import DppJson


def via_to_json(json_data):
    jx = []
    jy = []
    new_json = {}
    for _, value in json_data.items():
        for region in value['regions']:
            shape_name = region['shape_attributes']['name']
            region['shape_attributes']['name'] = 'polygon'
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
        new_json[filename]["filename"] = filename
        new_json[filename]["size"] = size
        regions = value["regions"]
        for region in regions:
            region["shape_attributes"]["name"] = "polygon"
            region["file_attributes"] = {}
        new_json[filename]["regions"] = regions
    return new_json


def json_to_yolo(img_path, json_data,dst):
    dj = DppJson(json_data)
    if dj.json_format == 'VIA':
        json_data = via_to_json(json_data)
    for key, value in json_data.items():
        filename = value['filename']
        im = cv2.imread(os.path.join(img_path, filename), 0)
        h, w = im.shape[:2]
        regions = value['regions']
        if len(regions) == 0:
            with open(os.path.join(dst, filename.replace(".bmp", ".txt")), "a", encoding="utf-8") as f:
                pass
        else:
            for region in regions:
                xs = region["shape_attributes"]['all_points_x']
                ys = region["shape_attributes"]['all_points_y']
                label = int(region['region_attributes']['regions'])-1
                center_x = (min(xs)+max(xs))/2/w
                center_y = (min(ys)+max(ys))/2/h
                bbox_w = (max(xs)-min(xs))/w
                bbox_h = (max(ys)-min(ys))/h
                with open(os.path.join(dst, filename.replace(".bmp", ".txt")), "a", encoding="utf-8") as f:
                    f.write("{} {} {} {} {}\n".format(
                        label, center_x, center_y, bbox_w, bbox_h))
