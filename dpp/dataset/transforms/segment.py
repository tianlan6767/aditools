import cv2
import json
import yaml
import os
import math
import numpy as np
from abc import abstractmethod
from dpp.common.util import cal_area, parse_region
from dpp.common.registry import Registry
from dpp.common.mylog import Logger
from dpp.dataset.transforms.pad import pad_im


SEGMENT_RULE_REGISTRY = Registry("SEGMENT_RULE")

def filename_mapper(filename):
    # 1_1_1 不需映射，直接返回
    if len(filename.split("_"))==3 and "-" not in filename:
        return filename.replace(".bmp", "").replace(".jpg", "")
    try:
        # 产品号-图片点位信息 提取点位
        new_name = filename.split("-")[1]
        new_name = "_".join(new_name.split("_")[:3])
        new_name = new_name.replace(".bmp", "").replace(".jpg", "")
        return new_name
    except:
        Logger.error("图片名<{}>命名不规范".format(filename))


def _segment_box(im, crop_size=2048,extra="DROP"):
    """
    :param  crop_size: 裁剪的尺寸 H*W
    :param image_size: 原图尺寸 H*W
    :return: [[x1,y1,x2,y2],...[]]
    """
    crop_height, crop_width = crop_size, crop_size
    height, width = im.shape[:2]
    split_y, split_x = int(height / crop_height), int(width / crop_width)
    box_list = []
    for y in range(0, split_y):
        for x in range(0, split_x):
            box = (crop_width * x, crop_height * y,
                   crop_width * (x + 1), crop_height * (y + 1))
            box_list.append(box)
            
    extra_py,extra_px = height % crop_height,width % crop_width
    if extra == "FILL":
        if extra_px != 0:
            for y in range(split_y):
                box_list.insert(split_x*(y+1)+y,[split_x*crop_size,y*crop_size,(split_x+1)*crop_size,(y+1)*crop_size])
        if extra_py != 0:
            for x in range(split_x):
                box_list.append([x*crop_size,split_y*crop_size,(x+1)*crop_size,(split_y+1)*crop_size])
        if extra_px != 0 and extra_py != 0:
            box_list.append([split_x*crop_size,split_y*crop_size,(split_x+1)*crop_size,(split_y+1)*crop_size])
    elif extra == "KEEP":
        if extra_px != 0:
            for y in range(split_y):
                box_list.insert(split_x*(y+1)+y,[split_x*crop_size,y*crop_size,width,(y+1)*crop_size])
        if extra_py != 0:
            for x in range(split_x):
                box_list.append([x*crop_size,split_y*crop_size,(x+1)*crop_size,height])
        if extra_px != 0 and extra_py != 0:
            box_list.append([split_x*crop_size,split_y*crop_size,width,height])
    else:
        pass
    return box_list


def crop_box(box, im,pad_size=None,extra=None):
    if set([isinstance(item, list) for item in box]) == {True}:
        merge_jf_im = np.vstack(
            (im[box[0][1]:box[0][3], box[0][0]:box[0][2]], im[box[1][1]:box[1][3], box[1][0]:box[1][2]]))
        if len(im.shape) == 2:
            new_im = np.vstack((merge_jf_im, np.zeros((144, 2048), np.uint8)))
        else:
            new_im = np.vstack((merge_jf_im, np.zeros((144, 2048, 3), np.uint8)))
    else:
        if extra=="FILL":
            im = pad_im(im,pad_size=(math.ceil(im.shape[0]/pad_size)*pad_size,math.ceil(im.shape[1]/pad_size)*pad_size))
        new_im = im[box[1]:box[3], box[0]:box[2]]
    return new_im


def counter_to_polygon(crop_jf_im, min_area=20):
    per_contours_polygon = []
    contours, _ = cv2.findContours(
        crop_jf_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        polygons = contour.flatten().tolist()
        xs, ys = polygons[0::2], polygons[1::2]
        area = cal_area(xs, ys)
        if area < min_area or len(xs) < 4:
            pass
        else:
            cate = crop_jf_im[ys[0]][xs[0]]
            new_dict = {'shape_attributes': {'all_points_x': xs, 'all_points_y': ys},
                        'region_attributes': {'regions': str(cate//15)}}
            per_contours_polygon.append(new_dict)
    return per_contours_polygon


def cal_crop_box_region(im, box_list, regions, min_area=20):
    jf_im = np.zeros(im.shape[:2], np.uint8)
    for region in regions:
        xs, ys, label = parse_region(region)
        counter = list(zip(xs, ys))
        cv2.fillPoly(jf_im, [np.array(counter)], int(label)*15)

    per_box_polygon = []
    for box in box_list:
        crop_jf_im = crop_box(box, jf_im)
        per_contours_polygon = counter_to_polygon(crop_jf_im)
        per_box_polygon.append(per_contours_polygon)
    return per_box_polygon


class Segment:
    def __init__(self):
        # ./transform/config.yaml
        self.ya = yaml.load(open("./dpp/dataset/transforms/config.yaml"))
        self.cfg = self.ya[self.__class__.__name__]

    @abstractmethod
    def crop_im(self, im_dict):
        """
        im_dict : {filename : img_numpy}
        """
        raise NotImplemented

    @abstractmethod
    def crop_polygon(self, im, box_list, regions):
        """
        box_list: LIST[List[x1,y1,x2,y2],...]
        regions : regions:[{},{}]
        """
        raise NotImplemented


@SEGMENT_RULE_REGISTRY.register()
class AvgSeg(Segment):
    def __init__(self):
        super().__init__()

    def crop_im(self, im_dict):
        filename = list(im_dict.keys())[0]
        im = im_dict[filename]
        box_list = _segment_box(im, crop_size=self.cfg["crop_size"],extra=self.cfg["extra"])
        return box_list

    def crop_polygon(self, im, box_list, regions):
        if len(regions):
            return cal_crop_box_region(im, box_list, regions)
        else:
            return [[]]*(len(box_list))


@SEGMENT_RULE_REGISTRY.register()
class CeSeg(Segment):
    def __init__(self):
        super().__init__()

    def crop_im(self, im_dict):
        filename = list(im_dict.keys())[0]
        im = im_dict[filename]
        h, w = im.shape[:2]
        return [[self.cfg["start"], 0, self.cfg["end"], h]]

    def crop_polygon(self, im, bbox_list, regions):
        h, w = im.shape[:2]
        box_list = [[self.cfg["start"], 0, self.cfg["end"], h]]
        if len(regions):
            return cal_crop_box_region(im, box_list, regions)
        else:
            return [[]]


@SEGMENT_RULE_REGISTRY.register()
class ThreeSeg(Segment):

    @property
    def box_list(self):
        return [[0, 0, 2048, 2048], [2048, 0, 4096, 2048], [[0, 2048, 2048, 3000], [2048, 2048, 4096, 3000]]]

    def crop_im(self, im_dict):
        filename = list(im_dict.keys())[0]
        im = im_dict[filename]
        if im.shape != (3000, 4096, 3):
            Logger.error("class<{}>只针对三工位3000,4096像素的图片".format(
                self.__class__.__name__))
        return self.box_list

    def crop_polygon(self, im, bbox_list, regions):
        if len(regions):
            return cal_crop_box_region(im, bbox_list, regions)
        else:
            return [[]]*3


@SEGMENT_RULE_REGISTRY.register()
class Cv2Seg(Segment):
    def __init__(self):
        super().__init__()

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.cfg["kernel_size"], self.cfg["kernel_size"]))

    def crop_im(self, im_dict):
        filename = list(im_dict.keys())[0]
        im = im_dict[filename]
        h, w = im.shape[:2]
        im = cv2.Canny(im, self.cfg["th_min"], self.cfg["th_max"])
        im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel=self.kernel)
        im = cv2.medianBlur(im, self.cfg["blur"])

        counters, _ = cv2.findContours(
            im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(counters) == 0:
            Logger.warning("图片 {} 未检测到边缘".format(filename))
            return [[0, 0, w, h]]

        num_counters = [c.flatten() for c in counters]
        all_points = [c for nc in num_counters for c in nc]
        xs, ys = all_points[::2], all_points[1::2]
        return [[min(xs)-self.cfg["pad_size"], 0, max(xs)+self.cfg["pad_size"], h]]

    def crop_polygon(self, im, bbox_list, regions):
        if len(regions):
            return cal_crop_box_region(im, bbox_list, regions)
        else:
            return [[]]


@SEGMENT_RULE_REGISTRY.register()
class JsonSeg(Segment):
    def __init__(self):
        super().__init__()
        self.bbox_dict = self.bbox_dict(self.cfg["jfs"])

    def parse_point_roi(self, jf):
        json_data = json.load(open(jf))
        point_dict = {}
        # 根据roi_w1.json解析工位名
        station = jf.split("roi_w")[-1].replace(".json", "")
        for bmp_k, bmp_v in json_data.items():
            # 解析点位
            for point_k, point_v in bmp_v.items():
                json_fn = "{}_{}_{}".format(station, int(point_k.replace(
                    "cam_", ""))+1, int(bmp_k.replace("point_", ""))+1)
                # 解析裁剪坐标
                bbox_list = []
                for index, v in enumerate(point_v):
                    x1 = point_v[index]["c1"]
                    x2 = point_v[index]["c2"]
                    y1 = point_v[index]["r1"]
                    y2 = point_v[index]["r2"]
                    bbox_list.append([x1, y1, x2, y2])
                point_dict[json_fn] = bbox_list
        return point_dict

    def bbox_dict(self, jfs):
        point_mapper_dict = {}
        assert isinstance(jfs, list), "必须传入裁剪json文件路径列表"
        [point_mapper_dict.update(self.parse_point_roi(jf)) for jf in jfs]
        return point_mapper_dict

    def crop_im(self, im_dict):
        filename = list(im_dict.keys())[0]
        point_name = filename_mapper(filename)
        try:
            out = self.bbox_dict[point_name]
            return out
        except:
            Logger.error("图片<{}>没有映射关系坐标".format(filename))

    def crop_polygon(self, im, bbox_list, regions):
        if len(regions):
            return cal_crop_box_region(im, bbox_list, regions)
        else:
            # if bbox_list == None:
            #     return [[0, 0, 2048, 2048]]
            return [[]]*len(bbox_list)
