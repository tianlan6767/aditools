import os
import sys
import glob
import json
import time
import cv2
from typing import List
import numpy as np
from dpp.common.mylog import Logger

class MultiDataset:
    def intersection(self, list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        if len(set1) < len(set2):
            return [i for i in set1 if i in set2]
        else:
            return [i for i in set2 if i in set1]
          
          
def flatten(l: list):
    for _ in l:
        if isinstance(_, list):
            yield from flatten(_)
        else:
            yield _


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_json(path):
    if path is None:
        return []
    elif os.path.isfile(path):
        data = open(path,encoding='utf-8')
        json_data = json.load(data)
        return json_data
    else:
        raise ValueError


def load_file(path, format="img"):
    if sys.platform == "win32":
        if format == "img":
            return glob.glob(path+'\*.[jb][pm][gp]')
        elif format == "json":
            return glob.glob(path+'\*.json')
        else:
            print("error")
    else:
        if format == "img":
            return glob.glob(path+'/*.[jb][pm][gp]')
        elif format == "json":
            return glob.glob(path+'/*.json')
        else:
            print("error")


def rect_area(bbox):
    """
    bbox:[x1,y1,x2,y2]
    """
    assert isinstance(bbox, List), "矩形框格式错误"
    return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])


def save_json(data, path, name):
    if len(data):
        make_dir(path)
        if os.path.exists(os.path.join(path, '{}.json'.format(name))):
            cur_time = time.strftime('%m%d%H%M%H%M%S', time.localtime(time.time()))
            name = name + '_' + cur_time
        out = os.path.join(path, '{}.json'.format(name))
        with open(out, "w") as f:
            json.dump(data, f)
        Logger.info("success save json: {}".format(out))
        
        
def save_image(fmt,im,img_path):
    cv2.imencode(".{}".format(fmt),im)[1].tofile(img_path)


def merge_two_list(list1, list2):
    """
    # zip
    """
    new_list = []
    assert len(list1) == len(list2), "2个list长度需一致"
    for index, item in enumerate(list1):
        new_list.append([list1[index], list2[index]])
    return new_list


def cal_area(pts_x, pts_y):
    pts_x = np.array(pts_x)
    pts_y = np.array(pts_y)
    return 0.5 * np.abs(pts_x.dot(np.roll(pts_y, 1)) - pts_y.dot(np.roll(pts_x, 1)))


def parse_region(region):
    assert isinstance(
        region, dict), "input type {} error,should be Dict".format(type(region))
    xs = region['shape_attributes']['all_points_x']
    ys = region['shape_attributes']['all_points_y']
    try:
        label = region['region_attributes']['regions']
    except:
        label = "-1"
        Logger.warning("有未标注缺陷类别")
    return xs, ys, label

def build_json(json_data,name,regions_list):
    json_data[name] = {}
    json_data[name]["filename"] = name
    json_data[name]["regions"] = regions_list
    return json_data
  
  
def scale_small_img(im_size,points,crop_size=120):
    """
    (im_h,im_w)
    (xs,ys)
    """
    im_h,im_w = im_size
    xs,ys = points
    w,h = max(xs)-min(xs),max(ys)-min(ys)
    scale = max(w,h)//crop_size + 1
    cx,cy = (max(xs)+min(xs))//2,(max(ys)+min(ys))//2
    start_x,end_x = int(cx - scale*crop_size/2),int(cx + scale*crop_size/2)
    start_y,end_y = int(cy - scale*crop_size/2),int(cy + scale*crop_size/2)
    if start_x<0:
        start_x = 0
        end_x = int(scale*crop_size)
    if start_x>im_w:
        start_x = int(im_w-scale*crop_size)
        end_x = im_w
    if start_y<0:
        start_y = 0
        end_y = int(scale*crop_size)
    if start_y>im_h:
        start_y = int(im_h-scale*crop_size)
        end_y = im_h
    return start_x,end_x,start_y,end_y
  
  
def keep_small_img(im_size,points,offset=20):
    """
    (im_h,im_w)
    (xs,ys)
    """
    im_h,im_w = im_size
    xs,ys = points
    start_x,end_x,start_y,end_y = min(xs)-offset,max(xs)+offset,min(ys)-offset,max(ys)+offset
    if start_x<0:
        end_x = end_x-start_x
        start_x = 0
    if start_x>im_w:
        end_x = im_w
        start_x = start_x-(end_x-im_w)
    if start_y<0:
        end_y = end_y-start_y
        start_y = 0
    if start_y>im_h:
        end_y = im_h
        start_y = start_y-(end_y-im_h)
    return start_x,end_x,start_y,end_y


def read_as_str(path):
    if path:
        with open(path,"r",encoding="utf-8") as f:
            str_data = f.read()
        return str_data
    else:
        return None
    


