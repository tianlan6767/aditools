import os,cv2,math
import json
import shutil
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from dpp.common.mylog import Logger
from dpp.common.dpp_json import DppJson
from dpp.common.util import *
from dpp.dataset.preprocessing.json_convert import via_to_json,json_to_via


def format_json(data):
    """
    1、去除 jf 中的数字后缀
    """
    format_data = {}
    for key in data:
        file_name = data[key]['filename']
        format_data[file_name] = data[key]

        if 'file_attributes' in format_data[file_name]:
            format_data[file_name].pop('file_attributes')
        if 'size' in format_data[file_name]:
            format_data[file_name].pop('size')
        if 'regions' in format_data[file_name]:
            regions = format_data[file_name]['regions']
            for region in regions:
                if 'name' in region['shape_attributes']:
                    region['shape_attributes'].pop('name')
                if 'score' in region['region_attributes']:
                    region['region_attributes'].pop('score')
                if 'fpn_levels' in region['region_attributes']:
                    region['region_attributes'].pop('fpn_levels')


def merge_json(json_data):
    merge_data = {}
    for data in json_data:
        for key in data:
            if key not in merge_data:
                merge_data[key] = data[key]
            else:
                for region in data[key]['regions']:
                    merge_data[key]['regions'].append(region)
    return merge_data


def merge_json_img(img_path, dst):
    all_files = []
    all_json = []
    for root, dirs, files in os.walk(img_path):
        if files:
            all_files.append([os.path.join(root, file) for file in files])
    for item in flatten(all_files):
        if item.endswith(".bmp") or item.endswith(".jpg"):
            shutil.move(item, os.path.join(dst, os.path.basename(item)))
        elif item.endswith(".json"):
            all_json.append(json.load(open(item)))
        else:
            pass
    if len(all_json):
        new_json = merge_json(all_json)
        save_json(new_json, dst, 'all_merge')


def filter_json(json_data, imgs):
    filter_dict = {}
    for img in imgs:
        filename = os.path.basename(img)
        filter_dict[filename] = json_data[filename]
        json_data.pop(filename)
    return filter_dict, json_data


def add_ok_json(imgs):
    new_json = {}
    for img in imgs:
        filename = os.path.basename(img)
        new_json[filename] = {}
        new_json[filename]["filename"] = filename
        new_json[filename]["regions"] = []
    return new_json


def del_empty_key(json_data):
    [json_data.pop(value) for value in list(json_data.keys()) if len(json_data[value]['regions']) == 0]
    return json_data


def json_cover(old_json_data, new_json_data):
    old_dj = DppJson(old_json_data)
    if old_dj.json_format.startswith('VIA'):
        old_json_data = via_to_json(old_json_data)
    new_dj = DppJson(new_json_data)
    if new_dj.json_format.startswith('VIA'):
        new_json_data = via_to_json(new_json_data)
    for key in new_json_data:
        old_json_data[key] = new_json_data[key]
    return old_json_data


def del_small_area(json_data, min_area):
    dataset = deepcopy(json_data)
    for key in list(dataset.keys()):
        regions = dataset[key]['regions']
        for index, region in enumerate(regions):
            area = cal_area(region['shape_attributes']['all_points_x'],
                            region['shape_attributes']['all_points_y'])
            if area <= min_area:
                regions.pop(index)
    return dataset


def filter_labels(json_data, labels):
    old_dj = DppJson(json_data)
    if "-1" in old_dj.labels_dict.keys():
        Logger.warning("有未分类标注")
    dataset = deepcopy(json_data)
    for key, value in dataset.items():
        regions = value['regions']
        regions_list = []
        for label in labels:
            region = list(filter(
                lambda region: region['region_attributes']['regions'] == str(label), regions))
            regions_list.extend(region)
        dataset[key]['regions'] = regions_list
    return dataset


def point_filter(x,min_value,max_value):
    result_list = []
    index_list=[]
    for index,item in enumerate(x):
        if item <max_value and item>min_value:
            result_list.append(item)
            index_list.append(index)
    return result_list,index_list

def split_limit_mask(img_path,json_data,limit_rate=12):
    add_json = {}
    for key,value in tqdm(json_data.items()):
        origin_im = cv2.imread(os.path.join(img_path,key),0)
        regions = value["regions"]
        add_json[key] = {}
        add_json[key]["filename"] = key
        # 标注加点
        jf_im = np.zeros(origin_im.shape, np.uint8)
        for region in regions:
            xs,ys,label = parse_region(region)
            counter = list(zip(xs, ys))
            cv2.fillPoly(jf_im, [np.array(counter)], int(label))
        contours, _ = cv2.findContours(jf_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 标注分段
        add_regions=[]
        for contour in contours:
            polygons = contour.flatten().tolist()
            xs, ys = polygons[0::2], polygons[1::2]
            label = str(jf_im[ys[0]][xs[0]])
            
            x = max(xs) - min(xs)
            y = max(ys) - min(ys)
            rate = max(x,y)/min(x,y)
            num = math.ceil(rate/limit_rate)
            if rate>limit_rate:
                if y>x: #竖直
                    each_y = math.ceil((max(ys) - min(ys))/num)
                    each_y_list = list(range(min(ys),max(ys),each_y))
                    each_y_list.append(max(ys))
                    for index,item in enumerate(range(num)):
                        ys_list,index_list = point_filter(ys,each_y_list[index],each_y_list[index+1])
                        xs_list = [xs[i] for i in index_list]         
                        new_dict = {'shape_attributes':{'all_points_x':xs_list,'all_points_y':ys_list},
                                'region_attributes':{'regions':label}}
                        add_regions.append(new_dict)
                else: # 水平
                    each_x = int((max(xs) - min(xs))/num)
                    each_x_list = list(range(min(xs),max(xs),each_x))
                    each_x_list.append(max(xs))
                    for index,item in enumerate(range(num)):
                        xs_list,index_list = point_filter(xs,each_x_list[index],each_x_list[index+1])
                        ys_list = [ys[i] for i in index_list]
                        new_dict = {'shape_attributes':{'all_points_x':xs_list,'all_points_y':ys_list},
                                'region_attributes':{'regions':label}}
                        add_regions.append(new_dict)
            else:
                new_dict = {'shape_attributes':{'all_points_x':xs,'all_points_y':ys},
                                'region_attributes':{'regions':label}}
                add_regions.append(new_dict)
        add_json[key]["regions"] = add_regions
    return add_json
  
  
def split_json_by_folder(folder_path,json_data):
    """
    根据文件夹切分json
    """
    if len(json_data):
        folders = os.listdir(folder_path)
        for item in folders:
            second_folder_path = os.path.join(folder_path,item)
            if os.path.isdir(second_folder_path):
                imgs = load_file(second_folder_path, format="img")
                filter_data, remain_data = filter_json(json_data,imgs)
                save_json(filter_data, second_folder_path, str(item))
                
                
def copy_via(json_data,imgs):
    assert len(json_data)==1,"确保只有1张图片的json"
    old_dj = DppJson(json_data)
    if old_dj.json_format.startswith('VIA'):
        json_data = via_to_json(json_data)
    new_json = {}
    for img in imgs:
        filename = os.path.basename(img)
        new_json[filename] = {}
        new_json[filename]["filename"] = filename
        new_json[filename]["regions"] = list(json_data.values())[0]["regions"]
    if old_dj.json_format.startswith('VIA'):
        new_json = json_to_via(imgs, new_json)
    return new_json
  
  
def match_point_json(json_data,imgs):
    from dpp.dataset.transforms.segment import filename_mapper
    old_dj = DppJson(json_data)
    if old_dj.json_format.startswith('VIA'):
        json_data = via_to_json(json_data)
    new_json = {}
    for img in imgs:
        filename = os.path.basename(img)
        new_name = filename_mapper(filename)+".{}".format(filename.split(".")[-1])
        new_json[filename] = {}
        new_json[filename]["filename"] = filename
        try:
            new_json[filename]["regions"] = json_data[new_name]["regions"]
        except:
            Logger.error("点位提取失败，请检查图片名 {},或检查映射json有无该点位标注".format(filename))
    if old_dj.json_format.startswith('VIA'):
        new_json = json_to_via(imgs, new_json)
    return new_json