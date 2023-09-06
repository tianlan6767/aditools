import cv2
import os
import copy 
import random
import numpy as np
from tqdm import tqdm
from enum import Enum, unique
from dpp.common.dpp_json import DppJson
from dpp.common.util import cal_area,parse_region
from dpp.dataset.preprocessing.json_operate import filter_labels,del_empty_key
from dpp.dataset.preprocessing.json_convert import via_to_json


@unique
class ColorMap(Enum):
    BLUE = (255, 0, 0)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)


def draw_polygon(im, pts, color=ColorMap["GREEN"].value, thickness=2,mask_type="json"):
    if mask_type == "inf":
        xs,ys = pts
        step_list = []
        for index,x in enumerate(xs):
            try:
              step_list.append(xs[index+1]-xs[index])
            except:
              pass
        index_list = []
        for index,item in enumerate(step_list):
            if abs(item) > 5:
                index_list.append(index)
        """
        a = [1,2,5,6,8,15,16,17,32,33,33,34,44]
        step = [1,3,1,2,7,1,1,15,1,0,1,10]
        indexs = [4,7,11]
        """
        xs_list,ys_list = [],[]
        if len(index_list)==1:
            xs_list.extend([xs[:index_list[0]+1],xs[index_list[0]+1:]])
            ys_list.extend([ys[:index_list[0]+1],ys[index_list[0]+1:]])
            points = [np.dstack((xs_list[index], ys_list[index])) for index,item in enumerate(xs_list)]
        elif len(index_list)>1:  
            for index,item in enumerate(index_list):
                if index==0:
                    xs_list.append(xs[:index_list[index]+1])
                    ys_list.append(ys[:index_list[index]+1])
                elif index==len(index_list)-1:
                    xs_list.append(xs[index_list[index-1]+1:])
                    ys_list.append(ys[index_list[index-1]+1:])
                else:
                    xs_list.append(xs[index_list[index-1]+1:index_list[index]+1])
                    ys_list.append(ys[index_list[index-1]+1:index_list[index]+1])
            points = [np.dstack((xs_list[index], ys_list[index])) for index,item in enumerate(xs_list)]
        else:
            xs_list = xs
            ys_list = ys
            points = [np.dstack((xs_list, ys_list))]
    else:
        points = [np.dstack((pts[0], pts[1]))]
    cv2.polylines(img=im, pts=points, isClosed=True,
                  color=color, thickness=thickness)

def draw_rect(im, pts, color=ColorMap["GREEN"].value, thickness=2):
    cv2.rectangle(im, (min(pts[0]), min(pts[1])), (max(
        pts[0]), max(pts[1])), color=color, thickness=thickness)

def draw_text(im, value, xy, color, font_size=1,thickness=2):
    h, w = im.shape[:2]
    length = 250
    if len(xy[0])>4:
        xy = [[(min(xy[0])+max(xy[0]))//2+60,(min(xy[0])+max(xy[0]))//2+100],[(min(xy[1])+max(xy[1]))//2+60,(min(xy[1])+max(xy[1]))//2+100]]
    position_x = min(xy[0])-length//5 if min(xy[0]) > w-length else min(xy[0])-5
    position_y = min(xy[1])-length//5 if min(xy[1]) > h-length else min(xy[1])-5
    position_x = max(xy[0]) if position_x < 50 else position_x
    position_y = max(xy[1]) if position_y < 50 else position_y
    cv2.putText(im, value, (position_x, position_y),cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
    
    
def draw_regions(im,regions,draw_cfg):
    for region in regions:
        xs,ys,label = parse_region(region)
        if draw_cfg["shape"] == "RECT":
          draw_rect(im,[xs,ys],color=draw_cfg["color"])
        else:
          draw_polygon(im,[xs,ys],color=draw_cfg["color"])
        value = ""
        if draw_cfg["label"]:
            value += label
        if draw_cfg["score"]:
            score_value = str(round(region['region_attributes']['score'], 2))
            value += " S:"+score_value
        if draw_cfg["area"]:
            area_value = str(int(cal_area(xs, ys)))
            value += " A:"+area_value    
        draw_text(im, value, [xs, ys], draw_cfg["color"],font_size=draw_cfg["font_size"])

def draw_mask(imgs, json_data_list, draw_cfg_list,dst,spotcheck=False):
    draw_json_data = []
    for json_data in json_data_list:
        dj = DppJson(json_data)
        if dj.json_format.startswith('VIA'):
            json_data = via_to_json(json_data)
        draw_json_data.append(json_data)
     
    # 一个json时默认值绘制有标注的图片
    if len(json_data_list) == 1 and spotcheck==False:
        for key,value in tqdm(draw_json_data[0].items()):
            filename_path = os.path.dirname(imgs[0])
            im = cv2.imread(os.path.join(os.path.join(filename_path,key)), 1)
            if len(value["regions"]) > 0:
                draw_regions(im,value["regions"],draw_cfg_list[0])
                cv2.imwrite(os.path.join(dst, key.replace(".bmp", ".jpg")), im)

    else:  # 两个及以上json时需要绘制全部图片;
        if spotcheck:  # 抽查时默认选取绘制50张
            random.shuffle(imgs)
            imgs = imgs[:50]
        for img in tqdm(imgs):
            im = cv2.imread(img, 1)
            filename = os.path.basename(img)
            for index,json_data in enumerate(draw_json_data):
                regions = json_data[filename]["regions"]
                draw_regions(im,regions,draw_cfg_list[index])
                
            cv2.imwrite(os.path.join(dst, filename.replace(".bmp", ".jpg")), im)
        
    
    
  
    # for index,json_data in enumerate(json_data_list):
    #     draw_cfg = copy.deepcopy(draw)
    #     jd = DppJson(json_data)
    #     if jd.json_format == "JSON":
    #         draw_cfg["color"] = ColorMap["GREEN"].value
    #         draw_cfg["mask_type"] = "json"
    #         if len(json_data_list)>1:
    #             draw_cfg["shape"] = "RECT"
    #             draw_cfg["label"] = False
    #     elif jd.json_format == "INF":
    #         draw_cfg["color"] = ColorMap["RED"].value
    #         draw_cfg["score"] = True
    #         draw_cfg["mask_type"] = "inf"
    #         if len(json_data_list)>1:
    #             draw_cfg["shape"] = "POLYGON"
    #     else: # jd.json_format.startswith("VIA")
    #         json_data = via_to_json(json_data)
    #         draw_cfg["shape"] = "RECT"
    #         draw_cfg["color"] = ColorMap["BLUE"].value
    #         draw_cfg["mask_type"] = "json"
    #     if draw["color"]:
    #         draw_cfg["color"] = draw["color"]
    #     if draw_cfg["classid"] and jd.json_format != "INF":
    #         json_data = filter_labels(json_data, draw_cfg["classid"])
    #         draw_cfg["mask_type"] = "filter_label"
    #     draw_json_data.append(json_data)
    #     draw_list.append(draw_cfg)
    # # 一个json文件时不绘制OK图,
    # if len(json_data_list)==1:
    #     json_data = del_empty_key(draw_json_data[0])
    #     draw_imgs = list(json_data.keys())
    #     imgs = [os.path.join(os.path.dirname(imgs[0]),img) for img in draw_imgs]
    # else:   # 多个文件时如果绘制某个标签时选择非推理文件图片数
    #     for index,item in enumerate(draw_list):
    #         if item["mask_type"]=="filter_label":
    #             json_data = del_empty_key(draw_json_data[index])
    #             draw_imgs = list(json_data.keys())
    #             imgs = [os.path.join(os.path.dirname(imgs[0]),img) for img in draw_imgs]
    #             break
            
        

    # # 抽查功能   
    # if spotcheck and len(imgs)>50:
    #     random.shuffle(imgs)
    #     imgs = imgs[:50]
      
    # for img in tqdm(imgs):
    #     im = cv2.imread(img, 1)
    #     filename = os.path.basename(img)
    #     for index,json_data in enumerate(draw_json_data):
    #         regions = json_data[filename]["regions"]
    #         for region in regions:
    #             xs,ys,label = parse_region(region)
    #             if draw_list[index]["shape"] == "RECT":
    #               draw_rect(im,[xs,ys],color=draw_list[index]["color"])
    #             else:
    #               draw_polygon(im,[xs,ys],color=draw_list[index]["color"],mask_type=draw_list[index]["mask_type"])
    #             value = ""
    #             if draw_list[index]["label"]:
    #                 value += label
    #             if draw_list[index]["score"]:
    #                 score_value = str(round(region['region_attributes']['score'], 2))
    #                 value += " S:"+score_value
    #             if draw_list[index]["area"]:
    #                 area_value = str(int(cal_area(xs, ys)))
    #                 value += " A:"+area_value    
    #             draw_text(im, value, [xs, ys], draw_list[index]["color"],font_size=draw_list[index]["font_size"])
    #     cv2.imwrite(os.path.join(dst, filename.replace(".bmp", ".jpg")), im)