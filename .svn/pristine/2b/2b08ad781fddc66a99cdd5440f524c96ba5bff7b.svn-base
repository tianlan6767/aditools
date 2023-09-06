import cv2
import os,copy,shutil,glob
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
from dpp.common.util import *
from dpp.common.mylog import Logger
from dpp.common.file import DppFile
from dpp.common.dpp_json import DppJson
from dpp.dataset.preprocessing.json_operate import merge_json
from dpp.dataset.visualization.json_draw import draw_polygon,ColorMap
from dpp.dataset.preprocessing.json_operate import split_json_by_folder


def crop_small_img(img_path, json_data, dst, scale,offset,crop_size):
    annotations = json_data.values()
    for annotation in tqdm(annotations):
        filename = DppFile.filename(annotation["filename"])
        fmt = DppFile.fmt(annotation["filename"])
        im = cv2.imread(os.path.join(img_path, filename), 1)
        regions = annotation['regions']
        for i, region in enumerate(regions):
            im_draw = im.copy()
            xs,ys,label = parse_region(region)     
            if scale: 
              start_x,end_x,start_y,end_y = scale_small_img(im.shape[:2],(xs,ys),crop_size=crop_size)  
            else:
              start_x,end_x,start_y,end_y = keep_small_img(im.shape[:2],(xs,ys),offset)  
            img_region_fn = filename.split('.')[0] + '_{}'.format(i) + '.'+ fmt
            im_region = im[start_y:end_y, start_x:end_x]
            
            draw_polygon(im_draw, [xs,ys], color=ColorMap["GREEN"].value, thickness=2)
            im_mask_region = im_draw[start_y:end_y, start_x:end_x]
            new_im = np.hstack([im_region,im_mask_region])
            label_path = os.path.join(dst, str(label))
            make_dir(label_path)
            cv2.imwrite(os.path.join(label_path, img_region_fn), new_im)


def crop_small_img_json(img_path, json_data, dst, scale,offset,crop_size):
    new_json = {}
    annotations = json_data.values()
    for annotation in tqdm(annotations):
        filename = DppFile.filename(annotation["filename"])
        fmt = DppFile.fmt(annotation["filename"])
        im = cv2.imread(os.path.join(img_path, filename), 1)
        regions = annotation['regions']
        for i, region in enumerate(regions):
            xs,ys,label = parse_region(region)     
            if scale: 
              start_x,end_x,start_y,end_y = scale_small_img(im.shape[:2],(xs,ys),crop_size=crop_size)  
            else:
              start_x,end_x,start_y,end_y = keep_small_img(im.shape[:2],(xs,ys),offset)  
            img_region_fn = filename.split('.')[0] + '_{}'.format(i) + '.'+ fmt          
            new_xs = [item-start_x for item in xs]
            new_ys = [item-start_y for item in ys]
            regions_list = [{'shape_attributes': {'all_points_x': new_xs, 'all_points_y': new_ys}, 'region_attributes': {'regions':label}}]
            new_json = build_json(new_json,img_region_fn,regions_list)
            new_im = im[start_y:end_y, start_x:end_x]
            label_path = os.path.join(dst, str(label))
            make_dir(label_path)
            cv2.imwrite(os.path.join(label_path, img_region_fn), new_im)
    return new_json
            
            

def classify_json(img_path,json_data):
    copy_data = copy.deepcopy(json_data)
    new_json = {}
    for folder in tqdm(os.listdir(img_path)):
        label_folder = os.path.join(img_path,folder)
        imgs = load_file(label_folder, format="img")
        for img in imgs:
            filename = os.path.basename(DppFile.recover_fn(img))
            index = DppFile.getIndex(img)
            regions = copy_data[filename]['regions'][int(index)]
            if filename not in new_json.keys():
                new_json[filename] = {}
                new_json[filename]["filename"] = filename
                new_json[filename]["regions"] = []
            
            regions['region_attributes'].update({'regions':str(folder)})
            new_json[filename]["regions"].append(regions)
            # regions[int(index)]['region_attributes'].update({'regions':str(folder)})
    return new_json


def multi_classify_json(folder_path):
    for item in os.listdir(folder_path):
        if item.endswith("_classify"):
            json_data = read_json(glob.glob(item.replace(item,"_classify")+"/*.json")[0])
            classify_json(item,json_data)


def image_partition(imgs,path,ratio,seed):
    random.seed(seed)
    random.shuffle(imgs)
    splice = max(int(round(len(imgs)*ratio,0)),1)
    train_path = os.path.join(os.path.dirname(path),"train")
    make_dir(train_path)
    test_path = os.path.join(os.path.dirname(path),"test")
    make_dir(test_path)
    for item in imgs[splice:]:
        shutil.move(os.path.join(path,item),os.path.join(train_path,item))
    for item in imgs[:splice]:
        shutil.move(os.path.join(path,item),os.path.join(test_path,item))

def dataset_partition(json_data,img_path,repeat,ratio,seed):
    dj = DppJson(json_data)
    labels = [item[0] for item in sorted(dj.labels_dict.items(), key=lambda x: x[1])]# list(dj.labels_dict.keys())
    labels_list = dj.new_json["regions"]
    filenames_list = dj.new_json["filename"]
    all_filenames = list(json_data.keys())

    for label in labels:
        begin = 0
        per_label_list = []
        # 获取label的索引
        for i in range(dj.labels_dict[label]):
            per_label_list.append(labels_list.index(label,begin,len(labels_list)))
            begin = labels_list.index(label,begin,len(labels_list)) + 1
        # 计算该每张图片中当前label个数
        filenames_counter = Counter([filenames_list[item] for item in per_label_list])
        multi_ng,repeat_ng,single_ng = [] ,[] ,[]    
        for item in filenames_counter.keys():
            if item in all_filenames:
              count = filenames_counter[item]
              if count>2*repeat:
                  multi_ng.append(item)
              elif count>repeat:
                  repeat_ng.append(item)
              else:
                  single_ng.append(item)
              all_filenames.remove(item)
            else:
              pass
        image_partition(multi_ng,img_path,ratio,seed)
        image_partition(repeat_ng,img_path,ratio,seed)
        image_partition(single_ng,img_path,ratio,seed)
        
    image_partition(dj.new_json["ok"],img_path,ratio,seed)       
      
  
def split_img_by_station(imgs):
    img_path = os.path.dirname(imgs[0])
    error_path = os.path.join(img_path,"error")
    make_dir(error_path)
    for item in tqdm(imgs):
      filename = os.path.basename(item)
      try:
        back = filename.split("-")[1]
        station_no,point_no = back.split("_")[0],back.split("_")[1]
        folder_path = os.path.join(img_path,station_no+"_"+point_no)
        make_dir(folder_path)
        shutil.move(item,os.path.join(folder_path,filename))
      except: # 未找到工位信息图片存放至error文件夹
        shutil.move(item,os.path.join(error_path,filename))
    

def label_partition_by_station(folder_path,json_data, scale,offset,crop_size):
    split_json_by_folder(folder_path,json_data)
    for item in os.listdir(folder_path):
        second_path = os.path.join(folder_path,str(item))
        dst = os.path.join(folder_path,str(item)+"_classify")
        if os.path.isdir(second_path):
            make_dir(dst)
            second_json_data = read_json(glob.glob(second_path+"/*.json")[0])
            crop_small_img(second_path, second_json_data, dst, scale,offset,crop_size)

   
def dataset_partition_by_station(folder_path,repeat,ratio,seed):
    origin_path_list = []
    new_json_list = []
    for item in os.listdir(folder_path):
        if item.endswith("_classify"):
            station_path = os.path.join(folder_path,str(item))
            origin_path = station_path.replace("_classify","")
            origin_path_list.append(origin_path)
            json_data = read_json(glob.glob(station_path.replace("_classify","")+"/*.json")[0])
            new_json = classify_json(station_path,json_data)
            new_json_list.append(new_json)
            save_json(new_json, station_path, item) 
            dataset_partition(new_json,origin_path,repeat,ratio,seed)
    return merge_json(new_json_list)