import datetime
import cv2
import os
import numpy as np
from tqdm import tqdm
from dpp.common.util import cal_area
from dpp.common.dpp_json import DppJson


def cal_mean_std(imgs, channel):
    means_list, std_list = [0.]*channel, [0.]*channel
    for img in tqdm(imgs):
        img = cv2.imread(img, channel//3)
        for i in range(channel):
            means_list[i] += img[:,:,i].mean()
            std_list[i] += img[:,:,i].std()
    means = np.asarray(means_list) / len(imgs)
    std = np.asarray(std_list) / len(imgs)
    print("image length : {}".format(len(imgs)))
    print('cfg.MODEL.PIXEL_MEAN = {}'.format([round(item,2) for item in means.tolist()]))
    print('cfg.MODEL.PIXEL_STD = {}'.format([round(item,2) for item in std.tolist()]))


def get_segmenation(coord_x, coord_y):
    seg = []
    for x, y in zip(coord_x, coord_y):
        seg.append(x)
        seg.append(y)
    return [seg]


def create_annotation_info(annotation_id, image_id, category_id, is_crowd,
                           area, bounding_box, segmentation):
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area,  # float
        "bbox": bounding_box,  # [x,y,width,height]
        "segmentation": segmentation  # [polygon]
    }
    return annotation_info


def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[1],
        "height": image_size[0],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }

    return image_info


def convert_coco(img_path, json_data):
    '''
    :param imgdir: directory for your images
    :param annpath: path for your annotations
    :return: coco_output is a dictionary of coco style which you could dump it into a json file
    as for keywords 'info','licenses','categories',you should modify them manually
    '''
    coco_output = {}
    coco_output['info'] = {
        "description": "Example Dataset",
        "url": "https://github.com/waspinator/pycococreator",
        "version": "0.1.0",
        "year": 2018,
        "contributor": "waspinator",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }
    coco_output['licenses'] = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]
    coco_output['categories'] = []
    category = list(sorted(DppJson(json_data).labels_dict.keys()))
    for cate in category:
        tmp = {}
        tmp['id'] = int(cate)
        tmp['name'] = cate
        tmp['supercategory'] = 'phone'
        coco_output['categories'].append(tmp)
    coco_output['images'] = []
    coco_output['annotations'] = []

    # annotations id start from zero
    ann_id = 0
    # in VIA annotations, keys are image name
    count = 0
    for img_id, key in enumerate(tqdm(json_data.keys())):
        filename = json_data[key]['filename']
        try:
            img = cv2.imread(os.path.join(img_path, filename))
            # make image info and storage it in coco_output['images']
            image_info = create_image_info(
                img_id, os.path.basename(filename), img.shape[:2])
        except:
            print("error image:", filename)
        coco_output['images'].append(image_info)
        regions = json_data[key]["regions"]
        # for one image ,there are many regions,they share the same img id
        for region in regions:
            cat = region['region_attributes']['regions']
            iscrowd = 0
            points_x = region['shape_attributes']['all_points_x']
            points_y = region['shape_attributes']['all_points_y']
            area = cal_area(points_x, points_y)
            min_x = min(points_x)
            max_x = max(points_x)
            min_y = min(points_y)
            max_y = max(points_y)
            box = [min_x, min_y, max_x-min_x, max_y-min_y]
            segmentation = get_segmenation(points_x, points_y)
            # make annotations info and storage it in coco_output['annotations']
            ann_info = create_annotation_info(
                ann_id, img_id, int(cat), iscrowd, area, box, segmentation)
            coco_output['annotations'].append(ann_info)
            ann_id = ann_id + 1

    return coco_output
