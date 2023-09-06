import os
import cv2
import random
from tqdm import tqdm
from collections import Counter
from dpp.dataset.transforms.segment import SEGMENT_RULE_REGISTRY, crop_box
from dpp.dataset.transforms.merge import MERGE_RULE_REGISTRY
from dpp.dataset.transforms.augmentation import *
from dpp.common.util import build_json


def merge_params(value):
    """
    裁剪拼接规则
    """
    if value == 4:
        row_num, col_num = [2, 2]
    elif value == 2:
        row_num, col_num = [1, 2]
    elif value == 8:                #
        row_num, col_num = [2, 4]   # 4096*8192=>2048*2048
    elif value == 16:
        row_num, col_num = [4, 4]   # 4096*4096=>1024*1024
    elif value == 32:
        row_num, col_num = [4, 8]   # 4096*8192=>1024*1024
    elif value == 64:
        row_num, col_num = [8, 8]   # 2048*2048=>256*256
    else:
        row_num, col_num = [1, 1]
    return row_num, col_num

def segment(name, imgs, json_data, dst):
    sr = SEGMENT_RULE_REGISTRY.get(name)()
    new_json = {}
    for img in tqdm(imgs):
        filename = os.path.basename(img)
        im = cv2.imread(img, -1)
        box_list = sr.crop_im({filename: im})
        if len(json_data) != 0:
            regions = json_data[filename]["regions"]
            points = sr.crop_polygon(im, box_list, regions)
        else:
            points = sr.crop_polygon(im, box_list, [])
        for index, item in enumerate(box_list):
            crop_filename = filename.replace(
                ".bmp", "")+"_s{}.bmp".format(index)
            new_im = crop_box(item, im)         
            new_json = build_json(new_json,crop_filename,points[index])
            cv2.imwrite(os.path.join(dst, filename.replace(
                ".bmp", "")+"_s{}.bmp".format(index)), new_im)
    return new_json


def merge(name, imgs, json_data, dst,size):
    mr = MERGE_RULE_REGISTRY.get(name)(size=size)
    new_json = {}
    filenames = [os.path.basename(img) for img in imgs]
    img_path = os.path.dirname(imgs[0])
    new_json = {}
    if name == "AvgMerge":
        fn_heads = [fn.split("_s")[0] for fn in filenames]
        fn_dict = Counter(fn_heads)  # Counter({fn: 8, ...)
        bar = tqdm(fn_dict)
        bar.set_description("开始合并原图>>>>>>>>")
        for item in bar:
            number = fn_dict[item]
            row_num, col_num = merge_params(number)
            imgs = [os.path.join(img_path, item+"_s{}.bmp".format(str(i)))
                    for i in range(number)]
            ims = np.array([cv2.imread(img, -1) for img in imgs])
            if len(ims.shape) == 3:
              ims = np.expand_dims(ims, axis=1)
            ims_array = np.split(ims,row_num)
            new_im = mr.merge_im(ims_array)
            if len(json_data):
              regions = [json_data[os.path.basename(img)]["regions"] for img in imgs]
              new_regions = mr.merge_polygon(ims_array,regions)
            else:
              new_regions = []
            out_name = item+".bmp"
            new_json = build_json(new_json,out_name,new_regions)
            cv2.imwrite(os.path.join(dst, item+".bmp"), new_im)
    else:
        random.shuffle(imgs)
        while (len(imgs)):
            merge_size = 0
            ims_list = []
            imgs_list = []
            while merge_size < size and len(imgs):
                img = imgs.pop(0)
                im = cv2.imread(img, -1)
                h, w = im.shape[:2]
                if len(im.shape) == 2:
                  im = np.expand_dims(im, axis=0)
                merge_size += w
                ims_list.append(im)
                imgs_list.append(img)
            w_list = [item.shape[-1] for item in ims_list]
            if sum(w_list) > size:
                imgs.append(img)
                ims_list.pop(-1)
                imgs_list.pop(-1)
            new_im = mr.merge_im(ims_list)
            if len(json_data):
              regions = [json_data[os.path.basename(item)]["regions"] for item in imgs_list]
              new_regions = mr.merge_polygon(ims_list,regions)
            else:
              new_regions = []
            out_name = os.path.basename(imgs_list[0]).split(
                ".")[0]+"_{}.bmp".format(dst.split("_")[-1])
            new_json = build_json(new_json,out_name,new_regions)
            cv2.imwrite(os.path.join(dst, out_name), new_im)
    return new_json


def img_aug(imgs, json_data, dst, tfms):
    new_json = {}
    for img in imgs:
        fn = os.path.basename(img)
        name, fmt = fn.split(".")
        out_name = "{}_{}.{}".format(name, dst.split("_")[-1], fmt)
        im = cv2.imread(img, 0)
        regions = json_data[fn]['regions']
        polygons = [list(zip(i['shape_attributes']['all_points_x'], i['shape_attributes']['all_points_y'])) for i in
                    regions]
        tf = random.choice(tfms)
        new_im = tf.apply_im(im)
        p = tf.apply_polygons(polygons)
        regions_list = []
        for index, region in enumerate(regions):
            region_dict = {'shape_attributes': {'all_points_x': p[index][:, 0].tolist(
            ), 'all_points_y': p[index][:, 1].tolist()}, 'region_attributes': region["region_attributes"]["regions"]}
            regions_list.append(region_dict)
        new_json = build_json(new_json,out_name,regions_list)
        cv2.imwrite(os.path.join(dst, out_name), new_im)
    return new_json
