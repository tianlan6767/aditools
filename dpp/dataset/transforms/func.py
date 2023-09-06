import os
import cv2
import random
from tqdm import tqdm
from collections import Counter
from dpp.common.util import build_json,save_image,load_file
from dpp.dataset.transforms.augmentation import *
from dpp.dataset.transforms.pad import pad_im
from dpp.dataset.transforms.segment import SEGMENT_RULE_REGISTRY, crop_box
from dpp.dataset.transforms.merge import MERGE_RULE_REGISTRY,merge_params
from dpp.common.mylog import Logger
from dpp.common.file import DppFile


class SegmentDispatch:
    def __init__(self,name, imgs, json_data, dst):
        self.dst = dst
        self.name = name
        self.imgs = imgs
        self.json_data = json_data
        self.sr = SEGMENT_RULE_REGISTRY.get(name)()
  
    def __call__(self):
        """
        box_list: [tuple(x1,y1,x2,y2),...] 
        """
        new_json = {}
        for img in tqdm(self.imgs):
            filename = DppFile.filename(img)
            fmt = DppFile.fmt(filename)
            try:
              im = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
            except:
              print(img)
            box_list = self.sr.crop_im({filename.split(".")[0]: im})
            if len(self.json_data):
                regions = self.json_data[filename]["regions"]
                points = self.sr.crop_polygon(im, box_list, regions)
            else:
                points = self.sr.crop_polygon(im, box_list, [])
            
            for index, item in enumerate(box_list):
                crop_filename = DppFile.seg_filename(index,filename)
                new_json = build_json(new_json,crop_filename,points[index])
                new_im = getattr(self,self.name.lower())(item,im)
                save_image(fmt,new_im,os.path.join(self.dst,crop_filename))                
        return new_json
  
    def avgseg(self,item,im):
        return crop_box(item, im,pad_size=self.sr.cfg["crop_size"],extra=self.sr.cfg["extra"]) 
             
    def jsonseg(self,item,im):
        return crop_box(item, im)  
      
    def ceseg(self,item,im):
        return crop_box(item, im)  
      
    def cv2seg(self,item,im):
        try:
            return crop_box(item, im)  
        except:
            Logger.warning("未检测到边缘")
            return im
      
    def threeseg(self,item,im):
        return crop_box(item, im)  
      

class MergeDispatch:
    def __init__(self,name, img_path, json_data, dst,size):
        self.mr = MERGE_RULE_REGISTRY.get(name)(size=size)
        self.imgs = load_file(img_path, format="img")
        self.name = name
        self.img_path = img_path
        self.json_data = json_data
        self.dst = dst
        self.size = size
  
    def __call__(self):
        new_json = getattr(self,self.name.lower())()
        return new_json

    def avgmerge(self):
        """
        需要后缀为 _s+小图序号 
        """
        new_json = {}
        filenames = [os.path.basename(img) for img in self.imgs]
        fn_heads = [fn.split("_s")[0] for fn in filenames]
        fn_dict = Counter(fn_heads)  # Counter({fn: 8, ...)
        bar = tqdm(fn_dict.items())
        bar.set_description("开始合并原图>>>>>>>>")
        for index,(item,number) in enumerate(bar):
            row_num, col_num = merge_params(number)
            fmt = DppFile.fmt(filenames[index*number])
            imgs = [os.path.join(self.img_path, item+"_s{}.{}".format(str(i),fmt))for i in range(number)]
            ims = np.array([cv2.imread(img, -1) for img in imgs]) # (BHWC)
            if len(ims.shape) == 3:
                ims = np.expand_dims(ims, axis=-1)
            ims_array = np.split(ims,row_num)
            new_im = self.mr.merge_im(ims_array)
            if len(self.json_data):
                regions = [self.json_data[os.path.basename(img)]["regions"] for img in imgs]
                new_regions = self.mr.merge_polygon(ims_array,regions)
            else:
                new_regions = []
            out_name = "{}.{}".format(item,fmt)
            new_json = build_json(new_json,out_name,new_regions)
            save_image(fmt,new_im,os.path.join(self.dst, out_name))
        return new_json
      
      
    def jsonmerge(self):
        new_json = {}
        filenames = [os.path.basename(img) for img in self.imgs]
        fmt = filenames[0].split(".")[-1]
        merge_data = self.mr.merge_data
        """
        {'master_1_1.jpg':{'images': ['1_1_1_s0'], 'bbox': [[...]]},'master_1_2.jpg':{'images': ['1_1_1_s1'], 'bbox': [[...]]} ...}
        """
        for key,value in tqdm(merge_data.items()):
            ims_list = []
            sum_value = 0
            """
            一张拼接结果图可能有多张小图；尺寸小的填充黑图,尺寸大的再次裁剪
            """
            for index,item in enumerate(value["images"]):
                h1,w1,h2,w2 = value["bbox"][index]
                point_seg_name = [item.split("-")[-1].split(".")[0] for item in filenames]
                select_index = point_seg_name.index(item)
                filename_path = self.imgs[select_index]
                if os.path.isfile(filename_path):
                    im = cv2.imdecode(np.fromfile(filename_path, dtype=np.uint8), -1)
                    h,w = im.shape[:2]
                    if h< h2-h1:
                        im = pad_im(im,(h2-h1,w),method="below")
                    h,w = im.shape[:2]
                    if w< w2-w1:
                        im = pad_im(im,(h,w2-w1),method="below")
                    else:   # roi裁剪宽度过大，再裁剪
                        im = im[:,:w2-w1]
                else:
                    im = np.zeros((h2-h1,w2-w1))
                    sum_value+=1
                ims_list.append(im)
            if sum_value < len(value["images"]) and len(ims_list): # 补黑图
                new_im = self.mr.merge_im(ims_list)
                if new_im.shape[1] != self.size:
                    new_im = pad_im(new_im,(self.size,self.size),method="below")
                # 待完成
                # mr.merge_polygon(ims_list,regions)
                # cv2.imwrite(os.path.join(dst, os.path.basename(filename_path)), new_im)
                
                cv2.imencode(".{}".format(fmt),new_im)[1].tofile(os.path.join(self.dst,"_".join(key.split("_")[1:])))
        return new_json
      
    def zkmerge(self):
        new_json = {}
        random.shuffle(self.imgs)
        while (len(self.imgs)):
            merge_size = 0
            ims_list = []
            imgs_list = []
            while merge_size < self.size and len(self.imgs):
                img = self.imgs.pop(0)
                im = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
                fmt = os.path.basename(img).split(".")[-1]
                h, w = im.shape[:2]
                if len(im.shape) == 2:
                    im = im[:,:,None]
                merge_size += w
                ims_list.append(im)
                imgs_list.append(img)
            w_list = [item.shape[1] for item in ims_list]
            if sum(w_list) > self.size:
                self.imgs.append(img)
                ims_list.pop(-1)
                imgs_list.pop(-1)
            new_im = self.mr.merge_im(ims_list)
            if len(self.json_data):
                regions = [self.json_data[os.path.basename(item)]["regions"] for item in imgs_list]
                new_regions = self.mr.merge_polygon(ims_list,regions)
            else:
                new_regions = []
            out_name = os.path.basename(imgs_list[0]).split(
                ".")[0]+"_{}.{}".format(self.dst.split("_")[-1],fmt)
            new_json = build_json(new_json,out_name,new_regions)
            cv2.imencode(".{}".format(fmt),new_im)[1].tofile(os.path.join(self.dst,out_name))
        return new_json


def img_aug(imgs, json_data, dst, tfms):
    new_json = {}
    for img in tqdm(imgs):
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
            ), 'all_points_y': p[index][:, 1].tolist()}, 'region_attributes': {'regions':region["region_attributes"]["regions"]}}
            regions_list.append(region_dict)
        new_json = build_json(new_json,out_name,regions_list)
        cv2.imwrite(os.path.join(dst, out_name), new_im)
    return new_json


def img_pad(imgs,dst,pad_size):
    for img in tqdm(imgs):
        im = cv2.imread(img,-1)
        new_im = pad_im(im,pad_size=pad_size)
        cv2.imwrite(os.path.join(dst,os.path.basename(img)),new_im)
        
        
        
"""
def segment(name, imgs, json_data, dst):
    sr = SEGMENT_RULE_REGISTRY.get(name)()
    new_json = {}
    for img in tqdm(imgs):
        filename = os.path.basename(img)
        fmt = filename.split(".")[-1]
        im = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
        box_list = sr.crop_im({filename.split(".")[0]: im})
        if len(json_data) != 0:
            regions = json_data[filename]["regions"]
            points = sr.crop_polygon(im, box_list, regions)
        else:
            points = sr.crop_polygon(im, box_list, [])
        for index, item in enumerate(box_list):
            crop_filename = filename.split(".")[0]+"_s{}.{}".format(index,fmt)
            if name == "AvgSeg":
                new_im = crop_box(item, im,pad_size=sr.cfg["crop_size"],extra=sr.cfg["extra"])   
            else:
                new_im = crop_box(item, im)         
            new_json = build_json(new_json,crop_filename,points[index])
            try:
                cv2.imencode(".{}".format(fmt),new_im)[1].tofile(os.path.join(dst,crop_filename))
            except:
                if name == "Cv2Seg":
                    print("{}未检测到边缘".format(filename))
                # else:
                #     Logger.error("error")
    return new_json
    
    
def merge(name, imgs, json_data, dst,size):
    mr = MERGE_RULE_REGISTRY.get(name)(size=size)
    new_json = {}
    filenames = [os.path.basename(img) for img in imgs]
    img_path = os.path.dirname(imgs[0])
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
    elif name == "ZkMerge":
        random.shuffle(imgs)
        while (len(imgs)):
            merge_size = 0
            ims_list = []
            imgs_list = []
            while merge_size < size and len(imgs):
                img = imgs.pop(0)
                im = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
                fmt = os.path.basename(img).split(".")[-1]
                h, w = im.shape[:2]
                if len(im.shape) == 2:
                  im = im[:,:,None]
                merge_size += w
                ims_list.append(im)
                imgs_list.append(img)
            w_list = [item.shape[1] for item in ims_list]
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
                ".")[0]+"_{}.{}".format(dst.split("_")[-1],fmt)
            new_json = build_json(new_json,out_name,new_regions)
            cv2.imencode(".{}".format(fmt),new_im)[1].tofile(os.path.join(dst,out_name))
    else:
        fmt = filenames[0].split(".")[-1]
        merge_data = mr.merge_data
        # {'master_1_1.jpg':{'images': ['1_1_1_s0'], 'bbox': [[...]]},'master_1_2.jpg':{'images': ['1_1_1_s1'], 'bbox': [[...]]} ...}
        for key,value in tqdm(merge_data.items()):
            ims_list = []
            sum_value = 0
            #一张拼接结果图可能有多张小图；尺寸小的填充黑图,尺寸大的再次裁剪
            for index,item in enumerate(value["images"]):
                h1,w1,h2,w2 = value["bbox"][index]
                point_seg_name = [item.split("-")[-1].split(".")[0] for item in filenames]
                select_index = point_seg_name.index(item)
                filename_path = imgs[select_index]
                if os.path.isfile(filename_path):
                    im = cv2.imdecode(np.fromfile(filename_path, dtype=np.uint8), 0)
                    h,w = im.shape
                    if h< h2-h1:
                        im = pad_im(im,(h2-h1,w),method="below")
                    h,w = im.shape
                    if w< w2-w1:
                        im = pad_im(im,(h,w2-w1),method="below")
                    else:   # roi裁剪宽度过大，再裁剪
                        im = im[:,:w2-w1]
                else:
                    im = np.zeros((h2-h1,w2-w1))
                    sum_value+=1
                ims_list.append(im)
            if sum_value < len(value["images"]) and len(ims_list): # 补黑图
                new_im = mr.merge_im(ims_list)
                if new_im.shape[1]!=size:
                    new_im = pad_im(new_im,(size,size),method="below")
                # 待完成
                # mr.merge_polygon(ims_list,regions)
                # cv2.imwrite(os.path.join(dst, os.path.basename(filename_path)), new_im)
                
                cv2.imencode(".{}".format(fmt),new_im)[1].tofile(os.path.join(dst,"_".join(key.split("_")[1:])))
    return new_json
"""        
