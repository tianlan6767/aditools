import os
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from abc import abstractmethod
from dpp.dataset.transforms.segment import counter_to_polygon
from dpp.common.util import parse_region
from dpp.common.registry import Registry
from dpp.common.util import read_json


MERGE_RULE_REGISTRY = Registry("MERGE_RULE")

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
  

class Merge:
    def __init__(self,size=2048):
      self.size = size

    @abstractmethod
    def merge_im(self, ims):
        raise NotImplemented

    @abstractmethod
    def merge_polygon(self, ims, regions):
        raise NotImplemented


@MERGE_RULE_REGISTRY.register()
class AvgMerge(Merge):
    def merge_im(self, ims):
        row_num = len(ims) #2,4
        row_list = []
        for row in range(row_num):
            row_list.append(np.concatenate(ims[row],axis=1))
        new_im = np.concatenate(row_list,axis=0)
        return new_im

    def merge_polygon(self,ims, regions):
        row_num,col_num = len(ims),ims[0].shape[0]
        for index in range(len(regions)):
            per_regions = regions[index]
            jf_im = np.zeros(ims[index//col_num][index%col_num].shape, np.uint8)
            for region in per_regions:
                xs, ys, label = parse_region(region)
                counter = list(zip(xs, ys))
                cv2.fillPoly(jf_im, [np.array(counter)], int(label)*15)
            ims[index//col_num][index%col_num] = jf_im
        jf_full_im = self.merge_im(ims)
        if jf_full_im.shape[-1] == 3:
            jf_full_im = cv2.cvtColor(jf_full_im,cv2.COLOR_BGR2GRAY)
        else:
            jf_full_im = jf_full_im.squeeze(-1)
        polygons = counter_to_polygon(jf_full_im, min_area=5)  # jf_full_im : H*W
        return polygons


@MERGE_RULE_REGISTRY.register()
class ZkMerge(Merge):
    def merge_im(self, ims):
        """
        BGR : HWC => CHW
        """
        stack_im = np.concatenate(ims,axis=1)
        pad_w = self.size-stack_im.shape[1]
        if pad_w!=0:
            stack_im = np.concatenate([stack_im,np.zeros((stack_im.shape[0],pad_w,stack_im.shape[2]))],axis=1)
        return stack_im

    def merge_polygon(self,ims, regions):
        for index,per_regions in enumerate(regions):
            jf_im = np.zeros(ims[index].shape[:2], np.uint8)
            for region in per_regions:
                xs, ys, label = parse_region(region)
                counter = list(zip(xs, ys))
                cv2.fillPoly(jf_im, [np.array(counter)], int(label)*15)
            ims[index] = jf_im
        jf_full_im = self.merge_im([item[:,:,None] for item in ims])
        polygons = counter_to_polygon(jf_full_im.astype(np.uint8).squeeze(-1), min_area=20)
        return polygons


@MERGE_RULE_REGISTRY.register()
class JsonMerge(Merge):
    """
    多目录：量跑数据格式 1 -> ORIG -> 1_1_1.bmp ...
    单目录：1-1_1_1.bmp ......
    """
    def __init__(self,size=2048):
        self.size = size
        self.ya = yaml.load(open("./dpp/dataset/transforms/config.yaml"))
        self.cfg = self.ya[self.__class__.__name__]
        self.merge_data = self.parse_json()
      
    def parse_json(self):
        merge_box = {}
        jfs = self.cfg["jfs"]
        for jf in jfs:
            station = os.path.basename(jf).split("_w")[-1].replace(".json","")
            try:    # S_splice_w1.json (手动添加前缀，防止主从机结果图命名冲突；重命名时加上前缀)
                client = os.path.basename(jf).split("_splice")[0]
            except: # 不加前缀
                client = ""
            json_data = read_json(jf)
            for key,values in json_data.items():
                no,fmt = key.split(".")
                filename = client+"_"+station+"_"+str(int(no)+1)+"."+fmt
                merge_box[filename] = {}
                images = []
                bbox = []
                for value in values:
                    images.append(station+"_"+str(values[value]["cam"]+1)+"_"+str(values[value]["no"]+1)+"_s"+str(values[value]["roi_no"]))
                    x1,y1,x2,y2 = values[value]["roi_dst"]
                    bbox.append([x1,y1,x2+1,y2+1])
                    # out_name = "{}_{}_{}_{}_s{}".format(client,station,camera,no,roi_no)
                merge_box[filename]["images"] = images
                merge_box[filename]["bbox"] = bbox
        return merge_box
          
    @abstractmethod
    def merge_im(self, ims):
        from dpp.dataset.transforms.pad import pad_im
        if len(ims) == 1:
            h,w = ims[0].shape[:2]
            pad_h,pad_w = self.size - h,self.size - w
            if h<self.size or w < self.size:
                new_im = pad_im(ims[0],(self.size,self.size),method="below")
            else:
                new_im = ims[0]
            return new_im
        else:
            h,w = ims[0].shape[:2]
            if h > w:
                new_im = np.hstack(ims)
            else:
                new_im = np.vstack(ims)
            return new_im
        # 


    @abstractmethod
    def merge_polygon(self, ims, regions):
        pass