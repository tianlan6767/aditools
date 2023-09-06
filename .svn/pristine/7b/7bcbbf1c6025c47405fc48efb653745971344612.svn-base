import shutil
import os
import re
from tqdm import tqdm
from dpp.common.dpp_json import DppJson
from dpp.dataset.preprocessing.json_convert import via_to_json,json_to_via
from dpp.common.util import load_file, make_dir


def move_img_by_json(src, json_data, dst, op="move"):
    for fn in json_data.keys():
        if op == "move":
            shutil.move(os.path.join(src, fn), os.path.join(dst, fn))
        else:
            shutil.copy(os.path.join(src, fn), os.path.join(dst, fn))


def move_img_by_product(src, dst, result):
    for num in tqdm(os.listdir(src)):
        root = os.path.join(src, num, result)
        imgs = load_file(root, format="img")
        for img in imgs:
            fn = os.path.basename(img)
            out = str(num)+"-"+fn
            shutil.copy(img, os.path.join(dst, out))


def move_img_by_result(src, dst, result="NG"):
    products = os.listdir(src)
    for p in tqdm(products):  # 产品号
        status = os.listdir(os.path.join(src, p))  # NG,OK,ORIG,Splice
        if result in status:
            imgs = os.listdir(os.path.join(src, p, result))
            for img in imgs:
                img_bmp = img.replace(".jpg", ".bmp")
                shutil.copy(os.path.join(src, p, result, img),
                            os.path.join(dst, "{}-{}".format(str(p), img)))
                shutil.copy(os.path.join(src, p, "ORIG", img_bmp),
                            os.path.join(dst, "{}-{}".format(str(p), img_bmp)))


def split_ng_ok(img_path, json_data):
    if img_path:
        ng_path = os.path.join(img_path, "NG")
        make_dir(ng_path)
        ok_path = os.path.join(img_path, "OK")
        make_dir(ok_path)
    ng_dict, ok_dict = {}, {}
    for k, v in json_data.items():
        regions = v["regions"]
        if len(regions):
            ng_dict[k] = json_data[k]
            if img_path:
              shutil.move(os.path.join(img_path, k), os.path.join(ng_path, k))
        else:
            ok_dict[k] = json_data[k]
            if img_path:
              shutil.move(os.path.join(img_path, k), os.path.join(ok_path, k))
    return ng_dict, ok_dict


def arrange_img_by_result(result_path,dst):
    for root,dirs,files in os.walk(result_path):
      if files:
        for item in files:
          out_path = root.replace(result_path,dst)
          if not os.path.exists(out_path):
            os.makedirs(out_path)
          shutil.move(os.path.join(dst,item.replace(".jpg",".bmp")),os.path.join(out_path,item.replace(".jpg",".bmp")))
  
  
def move_bmp_by_jpg(imgs,img_path,dst):
    for img in imgs:
        out = "".join(os.path.basename(img).split(".")[:-1])+".bmp"
        shutil.move(os.path.join(img_path,out),os.path.join(dst,out))
     
def checked_name(old,start=""):
    new_img = re.sub('[\u4e00-\u9fa5]', '', old)
    if len(start):
        new_img = new_img.replace(start,"")
    if new_img.endswith("-.bmp"):
        new_img = new_img.replace("-.bmp",".bmp")
    if "-" not in new_img:
        full = new_img.split("_")
        new_img = full[0]+"-"+"_".join(full[1:])
    return new_img
        
def check_filename(imgs,json_data,dst,start):
    for img in imgs:
        filename = os.path.basename(img)
        new_img = checked_name(filename,start)
        shutil.copy(img,os.path.join(dst,new_img))        
    if len(json_data):
        dj = DppJson(json_data)
        if dj.json_format.startswith('VIA'):
            json_data = via_to_json(json_data)
        new_json = {}
        for key,value in json_data.items():
            new_img = checked_name(key,start="")
            new_json[new_img]={}
            new_json[new_img]["filename"]=new_img
            new_json[new_img]["regions"] = value["regions"]
        if dj.json_format.startswith('VIA'):
            imgs = load_file(dst, format="img")
            new_json = json_to_via(imgs,new_json)
        return new_json
    else:
        return None
    