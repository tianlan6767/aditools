import cv2,shutil,os,copy
import numpy as np
from tqdm import tqdm
from dpp.common.mylog import Logger
from dpp.common.dpp_json import DppJson
from dpp.common.util import *
from dpp.dataset.transforms.pad import pad_im
from dpp.dataset.preprocessing.json_operate import filter_json
from dpp.dataset.preprocessing.json_convert import via_to_json


# def crop_small_img(img_path, json_data, dst, crop_size):
#     annotations = json_data.values()
#     for annotation in tqdm(annotations):
#         fn = annotation['filename']
#         img = cv2.imread(os.path.join(img_path, fn), 1)
#         regions = annotation['regions']
#         for i, region in enumerate(regions):
#             xs,ys,label = parse_region(region)      
#             start_x,end_x,start_y,end_y = scale_small_img(img.shape[:2],(xs,ys),crop_size=crop_size)  
#             img_region_fn = fn.split('.')[0] + '_{}'.format(i) + '.jpg'
#             img_mask_region = img[start_y:end_y, start_x:end_x]
#             label_path = os.path.join(dst, str(label))
#             make_dir(label_path)
#             cv2.imwrite(os.path.join(
#                 label_path, img_region_fn), img_mask_region)


def update_classify(path,imgs,json_data,copy_data):
    if len(imgs):
        set_imgs = list(set(imgs))
        new_json,_ = filter_json(json_data, set_imgs)
        save_json(new_json, path, os.path.basename(path))
        for img in set_imgs:
            filename = os.path.basename(img)
            shutil.move(os.path.join(os.path.dirname(path),filename),os.path.join(path,filename))
            copy_data.pop(filename) 


def classify_json(classify_path,json_data,img_path):
    copy_data = copy.deepcopy(json_data)
    folders = os.listdir(classify_path)
    mark_imgs,del_imgs=[],[]
    for folder in tqdm(folders):
        label_folder = os.path.join(classify_path,folder)
        imgs = load_file(label_folder, format="img")
        for img in imgs:
            split_fn = os.path.basename(img).split('_')
            index = split_fn[-1].split('.')[0]
            try:
                filename = '_'.join(split_fn[:-1])+".bmp" 
                regions = copy_data[filename]['regions']
            except:
                filename = '_'.join(split_fn[:-1])+".jpg" 
                regions = copy_data[filename]['regions']
            regions[int(index)]['region_attributes'].update({'regions':str(folder)})
            if folder=="mark":
                mark_imgs.append(filename)
            if folder=="del":
                del_imgs.append(filename)
    mark_path = os.path.join(img_path,"mark")
    del_path = os.path.join(img_path,"del")
    md = MultiDataset()
    inter_imgs = md.intersection(mark_imgs,del_imgs)
    if len(inter_imgs):
        Logger.error("原始图片中的小图 {} 分放冲突".format(";".join(inter_imgs)))
    update_classify(mark_path,mark_imgs,json_data,copy_data)
    update_classify(del_path,del_imgs,json_data,copy_data)
    return copy_data
  

def single_fill_crop_via(img_path,json_data,dst,crop=True,pad=0.2):
    old_dj = DppJson(json_data)
    if old_dj.json_format.startswith('VIA'):
        json_data = via_to_json(json_data)
    new_json = {}
    for key,value in tqdm(json_data.items()):
        fn_path = os.path.join(img_path,key)
        im = cv2.imread(fn_path,-1)
        if len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            BGR = True
        else:
            BGR = False
        mask = np.zeros(im.shape[:2], np.uint8)
        
        for index,region in enumerate(value["regions"]):
            xs, ys, label = parse_region(region)
            pts = np.array(list(zip(xs,ys)))
            # 在mask上将多边形区域填充
            # cv2.polylines(mask, [pts], 1, 255)
            cv2.fillPoly(mask, [pts], 255)
            
            if crop:
                crop_path = dst+"/crop" 
                make_dir(crop_path)
                mask_copy = np.zeros(im.shape[:2], np.uint8)
                cv2.fillPoly(mask_copy, [pts], 255)
                crop_im = cv2.bitwise_and(im[min(ys):max(ys),min(xs):max(xs)], im[min(ys):max(ys),min(xs):max(xs)], mask=mask_copy[min(ys):max(ys),min(xs):max(xs)])
                new_fn = key.split(".bmp")[0]+"_cp"+str(index)+".bmp"
                xs, ys, label = parse_region(region)
                new_h,new_w = crop_im.shape[:2]
                if 0<pad<1:
                    pad_h,pad_w = int(new_h*pad),int(new_w*pad)
                else:
                    pad_h,pad_w = pad,pad
                new_im = pad_im(crop_im,(new_h+pad_h,new_w+pad_w),method="center")    
                new_xs = [x-min(xs)+round(pad_w/2) for x in xs]
                new_ys = [y-min(ys)+round(pad_h/2) for y in ys]
                region_dict = {'shape_attributes': {'all_points_x': new_xs, 'all_points_y': new_ys}, 
                                'region_attributes': {'regions':label}}
                new_json[new_fn]={}
                new_json[new_fn]["filename"] = new_fn
                new_json[new_fn]["regions"] = [region_dict]
                if BGR:
                    new_im = cv2.cvtColor(new_im, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(os.path.join(crop_path,new_fn),new_im)
                
        # 逐位与，得到裁剪后图像，黑色背景
        black_im = cv2.bitwise_and(im, im, mask=mask)
        if BGR:
            black_im = cv2.cvtColor(black_im, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(dst,key),black_im)
                
    if crop:
        save_json(new_json, crop_path, 'crop')
        
        
def multi_fill_crop_via(img_path,json_data,dst,crop=True,pad=0.2):
    old_dj = DppJson(json_data)
    if old_dj.json_format.startswith('VIA'):
        json_data = via_to_json(json_data)
    crop_json,split_json = {},{}
    for key,value in tqdm(json_data.items()):
        fn_path = os.path.join(img_path,key)
        im = cv2.imread(fn_path,-1)
        if len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            BGR = True
        else:
            BGR = False       
        
        for index,region in enumerate(value["regions"]):
            mask_copy = np.zeros(im.shape[:2], np.uint8)
            xs, ys, label = parse_region(region)
            pts = np.array(list(zip(xs,ys)))
            # 在mask上将多边形区域填充
            # cv2.polylines(mask, [pts], 1, 255)
            cv2.fillPoly(mask_copy, [pts], 255)
            
            if crop:
                crop_path = dst+"/crop" 
                make_dir(crop_path)
                
                # cv2.fillPoly(mask_copy, [pts], 255)
                crop_im = cv2.bitwise_and(im[min(ys):max(ys),min(xs):max(xs)], im[min(ys):max(ys),min(xs):max(xs)], mask=mask_copy[min(ys):max(ys),min(xs):max(xs)])
                new_fn = key.split(".bmp")[0]+"_cp"+str(index)+".bmp"
                xs, ys, label = parse_region(region)
                new_h,new_w = crop_im.shape[:2]
                if 0<pad<1:
                    pad_h,pad_w = int(new_h*pad),int(new_w*pad)
                else:
                    pad_h,pad_w = pad,pad
                new_im = pad_im(crop_im,(new_h+pad_h,new_w+pad_w),method="center")    
                new_xs = [x-min(xs)+round(pad_w/2) for x in xs]
                new_ys = [y-min(ys)+round(pad_h/2) for y in ys]
                region_dict = {'shape_attributes': {'all_points_x': new_xs, 'all_points_y': new_ys}, 
                                'region_attributes': {'regions':label}}
                crop_json[new_fn]={}
                crop_json[new_fn]["filename"] = new_fn
                crop_json[new_fn]["regions"] = [region_dict]
                if BGR:
                    new_im = cv2.cvtColor(new_im, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(os.path.join(crop_path,new_fn),new_im)
                
            # 逐位与，得到裁剪后图像，黑色背景
            black_im = cv2.bitwise_and(im, im, mask=mask_copy)
            split_fn = key.split(".bmp")[0]+"_sp"+str(index)+".bmp"
            region_dict = {'shape_attributes': {'all_points_x': xs, 'all_points_y': ys}, 
                                'region_attributes': {'regions':label}}
            split_json[split_fn]={}
            split_json[split_fn]["filename"] = split_fn
            split_json[split_fn]["regions"] = [region_dict]
            if BGR:
                black_im = cv2.cvtColor(black_im, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(dst,split_fn),black_im)
    save_json(split_json, dst, 'split')          
    if crop:
        save_json(crop_json, crop_path, 'crop')
        
    
def fill_rect(img_path,dst,rect_json_data):
    rect_old_dj = DppJson(rect_json_data)
    if rect_old_dj.json_format=='VIA-RECT':
        rect_json_data = via_to_json(rect_json_data)
    for key,value in tqdm(rect_json_data.items()):
        fn_path = os.path.join(img_path,key)
        im = cv2.imread(fn_path,-1)
        h,w = im.shape[:2]
        if len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            BGR = True
        else:
            BGR = False  
        assert len(value["regions"])<2,"该检测框置黑只用于侧面相机,有且少于2个矩形框"
        if len(value["regions"])== 0:
            shutil.copy(os.path.join(img_path,key),os.path.join(dst,key))
        else:    
            for index,region in enumerate(value["regions"]):
                mask = np.zeros((h,w), np.uint8)
                xs, ys, label = parse_region(region)
                new_ys = [0,0,h,h]
                pts = np.array(list(zip(xs,new_ys)))
                # 在mask上将多边形区域填充
                # cv2.polylines(mask, [pts], 1, 255)
                cv2.fillPoly(mask, [pts], 255)
                black_im = cv2.bitwise_and(im, im, mask=mask)
                if BGR:
                    black_im = cv2.cvtColor(black_im, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(os.path.join(dst,key),black_im)


def crop_rect(img_path,dst,rect_json_data,mask_json_data,offset):
    rect_old_dj = DppJson(rect_json_data)
    if rect_old_dj.json_format=='VIA-RECT':
        rect_json_data = via_to_json(rect_json_data)
    new_json={}
    for key,value in tqdm(rect_json_data.items()):
        fn_path = os.path.join(img_path,key)
        im = cv2.imread(fn_path,-1)
        h,w = im.shape[:2]
        if len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            BGR = True
        else:
            BGR = False  
        assert len(value["regions"])<2,"该检测框置黑只用于侧面相机,有且少于2个矩形框"
        if mask_json_data: # 裁剪并且有缺陷标注
            mask_old_dj = DppJson(mask_json_data)
            if mask_old_dj.json_format.startswith('VIA'):
                mask_json_data = via_to_json(mask_json_data)
            regions_list = []
            crop_regions = mask_json_data[key]
            
            crop_xs,crop_ys, crop_label = parse_region(value["regions"][0])
            # new_ys = [0,0,h,h]
            crop_im = im[:,min(crop_xs)-offset:max(crop_xs)+offset]
                
            for index,region in enumerate(crop_regions["regions"]):
                xs, ys, label = parse_region(region)  
                new_xs = [x-min(crop_xs)+offset  for x in xs]
                region_dict = {'shape_attributes': {'all_points_x': new_xs, 'all_points_y': ys}, 'region_attributes': {'regions':label}}
                regions_list.append(region_dict)
                new_json = build_json(new_json,key,regions_list)
            cv2.imwrite(os.path.join(dst,key),crop_im)
        else:   # 只裁剪
            xs, ys, label = parse_region(value["regions"][0])
            new_ys = [0,0,h,h]
            crop_im = im[:,min(xs)-offset:max(xs)+offset]
            cv2.imwrite(os.path.join(dst,key),crop_im)
    if len(new_json):
        save_json(new_json, dst, 'crop')
      
def fill_crop_multi(img_path,json_data,dst):
    old_dj = DppJson(json_data)
    if old_dj.json_format.startswith('VIA'):
        json_data = via_to_json(json_data)
    for key,value in tqdm(json_data.items()):
        fn_path = os.path.join(img_path,key)
        im = cv2.imread(fn_path,-1)
        h,w = im.shape[:2]
        if len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            BGR = True
        else:
            BGR = False  
        
        mask = np.zeros((h,w), np.uint8)
        if len(value["regions"])== 0:
            shutil.copy(os.path.join(img_path,key),os.path.join(dst,key))
        else:    
            for index,region in enumerate(value["regions"]):
                xs, ys, label = parse_region(region)
                if label == "0":
                    pts = np.array(list(zip([int(x) for x in xs],[int(y) for y in ys])))
                    cv2.fillPoly(im, [pts], 0)
            for index,region in enumerate(value["regions"]):
                xs, ys, label = parse_region(region)
                if label != "0":
                    xs, ys, label = parse_region(region)
                    pts = np.array(list(zip([int(x) for x in xs],[int(y) for y in ys])))
                    # 在mask上将多边形区域填充
                    # cv2.polylines(mask, [pts], 1, 255)
                    cv2.fillPoly(mask, [pts], 255)
        black_im = cv2.bitwise_and(im, im, mask=mask)
        if BGR:
            black_im = cv2.cvtColor(black_im, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(dst,key),black_im)
        
        
