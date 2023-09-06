from dpp.dataset.preprocessing.json_operate import *
from dpp.dataset.preprocessing.rename import *
from dpp.dataset.preprocessing.json_convert import *
from dpp.dataset.preprocessing.file_move import *
from dpp.dataset.preprocessing.crop_regions import *
from dpp.dataset.visualization.json_vis import JsonVis
from dpp.dataset.visualization.json_draw import draw_mask
from dpp.dataset.transforms.func import *
from dpp.dataset.transforms.convert import *
from dpp.dataset.preprocessing.file_split import *
from dpp.common.util import load_file, read_json, save_json, make_dir
from dpp.train.save_coco import convert_coco,cal_mean_std
from evalmodel.dpp_excel import json_to_excel,model_to_excel,multi_model_to_excel
from dpp.dataset.visualization.analysis_train_dataset import Convention_Analysis, Work_Station_Analysis, bulid_analysis_report


# 现场标注重命名
def rename_one_folder_api(img_path, jf, dst, start):
    make_dir(dst)
    imgs = load_file(img_path, format="img")
    json_data = read_json(jf)
    rename_json = rename_one_folder(imgs, json_data, dst, start)
    save_json(rename_json, dst, "rename_one_folder")


def rename_classify_folder_api(src, dst, start):
    make_dir(dst)
    new_json = rename_classify_folder(src, dst, start)
    save_json(new_json, dst, "rename_classify")


def rename_product_folder_api(src, dst, start):
    make_dir(dst)
    new_json = rename_product_folder(src, dst, start)
    save_json(new_json, dst, "rename_multi_folder")


### 纠正中文命名及json
def correct_img_json_api(img_path,jf,dst):
    make_dir(dst)
    imgs = load_file(img_path, format="img")
    json_data = read_as_str(jf)
    new_json_data = correct_img_json(imgs,json_data,dst)
    if json_data:
        save_json(eval(new_json_data), dst, "correct_name")

# OK图重命名标注
def rename_img_api(img_path, dst, start):
    make_dir(dst)
    imgs = load_file(img_path, format="img")
    rename_img(imgs, dst, start)


def rename_img_json_api(img_path, dst, start):
    make_dir(dst)
    imgs = load_file(img_path, format="img")
    rename_img(imgs, dst, start)
    new_imgs = load_file(dst, format="img")
    ok_json = add_ok_json(new_imgs)
    save_json(ok_json, dst, "ok-pcs")


def read_json_api(jf,mini=True):
    json_data = read_json(jf)
    JsonVis(json_data,mini = mini)()


# JSON格式转换
def via_to_json_api(jf):
    json_data = read_json(jf)
    new_json = via_to_json(json_data)
    save_json(new_json, os.path.dirname(jf), "via_to_json")


def json_to_via_api(img_path, jf):
    imgs = load_file(img_path, format="img")
    json_data = read_json(jf)
    new_json = json_to_via(imgs, json_data)
    save_json(new_json, os.path.dirname(jf), "json_to_via")


def json_to_yolo_api(img_path, jf, dst,seg):
    json_data = read_json(jf)
    make_dir(dst)
    json_to_yolo(img_path, json_data, dst,seg)


# json操作
def merge_json_api(jf):
    jfs = load_file(jf, format="json")
    merge_data = merge_json([read_json(jf) for jf in jfs])
    save_json(merge_data, jf, "data_merge")


def del_empty_key_api(jf):
    json_data = read_json(jf)
    new_json = del_empty_key(json_data)
    save_json(new_json,os.path.dirname(jf),"remove_empty")


def json_cover_api(old_jf, new_jf, dst):
    old_json_data = read_json(old_jf)
    new_json_data = read_json(new_jf)
    cover_json = json_cover(old_json_data, new_json_data)
    save_json(cover_json, dst, 'data_cover')


def del_small_area_api(jf, min_area):
    json_data = read_json(jf)
    new_json = del_small_area(json_data, min_area)
    save_json(new_json, os.path.dirname(jf), 'del_small')


def filter_labels_api(jf, labels):
    json_data = read_json(jf)
    new_json = filter_labels(json_data, labels)
    save_json(new_json, os.path.dirname(jf), 'filter_labels')


def split_limit_mask_api(img_path, jf,limit_rate):
    # imgs = load_file(img_path, format="img")
    json_data = read_json(jf)
    new_json = split_limit_mask(img_path, json_data,limit_rate)
    save_json(new_json, img_path, 'split_mask')
    

def copy_via_api(jf,img_path):
    json_data = read_json(jf)
    imgs = load_file(img_path, format="img")
    new_json = copy_via(json_data,imgs)
    save_json(new_json, img_path, 'copy')
    
    
def match_point_json_api(jf,img_path):
    json_data = read_json(jf)
    imgs = load_file(img_path, format="img")
    new_json = match_point_json(json_data,imgs)
    save_json(new_json, img_path, 'mapper')
    
# 绘制标注
def draw_mask_api(img_path, jfs,draw_cfg_list, dst, spotcheck):
    make_dir(dst)
    imgs = load_file(img_path, format="img")
    json_data_list = [read_json(jf) for jf in jfs]
    draw_mask(imgs, json_data_list,draw_cfg_list, dst,spotcheck)


# 分割与拼接
def img_segment_api(name, img_path, jf, dst):
    make_dir(dst)
    imgs = load_file(img_path, format="img")
    json_data = read_json(jf)
    # new_json = segment(name, imgs, json_data, dst)
    new_json = SegmentDispatch(name, imgs, json_data, dst)()
    save_json(new_json, dst, 'segment')


def img_merge_api(name, img_path, jf, dst,size):
    make_dir(dst)
    # imgs = load_file(img_path, format="img")
    json_data = read_json(jf)
    new_json = MergeDispatch(name, img_path, json_data, dst,size)()
    # new_json = merge(name, img_path, json_data, dst,size)
    save_json(new_json, dst, 'merge')
    
    
def djp_img_json_merge_api(img_path,jf,dst):
    for product in os.listdir(img_path):
        folder_path = os.path.join(img_path,str(product),"outer","ORIG")
        seg_dst = dst+"_merge"+ "\{}".format(product)
        img_segment_api("JsonSeg", folder_path, jf, seg_dst)
        mer_dst = seg_dst+"_mer"
        img_merge_api("JsonMerge", seg_dst, jf, mer_dst,size=2048)
   
     
def ng_img_json_merge_api(img_path,jf):
    seg_dst = img_path+"_seg"
    mer_dst = img_path+"_mer"
    make_dir(mer_dst)
    make_dir(seg_dst)
    imgs = load_file(img_path, format="img")
    json_data = read_json(jf)
    for img in imgs:
        file_seg_dst = seg_dst+"\{}".format(os.path.basename(img).split(".")[0])
        make_dir(file_seg_dst)
        segment("JsonSeg", [img], json_data=[], dst=file_seg_dst)    
        seg_imgs = load_file(file_seg_dst, format="img")
        merge("JsonMerge", seg_imgs, json_data=json_data, dst = mer_dst,size=2048)


def img_aug_api(img_path, jf, dst, tfms):
    make_dir(dst)
    imgs = load_file(img_path, format="img")
    json_data = read_json(jf)
    new_json = img_aug(imgs, json_data, dst, tfms)
    save_json(new_json, dst, 'aug')


def convert_format_api(img_path, dst, color, out_channel):
    make_dir(dst)
    imgs = load_file(img_path, format="img")
    convert_format(imgs, dst, color, out_channel)
    

def img_pad_api(img_path, dst,pad_size):
    make_dir(dst)
    imgs = load_file(img_path, format="img")
    img_pad(imgs,dst,pad_size)
    
    
def fill_crop_via_api(img_path,jf,dst,multi,crop,pad):
    make_dir(dst)        
    json_data = read_json(jf)
    if multi:
        multi_fill_crop_via(img_path,json_data,dst,crop,pad)
    else:
        single_fill_crop_via(img_path,json_data,dst,crop,pad)
        

def fill_rect_api(img_path,dst,rect_jf,mask_jf):
    make_dir(dst) 
    rect_json_data = read_json(rect_jf)
    fill_rect(img_path,dst,rect_json_data)
    

def crop_rect_api(img_path,dst,rect_jf,mask_jf,offset):
    make_dir(dst) 
    rect_json_data = read_json(rect_jf)
    mask_json_data = read_json(mask_jf)
    crop_rect(img_path,dst,rect_json_data,mask_json_data,offset)
    
    
def fill_crop_multi_api(img_path,jf,dst):
    make_dir(dst) 
    json_data = read_json(jf)
    fill_crop_multi(img_path,json_data,dst)

## 数据拆分
def crop_small_img_api(img_path, jf, dst, scale,offset, crop_size):
    make_dir(dst)
    json_data = read_json(jf)
    crop_small_img(img_path, json_data, dst, scale,offset,crop_size)
    
  
def crop_small_img_json_api(img_path, jf, dst, scale,offset, crop_size):
    make_dir(dst)
    json_data = read_json(jf)
    new_json = crop_small_img_json(img_path, json_data, dst, scale,offset,crop_size)
    save_json(new_json, dst, 'crop_img') 


def classify_json_api(classify_path,jf,img_path):
    json_data = read_json(jf)
    new_json = classify_json(classify_path,json_data)
    save_json(new_json, img_path, 'reclassify') 


def dataset_partition_api(img_path,jf,repeat,ratio,seed):
    json_data = read_json(jf)
    train_path = os.path.join(os.path.dirname(img_path),"train")
    test_path = os.path.join(os.path.dirname(img_path),"test")
    make_dir(train_path)
    make_dir(test_path)
    dataset_partition(json_data,img_path,repeat,ratio,seed)
    filter_json_by_img_api(train_path, jf, train_path)
    filter_json_by_img_api(test_path, jf, test_path)
    
def split_json_by_folder_api(folder_path,jf):
    json_data = read_json(jf)
    split_json_by_folder(folder_path,json_data)
  
def split_img_by_station_api(img_path):
    imgs = load_file(img_path, format="img")
    split_img_by_station(imgs)
      

def label_partition_by_station_api(folder_path,jf, scale,offset,crop_size):
    json_data = read_json(jf)
    label_partition_by_station(folder_path,json_data, scale,offset,crop_size)

            
def dataset_partition_by_station_api(folder_path,repeat,ratio):
    new_json = dataset_partition_by_station(folder_path,repeat,ratio)
    train_imgs = load_file(os.path.join(folder_path,"train"), format="img")
    filter_json_data,remain_json_data = filter_json(new_json, train_imgs)
    save_json(filter_json_data, os.path.join(folder_path,"train"), 'train')
    JsonVis(filter_json_data,mini = True)()
    save_json(remain_json_data, os.path.join(folder_path,"test"), 'test')
    JsonVis(remain_json_data,mini = True)()

  


# 移动筛选
def filter_json_by_img_api(img_path, jf, dst):
    make_dir(dst)
    imgs = load_file(img_path, format="img")
    json_data = read_json(jf)
    filter_data, remain_data = filter_json(json_data,imgs)
    save_json(filter_data, dst, 'filter_json')


def move_img_by_json_api(img_path, jf, dst, move):
    make_dir(dst)
    imgs = load_file(img_path, format="img")
    json_data = read_json(jf)
    move_img_by_json(img_path, json_data, dst, move)
    

def split_ng_ok_api(img_path, jf):
    json_data = read_json(jf)
    ng_dict, ok_dict = split_ng_ok(img_path, json_data)
    save_json(ng_dict, img_path, 'ng')
    save_json(ok_dict, img_path, 'ok')
    
    
def move_bmp_by_jpg_api(jpg_path,img_path,dst):
    make_dir(dst)
    jpg_imgs = load_file(jpg_path, format="img")
    move_bmp_by_jpg(jpg_imgs,img_path,dst)


# 训练流程
def cal_mean_std_api(img_path, channel):
    imgs = load_file(img_path, format="img")
    cal_mean_std(imgs, channel)


def save_coco_api(img_path, jf, dst):
    make_dir(dst)
    json_data = read_json(jf)
    coco_output = convert_coco(img_path, json_data)
    save_json(coco_output, dst, 'train')


## 分析
def json_to_excel_api(img_path, jf):
    json_data = read_json(jf)
    img_path=os.path.dirname(jf)
    json_to_excel(img_path,json_data,crop_size = 120)
    
    
def model_to_excel_api(img_path,mask_jf,model_jf):
    mask_data = read_json(mask_jf)
    model_data = read_json(model_jf)
    model_to_excel(img_path,mask_data,model_data)


def multi_model_to_excel_api(img_path,mask_jf,weights_path):
    multi_model_to_excel(img_path,mask_jf,weights_path)
  
  

## 现场数据操作
def move_img_by_product_api(img_path, dst, result):
    move_img_by_product(img_path, dst, result)


def move_img_by_result_api(img_path, dst, result):
    make_dir(dst)
    move_img_by_result(img_path, dst, result)


def arrange_img_by_result_api(result_path,dst):
    arrange_img_by_result(result_path,dst)
    
    
def check_filename_api(img_path,jf,dst,start=""):
    make_dir(dst)
    imgs = load_file(img_path, format="img")
    json_data = read_json(jf)
    new_json = check_filename(imgs,json_data,dst,start)
    if new_json:
        save_json(new_json, dst, 'format')


#分析训练数据集, jf是json文件, dst是保存分析报告的路径
def read_json_analysis_api(jf, dst='.//'):
    make_dir(dst)
    json_data = read_json(jf)
    Convention_Analysis(json_data=json_data, dst=dst)()
    Work_Station_Analysis(json_data=json_data, dst=dst)()
    bulid_analysis_report(dst=dst)
