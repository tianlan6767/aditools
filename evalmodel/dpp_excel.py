import cv2,os
from tqdm import tqdm
from dpp.common.excel import *
from dpp.common.masks import *
from evalmodel import ModelAnalyzer
from dpp.common.mylog import Logger
from dpp.dataset.visualization.json_draw import draw_polygon
from shapely.geometry import Polygon,MultiPoint  
from dpp.common.util import parse_region,scale_small_img,cal_area,read_json,load_file


def cal_iou(data1,data2):
    poly1 = Polygon(data1).convex_hull
    poly2 = Polygon(data2).convex_hull
    union_poly = np.concatenate((data1,data2))   #合并两个box坐标，变为8*2
    if not poly1.intersects(poly2): #如果两四边形不相交
        iou = 0
    else:
        inter_area = poly1.intersection(poly2).area   #相交面积
        union_area = MultiPoint(union_poly).convex_hull.area
        if union_area == 0:
            iou= 0
        iou=float(inter_area) / union_area
    return iou

def cal_region_iou(regions1,regions2):
    """
    regions1: 标注via 的regions
    regions2: 模型json 的regions
    """
    iou_results = []
    if len(regions2):
        for mask1 in regions1:
            xs1 = mask1['shape_attributes']['all_points_x']
            ys1 = mask1['shape_attributes']['all_points_y']
            scene_mask = XX0YY_XY0XY([xs1,ys1])
            results = []
            for mask2 in regions2:
                xs2 = mask2['shape_attributes']['all_points_x']
                ys2 = mask2['shape_attributes']['all_points_y']
                pred_mask = XX0YY_XY0XY([xs2,ys2])
                results.append(cal_iou(scene_mask,pred_mask))
            iou_results.append(results)
    else:
        iou_results = [[0]]*len(regions1)
    return iou_results

class JsonExcel(BaseExcel):
    fn = CharField(column_name="图片名",column_width=20)
    origin_img = ImageField(column_name="原图",column_width=8.5)
    mask_img = ImageField(column_name="标注",column_width=8.5)
    label = CharField(column_name="类别",column_width=8.5)
    
    
class ModelExcel(BaseExcel):
    fn = CharField(column_name="图片名",column_width=20)
    origin_img = ImageField(column_name="原图",column_width=8.5)
    mask_label = CharField(column_name="标注类别",column_width=14)
    mask_area = CharField(column_name="标注面积",column_width=14)
    mask_img = ImageField(column_name="标注",column_width=8.5)
    model_img = ImageField(column_name="模型",column_width=8.5)
    model_label = CharField(column_name="模型类别",column_width=14)
    model_area = CharField(column_name="模型面积",column_width=14)
    score = CharField(column_name="模型得分",column_width=14)
    iou = CharField(column_name="IOU",column_width=8)
    

def json_to_excel(img_path,json_data,crop_size = 120):
  row = 0
  se = JsonExcel()
  for key,value in tqdm(json_data.items()):
      filename = value["filename"]
      im = cv2.imread(os.path.join(img_path,filename),1)
      im_h,im_w = im.shape[:2]
      regions = value["regions"]
      for index,region in enumerate(regions):
          im_rgb = im.copy()
          # im_rgb = cv2.cvtColor(im.copy(),cv2.COLOR_GRAY2RGB)
          xs,ys,label = parse_region(region)
          se.label = label
          se.fn = key.split(".")[0]+"_"+str(index)+".bmp"
          start_x,end_x,start_y,end_y = scale_small_img((im_h,im_w),(xs,ys),crop_size)
          se.origin_img = cv2.resize(im[start_y:end_y,start_x:end_x],(crop_size,crop_size))
          draw_polygon(im_rgb, [xs,ys]) 
          se.mask_img = cv2.resize(im_rgb[start_y:end_y,start_x:end_x],(crop_size,crop_size))
          se.save(row)
          row += 1
  out = os.path.join(img_path,'缺陷标注表.xlsx')
  wb.save(out)
  Logger.info("success save excel: {}".format(out))
  
  
def model_to_excel(img_path,mask_data,model_data,crop_size = 120):
    row = 0
    for key,value in tqdm(mask_data.items()):
        mask_regions = value['regions']
        model_regions = model_data[key]["regions"]
        iou_results = cal_region_iou(mask_regions,model_regions)
        index_list = [i.index(max(i)) for i in iou_results]
        excel = ModelExcel()
        for index,region in enumerate(mask_regions):
            origin_im = cv2.imread(os.path.join(img_path,key),1)
            copy_im = origin_im.copy()
            xs,ys,mask_label = parse_region(region)
            start_x,end_x,start_y,end_y = scale_small_img(origin_im.shape[:2],(xs,ys),crop_size)
            draw_polygon(copy_im, [xs,ys],color=[0,255,0])      # 标注绘制
            paste_origin_im = cv2.resize(origin_im[start_y:end_y, start_x:end_x], (crop_size, crop_size))
            excel.origin_img = paste_origin_im
            paste_mask_im   = cv2.resize(copy_im[start_y:end_y, start_x:end_x], (crop_size, crop_size))
            excel.mask_img = paste_mask_im
            excel.fn = key
            excel.mask_label = mask_label
            excel.mask_area = cal_area(xs,ys)
            # 缺陷可能未检出
            if max(iou_results[index])>0.00001:   
                model_xs,model_ys,model_label = parse_region(model_regions[index_list[index]])  
                excel.score = round(model_regions[index_list[index]]["region_attributes"]["score"],2)
                excel.model_area = cal_area(model_xs,model_ys)
                excel.model_label = model_label
                model_im_copy = origin_im.copy()
                draw_polygon(model_im_copy, [model_xs,model_ys],color=[255,0,0])
                paste_model_im = cv2.resize(model_im_copy[start_y:end_y, start_x:end_x], (crop_size, crop_size))
                excel.model_img = paste_model_im
                excel.iou = round(max(iou_results[index]),2)
            else:
                excel.model_area = -1
                excel.model_label = -1
                excel.score = -1
                excel.area = -1
                excel.model_img = paste_origin_im
                excel.iou = -1
            excel.save(row)
            row += 1
    out = os.path.join(img_path,'缺陷模型得分表.xlsx')
    wb.save(out)
    Logger.info("success save excel: {}".format(out))
  
def multi_model_to_excel(img_path,mark_jf,weights_path,crop_size=120):
  weights = load_file(weights_path, format="json")
  models = [os.path.basename(item).replace(".json","") for item in weights]
  
  MA = ModelAnalyzer.load(img_path,mark_jf,weights_path)
  result = MA.classify_defect(models)
  """
  {'PAD_B_1230_1_NG_ADH_404-1_1_1.bmp': {
    'filename': 'PAD_B_1230_1_NG_ADH_404-1_1_1.bmp', 
    'regions': [{'shape_attributes': {'all_points_x': [65, 4414, 4414, 65], 'all_points_y': [900, 900, 1076, 1076]}, 
                  'region_attributes': {'regions': '1', 'detect': ['1', '1', '1', '1', '1', '1'], 'scores': [0.6155, 0.6625, 0.6419, 0.5686, 0.6284, 0.6389], 
                  'ctr_scores': ['', '', '', '', '', ''], 'cls_scores': ['', '', '', '', '', ''], 'fpn': [3, 3, 3, 3, 3, 3]}}}]}
  """
  
  class MultiModelExcel(BaseExcel):        
    fn = CharField(column_name="图片名",column_width=20)
    origin_img = ImageField(column_name="原图",column_width=8.5)
    mask_img = ImageField(column_name="标注",column_width=8.5)
    mask_label = CharField(column_name="类别",column_width=8.5)
    for index,label in enumerate(models):
        locals()["M"+str(index+1)] = CharField(column_name=label.replace("model_",''),column_width=15)
        
  row = 0
  se = MultiModelExcel()
  for key,value in tqdm(result.items()):
      filename = value["filename"]
      im = cv2.imread(os.path.join(img_path,filename),1)
      regions = value["regions"]
      for index,region in enumerate(regions):
          im_rgb = im.copy()
          xs,ys,mask_label = parse_region(region)
          detect_list = region["region_attributes"]["detect"]  # list
          score_list = region["region_attributes"]["scores"]
          se.mask_label = mask_label
          for index,score in enumerate(score_list):
              setattr(se,"M"+str(index+1),round(score,2))
          se.fn = key.split(".")[0]+"_"+str(index)+".bmp"
          start_x,end_x,start_y,end_y = scale_small_img(im.shape[:2],(xs,ys),crop_size)
          se.origin_img = cv2.resize(im[start_y:end_y,start_x:end_x],(crop_size,crop_size))
          draw_polygon(im_rgb, [xs,ys]) 
          se.mask_img = cv2.resize(im_rgb[start_y:end_y,start_x:end_x],(crop_size,crop_size))
          se.save(row)
          row += 1
  out = os.path.join(img_path,'缺陷多模型得分表.xlsx')
  wb.save(out)
  Logger.info("success save excel: {}".format(out))
  
  

