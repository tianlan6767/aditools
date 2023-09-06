import numpy as np
from detectron2.utils.visualizer import GenericMask
from shapely.geometry import Polygon,MultiPoint  
from dpp.common.masks import XX0YY_XY0XY,XYXY_XY0XY


def parse_predictions(predictions):
    """
    解析每张图片模型推理结果predictions
    return: [],[],[],[]
    """
    pred_scores = predictions.scores.tolist()
    pred_classes = predictions.pred_classes.tolist()
    masks = predictions.pred_masks
    masks = [GenericMask(x, predictions.image_size[0], predictions.image_size[1]) for x in np.asarray(masks)]
    pred_masks = [mask.polygons for mask in masks]    # 一个推测结果也可能有多个标注，尤其是线状多个圈圈组合
    # pred_masks = [list(chain.from_iterable(polygon)) for polygon in polygons]
    fpn_levels = predictions.fpn_levels.tolist()
    return pred_scores,pred_classes,pred_masks,fpn_levels

def model_to_json(predictions): 
    """
    predictions convert to via mark format
    """
    pred_scores,pred_classes,pred_masks,fpn_levels = parse_predictions(predictions)
    regions = []
    for index in range(len(pred_scores)):
        masks = [i for i in pred_masks[index] if len(i)>8]
        # for mask in masks:      # 拆分一个instance多个分段描框
        #     points = np.array(mask,dtype=np.int32).tolist()
        #     new_dict = {'shape_attributes':{'all_points_x':points[::2],'all_points_y':points[1::2]},
        #             'region_attributes':{'regions':str(pred_classes[index]+1),"score":round(pred_scores[index],4),"fpn_levels":fpn_levels[index]}}
        #     regions.append(new_dict)
          
        masks = [i for i in pred_masks[index] if len(i)>8]
        if len(masks):
            points = np.array(np.concatenate(masks),dtype=np.int32).tolist()
            
            new_dict = {'shape_attributes':{'all_points_x':points[::2],'all_points_y':points[1::2]},
                'region_attributes':{'regions':str(pred_classes[index]+1),"score":round(pred_scores[index],4),"fpn_levels":fpn_levels[index]}}
            # new_dict = {'shape_attributes':{'all_points_x':points[::2],'all_points_y':points[1::2]},
            #     'region_attributes':{'regions':str(pred_classes[index]+1)}}
            regions.append(new_dict)
        
    return regions


def cal_prediction_iou(predictions,regions):
    """
    计算每张图片每个标注的检测结果（与模型结果对比）  # 可以优化，只要结果为1就可以停止计算iou
    predictions: per pic predictions
    regions: per pic via mask [{"shape_attributes": {"all_points_x": [], "all_points_y": []}, "region_attributes": {}},....]
    all_results: [[iou,iou....iou],[],[]], length:mask length *predictions length
    """
    iou_results = []
    _,_,pred_masks,_ = parse_predictions(predictions)
    if len(pred_masks): # 模型检测结果可能为空
        for region in regions:
            xs = region['shape_attributes']['all_points_x']
            ys = region['shape_attributes']['all_points_y']
            scene_mask = XX0YY_XY0XY([xs,ys])
            results = []
            for masks in pred_masks:
                pred_mask = XYXY_XY0XY(masks)
                results.append(cal_iou(scene_mask,pred_mask))
            iou_results.append(results)
    else:
        # iou_results.append([0]*len(regions))
        # iou_results = iou_results*len(regions)
        iou_results = [[0]]*len(regions)
    return iou_results



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
                # print("scene_mask:",scene_mask)
                # print("pred_mask:",pred_mask)
                results.append(cal_iou2(scene_mask,pred_mask))
            iou_results.append(results)
    else:
        iou_results = [[0]]*len(regions1)
    return iou_results

    
def cal_defect(result):
    """
    params: [[iou,...iou],......[]]
    return: ng_num:一个产品里标注的缺陷数, defect_num:检测出的缺陷数, img_num:图片检出与否
    """
    ng_num = len(result)
    result = [max(_) for _ in result]   # 取出每个缺陷检测出结果
    defect_num = len(list(filter(filter_NG,result)))
    if defect_num>0:
        img_num = 1
    else:
        img_num = 0
    return ng_num, defect_num, img_num

def cal_iou(data1, data2):
    """
    计算两个多边形是否相交
    """
    poly1 = Polygon(data1).convex_hull
    poly2 = Polygon(data2).convex_hull
    if not poly1.intersects(poly2):
        return 0
    else:
        return 1


def filter_NG(x):
    return x>0


def cal_iou2(data1,data2):
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

def cal_pixel_iou(mark_region,inf_region):
        """
            计算输入区域间的IOU

        :param other: 推理区域
        :return: IOU []
        """
        import cv2
   
        all_region = []
        all_region.extend(mark_region)
        all_region.extend(inf_region)

        # 计算合适的背景图大小
        mark_rect = cv2.boundingRect(np.array(all_region))
        shape = (mark_rect[3], mark_rect[2])
        mark_img = np.zeros(shape, dtype=np.uint8)  # shape = (row, col)
        inf_img = np.zeros(shape, dtype=np.uint8)

        # 生成推理和标注图
        if isinstance(mark_region[0], list):
            for region in mark_region:
                cv2.fillPoly(mark_img, [np.array(region) - (mark_rect[0], mark_rect[1])], color=255)
        else:
            cv2.fillPoly(mark_img, [np.array(mark_region) - (mark_rect[0], mark_rect[1])], color=255)
        if isinstance(inf_region[0], list):
            for region in inf_region:
                cv2.fillPoly(inf_img, [np.array(region) - (mark_rect[0], mark_rect[1])], color=255)
        else:
            cv2.fillPoly(inf_img, [np.array(inf_region) - (mark_rect[0], mark_rect[1])], color=255)

        # 计算交集和并集
        dst_img1 = np.zeros(shape, dtype=np.uint8)
        dst_img2 = np.zeros(shape, dtype=np.uint8)
        and_img = cv2.bitwise_and(mark_img, inf_img, dst_img1)
        or_img = cv2.bitwise_or(mark_img, inf_img, dst_img2)
        iou = (cv2.countNonZero(and_img)) / ((cv2.countNonZero(or_img)) + 1)


        return iou
