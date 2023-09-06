from .feature_63 import *
import os
import json
from tqdm import tqdm
import cv2
from copy import deepcopy
import numpy as np
import pandas as pd
import warnings
from glob import glob
import time


def extract_region(src: str, dst: str, jf: str, offset: int=0, square: bool=True):
    """提取并绘制标注区域

    Args:
        src (str): 图像来源
        dst (str): 提取图像的存储位置
        jf (str): 图像对应的json文件路径
        offset (int, optional): 偏移量. Defaults to 0.
        square (bool, optional): _description_. Defaults to True.
    """
    if os.path.exists(dst):
        warnings.warn(f'!!!已存在{dst}', Warning)
    assert os.path.exists(jf), f'!!!不存在{jf}'

    annotations = json.load(open(jf))
    pbar = tqdm(annotations, ncols=0, position=0)
    for k in pbar:
        pbar.set_description(k)
        imp = os.path.join(src, k)
        # tmp = k.split('_')
        # imp = os.path.join(src, tmp[1], "outer", "ORIG", '_'.join(tmp[2:]))
        # print(imp)
        if not os.path.exists(imp):
            warnings.warn(f'!!!未找到{k}', Warning)
            continue

        img = cv2.imdecode(np.fromfile(imp, dtype=np.uint8), cv2.IMREAD_COLOR)
        r, c = img.shape[:2]

        regions = annotations[k]["regions"]
        if not regions:
            warnings.warn(f'{k}上没有检出！', Warning)
            continue

        for i, region in enumerate(regions):
            # 缺陷类别：region_id
            try:
                region_id = str(region['region_attributes']['regions'])
            except KeyError:
                region_id = "0"
            os.makedirs(os.path.join(dst, region_id), exist_ok=True)

            pts_x = region['shape_attributes']['all_points_x']
            pts_y = region['shape_attributes']['all_points_y']
            x_min, x_max = min(pts_x), max(pts_x)
            y_min, y_max = min(pts_y), max(pts_y)

            if square:
                dis = (x_max - x_min) - (y_max - y_min)
                if dis > 0:
                    y_min = y_min - int(dis / 2)
                    y_max = y_max + int(dis / 2)
                else:
                    x_min = x_min + int(dis / 2)
                    x_max = x_max - int(dis / 2)

            x_min = x_min - offset if x_min - offset > 0 else 0
            x_max = x_max + offset if x_min + offset < c else c
            y_min = y_min - offset if y_min - offset > 0 else 0
            y_max = y_max + offset if y_min + offset < r else r

            img_copy = deepcopy(img)  # cv2.polylines会画在原图上
            img_region = deepcopy(img_copy[y_min: y_max, x_min: x_max])
            
            reg = np.dstack((pts_x, pts_y))
            img_mask = cv2.polylines(img_copy, pts=reg, isClosed=True, color=(0, 0, 255), thickness=1)
            img_mask_region = img_mask[y_min: y_max, x_min: x_max]
            
            s1, _, s2 = img_mask_region.shape
            bg_255 = np.ones((s1, 2, s2)) * 255
            img_joint = np.hstack((img_region, bg_255, img_mask_region))
            cv2.imencode('.bmp', img_joint)[1].tofile(
                os.path.join(dst, region_id, f"{k.split('.')[0]}_{i}.bmp"))
    print("==提取并绘制标注区域完成==")


def calcu_features(src, jf, wanted_features=None):
    """计算特征

    Args:
        src (str): 图像来源
        jf (str): 图像对应的json文件路径
        wanted_features (list, optional): 特征. Defaults to None.
        labels (bool, optional): 是否添加标签列. Defaults to False.
    """
    feature_map = get_feature_map()
    feature_names = list(feature_map.values())
    wanted_features = feature_names[:27] if wanted_features is None else wanted_features  # 默认计算编号为1~25的特征
    feature_method = get_feature(wanted_features)

    annotations = json.load(open(jf))
    info = {}
    pbar = tqdm(annotations, ncols=0, position=0)
    s = 0
    for k in pbar:
        pbar.set_description('calculate features: '+k)      
        regions = annotations[k]['regions']
        if not regions:
            warnings.warn(f'{k}上没有检出！', Warning)
            continue
        imp = os.path.join(src, k)
        # tmp = k.split('_')
        # imp = os.path.join(src, tmp[1], "outer", "ORIG", '_'.join(tmp[2:]))
        img0 = cv2.imdecode(np.fromfile(imp, dtype=np.uint8), 0)
        s1 = time.time()
        try:
            features = feature_method.get_image_feature(img0, regions)
        except:
            continue
        s += time.time() - s1
        for idx, feature in enumerate(features):
            feature_tmp = np.nan_to_num(feature)
            info[f"{k.split('.')[0]}_{idx}.bmp"] = feature_tmp
    # with open(r"\\Ds418\NAS3\A-ZK\raw_data\1205-1213-NG-baixian\features.txt", "w", encoding='utf-8') as f:
    #     f.write(pickle.dumps(info))
    info = pd.DataFrame(info).T
    print("*********特征提取耗时", s)
    print(info.shape)
    print(len(wanted_features))
    
    info.columns = wanted_features
    return info, wanted_features


def add_labels(excel: str, region_root: str):
    """将数据的类别插入excel中

    Args:
        excel (str): 需更新的excel文件路径
        region_root (str): 分好类的缺陷所在的根目录
    """
    df = pd.read_excel(excel, index_col='Unnamed: 0')
    labels = []
    for i in tqdm(df.index):
        try:
            img_glob = glob(os.path.join(region_root, '*', i.replace('.bmp', '.jpg')))[0]
        except:
            img_glob = glob(os.path.join(region_root, '*', i))[0]
        labels.append(int(img_glob.split('\\')[-2]))
    df["label"] = labels
    df.to_excel(os.path.join(os.path.split(excel)[0], 'features_with_labels.xlsx'))


def feature_selection(data: pd.DataFrame, target: str, method, max_features: int, **kwargs):
    """
        Filter features from the data according to method.

    :param data:
    :param target:
    :param method: filter method. ["DTC", "FSFC", "MIC", "chi2"]
            - If "DTC", 基于决策树的特征选择.
            - If "FSFC", 基于特征聚类的特征选择.
            - If "MIC", 计算每个特征和标签之间互信息，用于捕捉任意关系（包括线性和非线性关系）.
            - If "chi2", 计算每个非负特征和标签之间的卡方统计量.
    :param max_features: The retained value of max_features.
    :param kwargs:
    :return:
    """

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
    from sklearn.tree import DecisionTreeClassifier
    from .AEFD_FSFC import FSFC

    # feature_map = get_feature_map()
    # support_features = feature_map.values()
    # in_features = data.columns.tolist()
    # in_features = [f for f in in_features if f in support_features]
    in_features = data.columns.tolist()
    in_features.remove(target)
    x = data[in_features]
    y = data[target]

    if method == 'FSFC':
        return FSFC(x, **kwargs)[:max_features]
    elif method == 'MIC':
        X_ = StandardScaler().fit_transform(x)
        X = pd.DataFrame(X_, columns=x.columns)
        feature_scores = mutual_info_classif(X, y, random_state=0, n_neighbors=3, discrete_features='auto')
        fi_sort = sorted([(i, j) for i, j in zip(feature_scores, X.columns)], key=lambda u: u[0], reverse=True)
        return [i[1] for i in fi_sort[:max_features]]
    elif method == 'chi2':
        X_ = MinMaxScaler().fit_transform(x)
        X = pd.DataFrame(X_, columns=x.columns)
        selector = SelectKBest(chi2, k=max_features)
        selector.fit(X, y)

        outcome = selector.get_support()
        fi = [(i, j) for i, j in zip(outcome, X.columns)]
        return [f[1] for f in fi if f[0]]
    elif method == 'DTC':
        clf = DecisionTreeClassifier()
        # x = np.array(x)
        # y = np.array(y).astype("int")
        y = y.astype("int")
        clf.fit(x, y)
        feat_importance = clf.tree_.compute_feature_importances(normalize=False)
        feat_importance_sort = sorted([(i, j) for i, j in zip(feat_importance, x.columns)], key=lambda u: u[0],
                                      reverse=True)
        return [i[1] for i in feat_importance_sort[:max_features]]
    else:
        return in_features


if __name__ == "__main__":
    # src = r"\\Ds418\NAS\share\LX\tmp\OQC\val\val_data2\2_2"
    # jf = r"\\Ds418\NAS\share\LX\tmp\OQC\val\val_data2\2_2\data_updated.json"
    # info, wanted_features = calcu_features(src, jf)
    # info.to_excel(os.path.join(os.path.split(jf)[0], 'features.xlsx'))

    excel = r"\\Ds418\NAS\share\LX\tmp\OQC\val\val_data2\2_2-result\features.xlsx"
    region_root = r"\\Ds418\NAS\share\LX\tmp\OQC\val\val_data2\2_2-defect"
    add_labels(excel, region_root)