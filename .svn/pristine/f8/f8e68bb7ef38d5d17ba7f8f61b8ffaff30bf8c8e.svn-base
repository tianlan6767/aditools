from imblearn.over_sampling import ADASYN, SMOTEN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import pandas as pd
import numpy as np
import os
import json
import shutil
from tqdm import tqdm
from glob import glob
from collections import Counter
import warnings
import time
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_x_y(excel_file, drop_cols: list, target: str = '', index_col: str = ''):
    """读取数据，获取 X、y

    Args:
        excel_file: excel文件
        drop_cols: 不是特征的列
        target: 标签列
        index_col: id列

    Returns:
        X:
        y:
        X_cols: X对应的列名
        data.index.tolist(): 行索引
    """
    if index_col in drop_cols:
        drop_cols.remove(index_col)
    if target in drop_cols:
        drop_cols.remove(target)
    if index_col == '':
        data = pd.read_excel(excel_file)
    else:
        data = pd.read_excel(excel_file, index_col=index_col)

    X_cols_anti = drop_cols if target == '' else drop_cols+[target]
    X = np.array(data.drop(X_cols_anti, 1))

    # 获取 X 对应的列名
    X_cols = data.columns.tolist()
    for i in X_cols_anti:
        X_cols.remove(i)

    if target == '':
        return X, X_cols, data.index.tolist()

    y = np.array(data[target])
    return X, y, X_cols, data.index.tolist()


def split_tt(not_yy: np.array, yy: np.array, cols: list, dst: str):
    """划分数据集

    Args:
        not_yy: 除了标签列的其他列，可以不只是特征
        yy: 标签
        cols: xx的列名 + yy的列名
        dst: 保存地址
    """
    not_yy_train, not_yy_test, yy_train, yy_test = \
        train_test_split(not_yy, yy, test_size=0.3, random_state=42, stratify=yy)

    train_data = pd.DataFrame(np.hstack([not_yy_train, yy_train.reshape(-1, 1)]), columns=cols)
    test_data = pd.DataFrame(np.hstack([not_yy_test, yy_test.reshape(-1, 1)]), columns=cols)

    os.makedirs(dst, exist_ok=True)
    save_path_train = os.path.join(dst, 'train_data-7_10.xlsx')
    save_path_test = os.path.join(dst, 'test_data-3_10.xlsx')
    if os.path.exists(save_path_train):
        warnings.warn(f'!!!已存在{save_path_train}，已被覆盖', Warning)
    if os.path.exists(save_path_test):
        warnings.warn(f'!!!已存在{save_path_test}，已被覆盖', Warning)
    train_data.to_excel(save_path_train, index=False)
    test_data.to_excel(save_path_test, index=False)


def fold_split(X: np.array, y: np.array):
    skf = StratifiedKFold(n_splits=3)
    for train, test in skf.split(X, y):
        print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))


def split_tt_plus(not_yy: np.array, yy: np.array, cols: list, dst: str):
    """划分数据集

    Args:
        not_yy: 除了标签列的其他列，可以不只是特征
        yy: 标签
        cols: xx的列名 + yy的列名
        dst: 保存地址
    """

    not_yy_train, not_yy_test, yy_train, yy_test = \
        np.array([]).reshape(0, not_yy.shape[1]), np.array([]).reshape(0, not_yy.shape[1]), \
        np.array([]), np.array([])

    yy_count = dict(Counter(yy))
    print('数据总体情况：', yy_count)
    for k,v in yy_count.items():
        yy_sub = yy[yy==k]
        not_yy_sub = not_yy[yy==k]
        # print(yy_sub.shape, not_yy_sub.shape)
        if k == 0:
            print("{}类别的数量为{}，未放入训练集".format(k, v))
            not_yy_sub_train = np.array([]).reshape(0, not_yy.shape[1])
            not_yy_sub_test = not_yy_sub
            yy_sub_train = np.array([])
            yy_sub_test = yy_sub
        elif v > 20:
            not_yy_sub_train, not_yy_sub_test, yy_sub_train, yy_sub_test = \
                train_test_split(not_yy_sub, yy_sub, test_size=0.3, random_state=42, stratify=yy_sub)
        elif v > 6:
            if v >= 14:
                test_num = 4  # 测试集分4个样本
            elif v > 12:
                test_num = v - 10
            else:
                test_num = 2
            not_yy_sub_train = not_yy_sub[test_num:]
            not_yy_sub_test = not_yy_sub[:test_num]
            yy_sub_train = yy_sub[test_num:]
            yy_sub_test = yy_sub[:test_num]
        else:
            print("{}类别的数量为{}，未放入训练集".format(k, v))
            not_yy_sub_train = np.array([]).reshape(0, not_yy.shape[1])
            not_yy_sub_test = not_yy_sub
            yy_sub_train = np.array([])
            yy_sub_test = yy_sub

        not_yy_train = np.concatenate((not_yy_train, not_yy_sub_train))
        not_yy_test = np.concatenate((not_yy_test, not_yy_sub_test))
        yy_train = np.concatenate((yy_train, yy_sub_train))
        yy_test = np.concatenate((yy_test, yy_sub_test))

    print('训练集情况：', tuple(Counter(yy_train)))
    print('测试集情况：', tuple(Counter(yy_test)))

    train_data = pd.DataFrame(np.hstack([not_yy_train, yy_train.reshape(-1, 1)]), columns=cols)
    test_data = pd.DataFrame(np.hstack([not_yy_test, yy_test.reshape(-1, 1)]), columns=cols)

    os.makedirs(dst, exist_ok=True)
    save_path_train = os.path.join(dst, 'train_data-7_10.xlsx')
    save_path_test = os.path.join(dst, 'test_data-3_10.xlsx')
    if os.path.exists(save_path_train):
        warnings.warn(f'!!!已存在{save_path_train}，已被覆盖', Warning)
    if os.path.exists(save_path_test):
        warnings.warn(f'!!!已存在{save_path_test}，已被覆盖', Warning)
    train_data.to_excel(save_path_train, index=False)
    test_data.to_excel(save_path_test, index=False)


def count_balancing(xx, yy, cols, dst):
    """平衡数据集

    Args:
        xx: 特征
        yy: 标签
        cols: xx的列名 + yy的列名
        dst: 保存地址

    Returns:
        xx_resampled, yy_resampled
    """
    # xx_resampled, yy_resampled = ADASYN().fit_resample(xx, yy)
    xx_resampled, yy_resampled = SMOTEN().fit_resample(xx, yy)
    resampled_data = pd.DataFrame(np.hstack([xx_resampled, yy_resampled.reshape(-1, 1)]), columns=cols)

    # 保存标准化数据
    save_path = os.path.join(dst, 'resampled.xlsx')
    if os.path.exists(save_path):
        warnings.warn(f'!!!已存在{save_path}', Warning)
        cur_time = time.strftime('%m%d%H%M', time.localtime(time.time()))
        save_path = os.path.join(dst, 'resampled_{}.xlsx'.format(cur_time))
    resampled_data.to_excel(save_path)
    return xx_resampled, yy_resampled


def train_data_std(xx: np.array, dst):
    """训练标准化模型，顺便获取训练集的标准化数据

    Returns:
        mmc: 标准化模型
        xx_mmc: 标准化的数据
    """
    mmc = MinMaxScaler()
    xx_mmc = mmc.fit_transform(xx)

    # 保存标准化模型
    save_path = os.path.join(dst, 'mmc.pkl')
    if os.path.exists(save_path):
        warnings.warn(f'!!!已存在{save_path}', Warning)
        cur_time = time.strftime('%m%d%H%M', time.localtime(time.time()))
        save_path = os.path.join(dst, 'mmc_{}.pkl'.format(cur_time))
    joblib.dump(mmc, save_path)
    return mmc, xx_mmc


def test_data_std(xx: np.array, mmc=None, mmc_file=''):
    """根据标准化模型，获取数据集的标准化数据

    Args:
        xx:
        mmc: 标准化模型
        mmc_file: 标准化模型的路径

    Returns:
        xx_mmc: 标准化后的数据
    """
    if mmc is None:
        if mmc_file != '' and os.path.exists(mmc_file):
            mmc = joblib.load(mmc_file)
    xx_mmc = mmc.transform(xx)
    return xx_mmc


def train_classifying(xx, yy, dst, m='RandomForest'):
    """
    训练分类模型
    """
    if m == 'RandomForest':
        clf = RandomForestClassifier()
    elif m == 'DecisionTree':
        clf = DecisionTreeClassifier()
    elif m == 'SVC':
        clf = SVC(probability=True)
    elif m == 'LogisticRegression':
        clf = LogisticRegression()
    elif m == 'KNeighbors':
        clf = KNeighborsClassifier()
    elif m == 'GaussianNB':
        clf = GaussianNB()
    elif m == 'BernoulliNB':
        clf = BernoulliNB()
    elif m == 'AdaBoost':
        clf = AdaBoostClassifier()
    elif m == 'GradientBoosting':
        clf = GradientBoostingClassifier()
    elif m == 'HistGradientBoosting':
        clf = HistGradientBoostingClassifier()
    elif m == 'stacking':
        clf1 = RandomForestClassifier()
        clf2 = GradientBoostingClassifier()
        clf3 = HistGradientBoostingClassifier()
        lr = LogisticRegression()
        clf = StackingClassifier(estimators=[("RF", clf1), ("GB", clf2), ("HGB", clf3)], final_estimator=lr)
    else:
        raise ValueError(f"!!!没有配备{m}这种模型")

    clf.fit(xx, yy)

    # 保存分类模型
    save_path = os.path.join(dst, 'clf.pkl')
    if os.path.exists(save_path):
        warnings.warn(f'!!!已存在{save_path}', Warning)
        cur_time = time.strftime('%m%d%H%M', time.localtime(time.time()))
        save_path = os.path.join(dst, 'clf_{}.pkl'.format(cur_time))
    joblib.dump(clf, save_path)
    return clf


def test_classifying(dst, ids, xx, yy=None, clf=None, clf_file=''):
    """根据分类模型，获取测试结果

        Args:
            dst:
            ids: 行索引
            xx: 特征
            yy: 真实标签
            mmc: 分类模型
            mmc_file: 分类模型的路径
        """
    if clf is None:
        if clf_file != '' and os.path.exists(clf_file):
            clf = joblib.load(clf_file)
    yy_pre = clf.predict(xx)
    yy_prob = clf.predict_proba(xx)

    if yy is not None:
        classes = sorted(list(set(list(yy) + list(yy_pre))))
        print("classes", classes)

        report = classification_report(yy, yy_pre)
        print(report)
        confusion = confusion_matrix(yy, yy_pre, labels=classes)
        print(confusion)

        # 绘制热度图
        figure = plt.figure()
        plt.imshow(confusion, cmap=plt.cm.Oranges)
        indices = range(len(confusion))
        plt.xticks(indices, classes)
        plt.yticks(indices, classes)
        plt.colorbar()
        plt.xlabel('y_pred')
        plt.ylabel('y_true')

        # 显示数据
        for first_index in range(len(confusion)):
            for second_index in range(len(confusion[first_index])):
                plt.text(second_index, first_index, confusion[first_index][second_index])
        # print(yy_prob.shape, len(list(set(yy_pre))))
        # print(clf.classes_)
        result = pd.DataFrame(np.hstack([yy_prob, yy_pre.reshape(-1, 1), yy.reshape(-1, 1)]),
                              columns=["{}_prob".format(i) for i in clf.classes_] + ["pre", "real"],
                              index=ids)
    else:
        result = pd.DataFrame(np.hstack([yy_prob, yy_pre.reshape(-1, 1)]),
                              columns=["{}_prob".format(i) for i in clf.classes_] + ["pre"],
                              index=ids)
    # 保存分类结果
    save_path = os.path.join(dst, 'classifying_result.xlsx')
    if os.path.exists(save_path):
        warnings.warn(f'!!!已存在{save_path}', Warning)
        cur_time = time.strftime('%m%d%H%M', time.localtime(time.time()))
        save_path = os.path.join(dst, 'classifying_result_{}.xlsx'.format(cur_time))
    result.to_excel(save_path)
    return result


def copy_img_cover_y_pre(img_src, dst, result: pd.DataFrame = None, result_file='', real_c=False):
    """根据分类结果复制数据，以观察图像

    Args:
        img_src:
        dst: 
        result: 分类结果
        result_file: 分类结果保存到的excel文件
        real_c: 是否按真实类别保存分类后的数据，默认否
    """
    if result is None:
        if result_file != '' and os.path.exists(result_file):
            result = pd.read_excel(result_file, index_col='Unnamed: 0')

    if real_c and 'real' in result.columns:
        for name in tqdm(result.index):
            pre = int(result.loc[name, 'pre'])
            real = int(result.loc[name, 'real'])
            img_path = glob(os.path.join(img_src, "*", name))[0]
            if len(img_path) < 1:
                warnings.warn(f'!!!未找到{name}', Warning)
                continue
            os.makedirs(os.path.join(dst, str(pre), str(real)), exist_ok=True)
            shutil.copy(img_path, os.path.join(dst, str(pre), str(real), name))
    else:
        for name in tqdm(result.index):
            pre = int(result.loc[name, 'pre'])
            img_path = glob(os.path.join(img_src, "*", name))[0]
            if len(img_path) < 1:
                warnings.warn(f'!!!未找到{name}', Warning)
                continue
            os.makedirs(os.path.join(dst, str(pre)), exist_ok=True)
            shutil.copy(img_path, os.path.join(dst, str(pre), name))


if __name__ == "__main__":
    # # # 划分数据集
    # # excel表格路径
    # excel_file = r"\\Ds418\nas\share\LX\tmp\OQC\station_cls_ext\2_1_ext - 副本\data.xlsx"
    # # 标签列
    # target = 'label'
    # not_y, y, not_y_cols, index = get_x_y(excel_file, drop_cols=[], target=target)
    # dst = os.path.join(os.path.split(excel_file)[0], os.path.basename(excel_file).split('.')[0])
    # split_tt(not_y, y, not_y_cols+[target], dst)

    # # # 平衡训练集
    # excel_file = r"\\Ds418\NAS3\A-ZK\raw_data\1222-1226\features\432_438_439-7\train_data-7_10.xlsx"
    # drop_cols = ['defect_name_0.4', 'defect_name', 'preLb']
    # target = 'qcJdg'
    # index_col = 'defect_name_0.4'
    # X, y, X_cols = get_x_y(excel_file, drop_cols, target, index_col)
    # dst = os.path.join(os.path.split(excel_file)[0])
    # xx_resampled, yy_resampled = count_balancing(X, y, X_cols+[target], dst)

    # # 测试模型
    excel_file = r"\\Ds418\NAS\share\LX\tmp\OQC\val\val_data2\2_2-result\二阶段6、8、13\features_with_labels-step1_pre=6_wanted.xlsx"
    # drop_cols = ['jpg', 'label']
    target = 'label'
    # index_col = 'jpg'
    # X, y, X_cols, index = get_x_y(excel_file, drop_cols, index_col)
    drop_cols = ['Unnamed: 0']
    index_col = 'Unnamed: 0'
    X, y, X_cols, index = get_x_y(excel_file, drop_cols, target, index_col)

    # 模型和模型路径只要填一个
    # clf = clf
    clf_file = r"\\Ds418\NAS\share\LX\tmp\OQC\station_cls_ext\2_2_ext - 副本\try2-ok\data\二阶段6、8、13\clf.pkl"
    dst = os.path.split(excel_file)[0]
    # result = test_classifying(dst, index, X, y, clf=clf)
    test_classifying(dst, index, X, y, clf_file=clf_file)
    # test_classifying(dst, index, X, clf_file=clf_file)
