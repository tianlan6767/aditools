# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 01:27:36 2022

@author: y

AEFD: 近似等频离散化算法(approximate equal frequency discretization method)
FSFC: 基于特征聚类的特征选择方法(feature selection based on geature clustering)
"""
from scipy.stats import norm
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor as LOF
from math import log
import random
import operator


def discretization(feature:pd.Series):
    '''
    【离散化数据】
    '''
    clf = LOF()
    res = clf.fit_predict(np.array(feature).reshape(-1,1))
    inliers = feature[res > 0]
    m = np.mean(inliers)
    s = np.std(inliers)
    if m == s == 0:
        return np.zeros(len(feature))
    ppf_list = norm.ppf(q=np.array(range(1,40))/40, loc=m, scale=s)
    inds = np.digitize(feature, ppf_list)+1
    return inds


def p(feature: pd.Series):
    '''
    【计算（香农）熵】
    '''
    numEntries = len(feature)
    labelCounts = {}
    for f in feature:
        if f not in labelCounts.keys():
            labelCounts[f] = 0
        labelCounts[f] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def diction_p(feature_i: pd.Series, feature_j: pd.Series):
    '''
    【计算条件熵】
    '''
    numEntries = len(feature_i)
    labelCounts = {}
    for k in range(numEntries):
        f_i = feature_i[k]
        f_j = feature_j[k]
#         f = f'{f_i}|{f_j}'
        if f_j not in labelCounts.keys():
            labelCounts[f_j] = {}
        if f_i not in labelCounts[f_j].keys():
            labelCounts[f_j][f_i] = 0
        labelCounts[f_j][f_i] += 1
    shannonEnt = 0.0
    for key_j in labelCounts:
        tmp = 0.0
        tmp_numEntries = sum(labelCounts[key_j].values())
        for key_i in labelCounts[key_j]:
            prob = float(labelCounts[key_j][key_i])/tmp_numEntries
            tmp -= prob * log(prob, 2)
        prob_j = float(tmp_numEntries)/numEntries
        shannonEnt += prob_j * tmp
    return shannonEnt


def FSFC(df,a=None):
    '''
    【筛选特征】
    
    Parameters
    ----------
    df : 
        要进行筛选的特征数据.
    a :
        筛选特征的阈值.

    Returns
    -------
    筛选的特征.
    '''
    feature_names = list(df.columns)
    feature_num = len(df.columns)
    print('\n离散化数据')
    info = pd.DataFrame(columns=df.columns, index=df.index)  # 存储离散化的值
    for c in df.columns:
        info[c] = discretization(df[c])
    
    print('\n计算对称的不确定性')    
    SU = np.zeros((feature_num, feature_num))  # 存储相关性的值
    for i in range(SU.shape[0]):
        H_i = p(info[feature_names[i]])
        for j in range(i+1, SU.shape[1]):
            H_j = p(info[feature_names[j]])
            H_ij = diction_p(info[feature_names[i]], info[feature_names[j]])
            if H_j == H_ij:
                SU[i][j] = SU[j][i] = 0
            else:
                SU[i][j] = SU[j][i] = 2*(H_i-H_ij)/(H_i+H_j)
            
    if not a:
        print('\n求阈值')         
        SU_above_d = []  # 存储SU对角线上方元素
        for i in range(SU.shape[0]-1):
            for j in range(i+1,SU.shape[1]):
                SU_above_d.append(SU[i][j])        
        SU_above_d = np.array(SU_above_d)
        rand_num = random.randint(1, len(SU_above_d))  # 闭区间
        rand_index = random.sample(range(0,len(SU_above_d)),rand_num)  # 左闭右开
        sub_SU_above_d = SU_above_d[rand_index]
        m = np.mean(sub_SU_above_d)
        d = np.std(sub_SU_above_d)
        a = random.uniform(m, m+1/3*d)  # 阈值
        print(a)
    
    print('\n求平均相关度，找平均相关度最大的特征')
    all_rels = sum(SU)/(SU.shape[0]-1)
    all_rels_sort = sorted(enumerate(all_rels), key=operator.itemgetter(1))
    l_1_arg, l_1 = all_rels_sort[-1]
    print(l_1_arg, l_1)
    
    print('\n对特征进行聚类')
    C = [l_1_arg]  # 当前簇中的特征
    # C0 = []  # 已分到簇中的特征
    rest = set(range(feature_num))  # 还未分到簇中的特征
    r = []  # 聚类结果集
    while True:
        while True:  
            rest = rest - set(C) # 未分到簇中的特征索引
            if len(rest) == 0:  # 没有未分到簇的特征，跳出循环
                break
            rest_C_rel = {}  # rest中的特征与簇C的相关度
            for i in rest:
                tmp_sum = 0.0
                tmp = SU[i]  # 特征i的对称的不确定性
                for j in C:
                    tmp_sum += tmp[j]
                rest_C_rel[i] = tmp_sum/len(C)
            # 小于阈值的最大相关度的特征放入当前簇中，如果不存在，就跳出循环
            to_C = max(rest_C_rel.items(),key=lambda x:x[1]>=a)  
            if to_C[1] >= a:
                C.append(to_C[0])
            else:
                break
        # C0.extend(C)
        print('  当前簇的特征数量', len(C))
        r.append(C)
        # rest = rest - set(C0)
        # 从未分到簇中的特征里找到最大相关度的特征，作为新簇中的元素
        to_ = max(rest_C_rel.items(),key=lambda x:x[1]<a)
        if to_[0] in C:
            break
        else:
            C = [to_[0]]
    
    print('\n特征筛选')
    feature_subset = []
    for C in r:
        if len(C) <= 2:
            feature_subset.extend(C)
        else:
            # 特征i与簇中其他特征的平均相关度
            i_C_rel = {}
            for i in C:
                C_ = set(C) - set([i])  
                tmp_sum = 0
                tmp = SU[i]
                for j in C_:
                    tmp_sum += tmp[j]
                i_C_rel[i] = tmp_sum/len(C_)
            i_C_rel_sort = sorted(i_C_rel.items(), key=lambda item:item[1])
            to_f = i_C_rel_sort[-1]  # 平均相关度最大的特征
            feature_subset.append(to_f[0])
            
            # 获取平均相关度最大的特征的【所有对称的不确定性】
            x = SU[to_f[0]]
            x_index_and_x = enumerate(x)
            inner_C = [i for i in x_index_and_x if i[0] != to_f[0] and i[0] in C]
            inner_C_sort = sorted(inner_C,key=operator.itemgetter(1))
            min_index, min_number = inner_C_sort[0]
            feature_subset.append(min_index)
                
            # 当簇中特征的个数大于5时，获取第二大
            if len(C) > 5:
                to_f = i_C_rel_sort[-2]
                feature_subset.append(to_f[0])
    print(feature_subset)
    return [feature_names[i] for i in feature_subset]
    

if __name__ == '__main__':    
    print('读取数据')
    df = pd.read_excel(r"C:\Users\1\Desktop\df_train.xlsx", index_col='Unnamed: 0')
    df = df[df['label'] == 1]
    df = df.drop(['label'], axis=1)
    print(df.shape)
    feature_subset = FSFC(df)