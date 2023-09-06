import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque,Counter
from dpp.common.util import cal_area
from dpp.common.dpp_json import DppJson
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class JsonVis:
    def __init__(self, json_data, mini, nrows=1,area_range=[100, 500, 1000, 5000]):
        self._dj = DppJson(json_data)
        if mini:
          ncols=2
          figsize=(5, 2) 
        else:
          ncols = 4
          figsize=(9, 2)
          self._area_range = list(sorted(area_range))
        _, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=figsize)
        plt.tight_layout(h_pad=nrows, w_pad=nrows)
        for i in range(nrows*ncols):
            setattr(self, '_ax'+str(i), axes.flatten()[i])

    @property
    def _xys(self):
        all_points_x = self._dj.new_json["all_points_x"]
        all_points_y = self._dj.new_json["all_points_y"]
        return all_points_x, all_points_y

    @property
    def _area(self):
        xs, ys = self._xys
        area_list = list(sorted([cal_area(x, y) for x, y in zip(xs, ys)]))
        area_queue = deque()

        for item in self._area_range:
            if item < min(area_list):
                pass
            elif item > max(area_list):
                pass
            else:
                area_queue.append(item)
        area_queue.append(int(max(area_list)))
        area_queue.appendleft(int(min(area_list)-1))
        return area_list, area_queue

    def bar_basic(self):
        ng_ok_nums = [self._dj.ng_pcs+self._dj.ok_pcs,
                      len(self._dj.new_json["all_points_x"]), self._dj.ng_pcs, self._dj.ok_pcs]
        ng_ok_name = ["图片数", "缺陷数", "NG图片", "OK图片"]

        self._ax0.bar(np.arange(len(ng_ok_name)), ng_ok_nums,
                      tick_label=ng_ok_name, width=0.3, align='center')
        for i, v in enumerate(ng_ok_nums):
            self._ax0.text(i, v, str(v),color='red', fontsize=7,fontweight='bold')
        self._ax0.set_title('NG/OK数量分布')

    def bar_label(self):
        label_keys = [int(item) for item in list(self._dj.labels_dict.keys())]
        label_values = list(self._dj.labels_dict.values())
        labels_list = self._dj.new_json["regions"]
        
        sorted_label_values = [item[1] for item in sorted(dict(zip(label_keys,label_values)).items(),key=lambda x:x[0])]
        sorted_label_keys = sorted(label_keys)
        
        label_pcs = []
        for label in sorted_label_keys:
            begin = 0
            per_label_list = []
            for i in range(self._dj.labels_dict[str(label)]):
                per_label_list.append(labels_list.index(str(label),begin,len(labels_list)))
                begin = labels_list.index(str(label),begin,len(labels_list)) + 1
            label_pcs.append(len(set([self._dj.new_json["filename"][index] for index in per_label_list])))
            
        # self._ax1.bar(np.array(sorted_label_keys)-0.25+0.8*np.arange(len(sorted_label_keys)),  sorted_label_values,
        #               tick_label=sorted_label_keys, width = 0.5,align='center')
        # self._ax1.bar(np.array(sorted_label_keys)+0.25+0.8*np.arange(len(sorted_label_keys)), label_pcs,
        #               tick_label=sorted_label_keys, width = 0.5,align='center')
        self._ax1.bar(np.arange(len(sorted_label_keys))-0.25+np.arange(len(sorted_label_keys))*0.5,  sorted_label_values,
                      tick_label=sorted_label_keys, width = 0.5,align='center')
        self._ax1.bar(np.arange(len(sorted_label_keys))+0.25+np.arange(len(sorted_label_keys))*0.5, label_pcs,
                      tick_label=sorted_label_keys, width = 0.5,align='center')
        for i, v in enumerate(sorted_label_values):
            self._ax1.text(i-0.25+i*0.5, v, str(v), color='blue', fontsize=8,fontweight='bold')
        for i, v in enumerate(label_pcs):
            self._ax1.text(i+0.25+i*0.5, v, str(v), color='red', fontsize=6,fontweight='bold')
        self._ax1.legend(("缺陷数","图片数"),loc='best',fontsize="xx-small",borderpad=0.1,labelspacing=0.2,columnspacing=0.2,framealpha=0.5)
        self._ax1.set_title('缺陷类别数量分布')

    def scatter_hw(self):
        xs, ys = self._xys
        c = [int(x) for x in self._dj.new_json["regions"]]
        width = [max(x)-min(x) for x in xs]
        height = [max(y)-min(y) for y in ys]
        scatter = self._ax2.scatter(width, height, c=c, s=2)
        self._ax2.legend(*scatter.legend_elements(), loc='upper right',ncol=2,fontsize="xx-small",borderpad=0.1,labelspacing=0.2,columnspacing=0.2)
        self._ax2.set_title('缺陷宽高大小分布')

    def pie_area(self):
        area_list, area_queue = self._area
        explode = [0.1 for _ in range(len(area_queue)-1)]
        cats = pd.cut(area_list, area_queue)
        vc = pd.value_counts(cats)
        sizes = vc.values
        labels = [str(item) for item in vc.index]
        self._ax3.pie(sizes, labels=labels, explode=explode,
                      labeldistance=0.8, autopct='%1.1f')
        # self._ax3.legend(loc="upper right", fontsize=10,
        #                  bbox_to_anchor=(1.1, 1.05), borderaxespad=0.3)
        self._ax3.set_title('缺陷面积分布占比图')

    def __call__(self):
        for func in dir(self):
            if not func.startswith("_"):
                try:
                    getattr(self, func)()
                except:
                    pass
        plt.show()
