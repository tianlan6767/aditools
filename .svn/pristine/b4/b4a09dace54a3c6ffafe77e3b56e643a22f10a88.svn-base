import json
import os
import shutil
import time
import cv2
import shelve
import pandas as pd
import numpy as np
from typing import DefaultDict, Callable, Dict, Any, List, Union, Tuple
from PIL import Image as PImage
from tqdm import tqdm
from glob import glob
from copy import deepcopy
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter, column_index_from_string
from collections import defaultdict
from warnings import warn
from evalmodel.utils.comm import *
from evalmodel.utils.visualizer import *
from evalmodel.data_info import *
from evalmodel.filter import *
from evalmodel.utils.clip import *


__all__ = ['ModelAnalyzer']


def _draw(img_info, save_folder, region_type, compared, draw_label, draw_single, color, thickness,
          save_format, min_iou, **kwargs) -> Any:
    """封装绘制缺陷接口"""
    result = img_info.draw_regions(save_folder=save_folder, region_type=region_type, compared=compared,
                                   draw_label=draw_label, draw_single=draw_single, color=color,
                                   thickness=thickness, save_format=save_format, min_iou=min_iou, **kwargs)
    return result


class ModelAnalyzer:
    """模型分析器"""

    def __init__(self, img_folder: str, mrk_json: str, work_space: str, is_via=False):
        """
            构造函数

        :param img_folder: 图像文件夹路径
        :param mrk_json: 模型公共标注数据，若是空字符串，则自动根据图像生成OK标注
        :param work_space: 模型分析数据保存路径
        :param is_via: 指示标注数据是否是直接来源于VIA数据，若True则需要进行转换
        """

        assert mrk_json == '' or os.path.exists(mrk_json), f"json file doesn't exist. {mrk_json}"
        assert mrk_json == '' or get_json_type(mrk_json) == 'mark', f"json file isn't mark-type. {mrk_json}"
        if mrk_json:
            self.mark_data = load_via_json(mrk_json) if is_via else json.load(open(mrk_json))
        else:
            warn("Don't find mark infos, will create OK MARK infos based on the images", UserWarning)
            self.mark_data = get_ok_json(img_folder)
        assert len(self.mark_data), 'No images to evaluate!'
        self.model_infos: DefaultDict[str, ModelInfo] = defaultdict(ModelInfo)
        self._work_temp: str = ''
        self.work_space: str = work_space
        self.image_folder = img_folder

    @property
    def work_space(self) -> str:
        """模型分析数据保存路径"""
        return self._work_space

    @work_space.setter
    def work_space(self, work_space):
        """模型分析数据保存路径"""
        self._work_space = create_dir(work_space)
        self._work_temp = create_dir(os.path.join(work_space, '.temp'))

    @property
    def model_keys(self) -> List[str]:
        """分析器中包含的模型标识"""
        return [model_key for model_key in self.model_infos.keys()]

    @classmethod
    def load(cls, image_folder, mrk_json, model_folder, recursion=False, is_via=False, cached: bool = False,
             gen_box_func: Callable[[ImageInfo, Any], List[RegionInfo]] = None, **kwargs) -> 'ModelAnalyzer':
        """
            通过解析模型文件夹，获取模型分析器

            根据需求，可重载该方法
        :param image_folder: 图像文件夹
        :param mrk_json: 标注信息文件，若是空字符串，则自动根据图像生成OK标注
        :param model_folder: 模型文件夹
        :param recursion: 是否递归查询模型文件夹，搜索模型文件
        :param is_via: 指示标注数据是否是直接来源于VIA数据，若True则需要进行转换
        :param cached: 是否缓存加载信息
        :param gen_box_func: 生成目标框函数, (ImageInfo, Dict) -> List[RegionInfo]
        :param kwargs: 生成目标框函数的参数列表
        :return: 'ModelAnalyzer'
        """

        ana = cls(image_folder, mrk_json, work_space=os.path.join(model_folder, 'analyzer'), is_via=is_via)
        inf_jsons = []
        if recursion:
            for root, dirs, files in os.walk(model_folder):
                for file in files:
                    if os.path.splitext(file)[-1] == '.json':
                        inf_jsons.append(os.path.join(root, file))
        else:
            inf_jsons = glob(os.path.join(model_folder, "*.json"))
        for i, inf_json in enumerate(inf_jsons):
            if get_json_type(inf_json) == 'inf':
                model_key = get_file_infos(inf_json)[-2]
                ana.add_model(inf_json, model_key, cached, gen_box_func=gen_box_func, **kwargs)
            print(f'加载完成：{i + 1}/{len(inf_jsons)}')
        time.sleep(0.05)
        return ana

    def add_model(self, inf_json, model_key: str = None, cached: bool = False,
                  gen_box_func: Callable[[ImageInfo, Dict], List[RegionInfo]] = None, **kwargs) -> None:
        """
            添加模型数据

        :param inf_json: 推理数据
        :param model_key: 模型标识，默认为推理文件的名称
        :param cached: 是否缓存加载信息
        :param gen_box_func: 生成目标框函数, (ImageInfo, Dict) -> List[RegionInfo]
        :param kwargs: 生成目标框函数的参数列表
        :return: None
        """

        assert os.path.exists(inf_json), f'{inf_json} does NOT exist.'
        model_key = model_key if model_key else get_file_infos(inf_json)[-2]
        assert model_key not in self.model_infos, 'model info has existed!'

        with shelve.open(os.path.join(self._work_temp, f"{model_key}_10_0.4.db"), flag='c') as db:
            if model_key in db:
                print(f'解析信息: {model_key}')
                self.model_infos[model_key] = db[model_key]
            else:
                inf_data = json.load(open(inf_json))
                image_infos = []
                pro_bar = tqdm(range(len(self.mark_data) + 1), ncols=100)
                for k, v in self.mark_data.items():
                    pro_bar.set_description(f'Add Model <{model_key}>')
                    image_info = ImageInfo(self.image_folder, model_key)
                    image_info.update_region_info(v)
                    if k in inf_data:
                        image_info.update_region_info(inf_data[k], gen_box_func=gen_box_func, **kwargs)
                    image_infos.append(image_info)
                    pro_bar.update()
                self.model_infos[model_key] = ModelInfo(model_key, image_infos)
                pro_bar.update()
                pro_bar.close()
                if cached:
                    db[model_key] = self.model_infos[model_key]

    def __getitem__(self, model_key: Union[int, str]) -> ModelInfo:
        """
            通过模型标识获取模型对象

        :param model_key: 模型标识或者序列
        :return: ModelInfo
        """
        if isinstance(model_key, int) and model_key < len(self.model_infos):
            return list(self.model_infos.values())[model_key]
        elif model_key in self.model_infos:
            return self.model_infos[model_key]
        else:
            print(f'model-[{model_key}] does not exist')
            warn(f'model-[{model_key}] does not exist', UserWarning)
            return ModelInfo.empty()

    def filter_inf_regions(self, min_score=0.4, min_area=10, cached=False,
                           filter_boxes: List[RegionInfo] = None, special_scores=None) -> None:
        """
            过滤推理结果

        :param min_score: 最小得分
        :param min_area: 最小面积
        :param cached: 是否缓存加载信息
        :param filter_boxes: 过滤框。包含框，推理结果在框外就会过滤掉；排除框，推理结果在框内就会过滤掉
        :param special_scores: 特定类别的得分过滤阈值
        :return: None
        """

        pro_bar = tqdm(list(self.model_infos.keys()), ncols=80)
        for model_key in pro_bar:
            pro_bar.set_description(f'Filter {model_key} Region')
            with shelve.open(os.path.join(self._work_temp, f"{model_key}_{min_area}_{min_score}.db"), flag='c') as db:
                if model_key in db:
                    self.model_infos[model_key] = db[model_key]
                else:
                    model_info = self.model_infos[model_key]
                    model_info.filter(min_score, min_area, filter_boxes, special_scores)
                    model_info.update(model_info.min_iou)
                    if cached:
                        db[model_key] = model_info
        pro_bar.close()

    def evaluate_model_ex(self, little_sample_thresh=50, little_sample_factor=1.0, min_iou=0.1, **kwargs) -> dict:
        """
            模型评估

            通过衡量模型每个类别缺陷的检出率作为评估指标，评价公式：\r\n
            M-score = ∑(αi*Ri) \r\n
            M-score的范围是(0, 1], 值越大, 说明模型的性能越好 \r\n
            其中, αi表示i类缺陷的检出系数, Ri表示i类缺陷的检出率 \r\n
            当测试实例个数为0时，评价公式： \r\n
            M-score = 1 / (2 * over_num + 1), over_num >= 0 \r\n
        :param little_sample_thresh: 样本数量≤little_sample_thresh的样本为少数样本
        :param little_sample_factor: 少数样本与多数样本的系数比列，设定值大于1表示增强少数样本的影响，反之，小于1表示抑制
        :param min_iou: 推理与标注的最小IOU阈值(0, 1), 默认0.1
        :param kwargs: 模型评估函数兼容处理
        :return: 模型的检出率，测试数据纯OK时返回图像检出率；包含NG数据时返回缺陷检出率
        """

        assert 0 <= little_sample_factor <= 1.0, 'over_factor must be between 0 and 1!'

        dft_num = -1
        model_chk_rates = {}
        pro_bar = tqdm(range(len(self.model_infos)), desc='Eva Model', ncols=100)
        for mode_key, mode_info in self.model_infos.items():
            # 检查模型是否测试集一致，以实例个数为估计值
            if dft_num == -1:
                dft_num = mode_info.dft_num
            elif dft_num != mode_info.dft_num:
                raise ValueError('Test set is not same!')

            # 更新模型信息
            mode_info.update(min_iou)

            # 模型评估
            chk_num, over_num = mode_info.dft_chk_num, mode_info.dft_over_num
            if not dft_num:
                score = 1.0 / (2 * over_num + 1)
                model_chk_rates[mode_key] = mode_info.img_chk_rate
            else:
                dft_infos = mode_info.get_defect_desc()
                dft_labels = len(dft_infos['total'])
                dft_rates = [chk_num / dft_infos['total'][lb] for lb, chk_num in dft_infos['check_mrk'].items()]
                # 获取少数样本和多数样本
                sample_types = [1 if num > little_sample_thresh else 0 for lb, num in dft_infos['total'].items()]
                much_sample_num = sum(sample_types)
                little_sample_num = dft_labels - much_sample_num
                # 计算多数样本的检出率系数
                much_sample_alpha = 1 / (much_sample_num + little_sample_num * little_sample_factor)
                little_sample_alpha = much_sample_alpha * little_sample_factor
                # 计算评估指标
                score = 0
                for idx, dft_rate in enumerate(dft_rates):
                    score += dft_rate * (much_sample_alpha if sample_types[idx] else little_sample_alpha)
                model_chk_rates[mode_key] = chk_num / dft_num
            mode_info.score = score
            pro_bar.update()
        pro_bar.close()

        return model_chk_rates

    def evaluate_model(self, over_factor=0.5, eva_image=False, min_iou=0.1, **kwargs) -> dict:
        """
            模型评估

            评估模型检出实例的情况，评价公式：\r\n
            M-score = (check_num - over_factor * over_num) / defect_num \r\n
            M-score的范围是(-∝, 1], 值越大, 说明模型的性能越好 \r\n
            其中, over_factor表示过检的影响因素, 取值为0时表示不考虑过检 \r\n
            "*************************" \r\n
            当测试实例个数为0时，评价公式： \r\n
            M-score = 1 / (2 * over_num + 1), over_num >= 0 \r\n
            M-score的范围是(0, 1], 值越大, 说明模型的性能越好 \r\n
            "=================================" \r\n
            "=================================" \r\n
            当模型评估考虑图像维度正确检出(eva_image=True)时，评价公式：\r\n
            M-score = image_check_rate * M-score

        :param over_factor: 过检影响因子 [0, 1]，设定为0时表示不考虑过检
        :param eva_image: 模型评价是否考虑图像
        :param min_iou: 推理与标注的最小IOU阈值(0, 1), 默认0.1
        :param kwargs: 模型评估函数兼容处理
        :return: 模型的检出率，测试数据纯OK时返回图像检出率；包含NG数据时返回缺陷检出率
        """

        assert 0 <= over_factor <= 1.0, 'over_factor must be between 0 and 1!'

        dft_num = -1
        model_chk_rates = {}
        pro_bar = tqdm(range(len(self.model_infos)), desc='Eva Model', ncols=100)
        for mode_key, mode_info in self.model_infos.items():
            # 检查模型是否测试集一致，以实例个数为估计值
            if dft_num == -1:
                dft_num = mode_info.dft_num
            elif dft_num != mode_info.dft_num:
                raise ValueError('Test set is not same!')

            # 更新模型信息
            mode_info.update(min_iou)

            # 模型评估
            chk_num, over_num = mode_info.dft_chk_num, mode_info.dft_over_num
            image_factor = mode_info.img_chk_rate if eva_image else 1.0
            if not dft_num:
                score = image_factor / (2 * over_num + 1)
                model_chk_rates[mode_key] = mode_info.img_chk_rate
            else:
                score = image_factor * (chk_num - over_factor * over_num) / dft_num
                model_chk_rates[mode_key] = chk_num / dft_num
            mode_info.score = score
            pro_bar.update()
        pro_bar.close()

        return model_chk_rates

    def recommend_model(self, min_chk_rate: float = 0.5, min_iou: float = 0.1, top_k: int = 3, eva_method=0,
                        show_message=False, **kwargs) -> List[str]:
        """
            模型推荐

        :param min_chk_rate: 模型最小检出率 [0, 1], 默认0.5
        :param min_iou: 推理与标注的最小IOU阈值(0, 1), 默认0.1
        :param top_k: 模型推荐个数, 默认推荐3个最优模型
        :param eva_method: 评价方法 0-evaluate_model；1-evaluate_model_ex
        :param show_message: 控制台是否输出缺陷检出率前5的模型信息
        :param kwargs: 评价方法参数，具体参数参考对应方法
        :return: 推荐模型的标识
        """

        if len(self.model_infos) == 0:
            print('Please Add model firstly!')
            warn("Please Add model firstly!", UserWarning)
            return []

        assert 0 <= min_chk_rate <= 1.0, 'min_chk_rate must be between 0 and 1!'
        assert 0 < min_iou < 1.0, 'min_iou must be greater than 0 and less than 1  !'
        top_k = top_k if top_k > 1 else 1

        # 模型评估
        model_chk_rates = self.evaluate_model(min_iou=min_iou, **kwargs) if eva_method == 0 else \
            self.evaluate_model_ex(min_iou=min_iou, **kwargs)
        model_scores = {m.model_key: m.score for m in self.model_infos.values()}

        # 输出检出率前5的模型信息
        if show_message:
            model_chk_tops = list(model_chk_rates.keys())[:5]
            for idx, k in enumerate(model_chk_tops):
                if not idx:
                    print(f'☞检出前{len(model_chk_tops)}的模型信息：')
                print(f'{idx + 1}: {self.model_infos[k]}')

        # 获取推荐模型
        model_chk_rates = {k: model_chk_rates[k]
                           for k in list(model_chk_rates.keys()) if model_chk_rates[k] >= min_chk_rate}
        model_chk_rates = sort_dict(model_chk_rates, sort_by_key=False, reverse=True)
        model_scores = {k: model_scores[k] for k in list(model_chk_rates.keys())[:top_k * 3]}
        model_scores = sort_dict(model_scores, sort_by_key=False, reverse=True)
        if not len(model_scores):
            msg = 'No model meets the minimum check rate!' \
                if len(self.model_infos) else 'There are no models to evaluate!'
            print(msg)
            warn(msg, UserWarning)

        return list(model_scores.keys())[:top_k]

    def to_excel(self, data_type: str = 'all', show_message=False) -> None:

        """
            输出保存信息

        :param data_type: 输出的数据 ['all', 'capability', 'defect']
        :param show_message: 控制台是否输出信息
        :return: None
        """

        if data_type in ['all', 'capability']:
            model_capability = [model.get_performance() for model in self.model_infos.values() if not model.is_empty]
            self._to_excel(model_capability, '模型性能指标', None, show_message)
        if data_type in ['all', 'defect']:
            defect_infos = {k: model.get_defect_desc() for k, model in self.model_infos.items() if not model.is_empty}
            defect_capability = []
            for k, v in defect_infos.items():
                capability = {'model_key': k}
                for lb, n in v['total'].items():
                    capability[f"{lb}_miss"] = v['miss'][lb]
                    capability[f"{lb}_check"] = v['check_mrk'][lb]
                    capability[f"{lb}_rate"] = round(v['check_mrk'][lb] / n, 4)
                defect_capability.append(capability)
            self._to_excel(defect_capability, '模型缺陷指标', None, show_message)

    def _to_excel(self, data, excel_name: str = None, sheet_name: str = None,
                  show_message=False, freeze_panes: Union[int, str] = None) -> None:
        """
            保存模型性能指标到excel中

        :param data: 需保存的数据
        :param excel_name: excel文件名称，默认为 ‘模型性能指标’
        :param sheet_name: excel表名称
        :param show_message: 控制台是否输出模型信息
        :param freeze_panes: 冻结第二行第几列窗口（默认不冻结窗口）, (1 -> 'A')
        :return:
        """

        if not len(data):
            return

        excel_name = excel_name if excel_name else '指标文件'
        file_name = excel_name if excel_name.endswith('.xlsx') else f'{excel_name}.xlsx'
        file = os.path.join(self.work_space, file_name)
        capability = pd.DataFrame(data)

        def _save(f):
            work_book = None
            try:
                if os.path.exists(f):
                    work_book = load_workbook(f)
                with pd.ExcelWriter(f, engine='openpyxl') as writer:
                    if work_book:
                        writer.book = work_book
                    sh_name = f'ana{len(self.model_infos)}_{get_cur_time()}' if sheet_name is None else sheet_name
                    capability.to_excel(excel_writer=writer, sheet_name=sh_name, index=True)

                    # 冻结窗口
                    worksheet = writer.sheets[sh_name]
                    if freeze_panes:
                        col_letter = get_column_letter(freeze_panes) if isinstance(freeze_panes, int) else freeze_panes
                        worksheet.freeze_panes = f'{col_letter}2'
                print(f'保存{excel_name}文件成功！"{f}:0"')
            except Exception as e:
                raise e
            finally:
                if work_book:
                    work_book.close()

        try:
            _save(file)
        except PermissionError:
            print(f'保存{excel_name}异常！尝试重命名进行保存。')
            t = time.strftime('%H%M%S', time.localtime(time.time()))
            temp_file = file.replace('.xlsx', f'_{t}.xlsx')
            if os.path.exists(file):
                shutil.copy(file, temp_file)
            _save(temp_file)

        if show_message:
            print(capability.to_string())

    def _get_model_infos(self, keys: Union[str, List[str]]) -> List[ModelInfo]:
        """通过关键词获取模型信息对象"""
        if isinstance(keys, str):
            model_infos = [self[keys]] if not self[keys].is_empty else []
        else:
            model_infos = [self[k] for k in keys if not self[k].is_empty]
        return model_infos

    def get_model_scores(self, keys: Union[str, List[str]], score_type: str) -> dict:
        """
            获取模型检测得分

        :param keys: 模型标识，支持输入单个标识或者标识列表
        :param score_type: 得分类型, ['check', 'mrk_check', 'inf_check', 'over', 'inf'], check和mrk_check等效
        :return: 得分
        """

        model_infos = self._get_model_infos(keys)
        scores = {m.model_key: m.get_scores(score_type) for m in model_infos}
        return scores

    def get_score_hist(self, keys: Union[str, List[str]], score_type: str, show_hist=False) -> dict:
        """
            获取模型得分分布

        :param keys: 模型标识，支持输入单个标识或者标识列表
        :param score_type: 得分类型, ['check', 'mrk_check', 'inf_check', 'over', 'inf'], check和mrk_check等效
        :param show_hist: 是否显示图像
        :return: 得分
        """

        scores = self.get_model_scores(keys, score_type)
        model_infos = self._get_model_infos(keys)
        for model_info in model_infos:
            k = model_info.model_key
            name = f'{score_type.capitalize()}Score'
            save_folder = create_dir(os.path.join(self.work_space, 'hist_scores'))
            show_feature_hist(pd.Series(scores[k], name=name, dtype='float64'), subtitle=k,
                              bins=[0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 1.0],
                              save_path=os.path.join(save_folder, f'{k}_{name}.jpg'), show_hist=show_hist)
        return scores

    def get_iou_hist(self, keys: Union[str, List[str]], show_hist=False) -> dict:
        """
            获取模型IOU分布

        :param keys: 模型标识，支持输入单个标识或者标识列表
        :param show_hist: 是否显示图像
        :return: IOU
        """

        model_infos = self._get_model_infos(keys)
        ious = {}
        for model_info in model_infos:
            k = model_info.model_key
            ious[k] = []
            for image_info in model_info.img_infos:
                for region_info in image_info.get_check_regions(model_info.min_iou, 'inf'):
                    ious[k].append(region_info.iou)

            name = 'CheckIOU'
            save_folder = create_dir(os.path.join(self.work_space, 'hist_ious'))
            show_feature_hist(pd.Series(ious[k], name=name, dtype='float64'), subtitle=k,
                              save_path=os.path.join(save_folder, f'{k}_{name}.jpg'), show_hist=show_hist)
        return ious

    def get_over_pie(self, keys: Union[str, List[str]], show_pie=False) -> dict:
        """
            获取过检数据类型分布

        :param keys: 模型标识，支持输入单个标识或者标识列表
        :param show_pie: 是否显示图像
        :return: 类型分布频率
        """

        model_infos = self._get_model_infos(keys)
        classes = {}
        for model_info in model_infos:
            k = model_info.model_key
            classes[k] = defaultdict(int)
            for image_info in model_info.img_infos:
                for region_info in image_info.get_over_regions(model_info.min_iou):
                    classes[k][int(region_info.label)] += 1

            classes[k] = sort_dict(classes[k], reverse=True, sort_by_key=False)
            name = 'OverDefect'
            save_folder = create_dir(os.path.join(self.work_space, 'hist_overs'))
            show_feature_pie(pd.Series(classes[k], name=name, dtype='float64'), title=f'缺陷过检类型分布 - {k}',
                             save_path=os.path.join(save_folder, f'{k}_{name}.jpg'), show_hist=show_pie)
        return classes

    def get_image_hist(self, keys: Union[str, List[str]], show_hist=False) -> dict:
        """
            获取模型图像检出分布

        :param keys: 模型标识，支持输入单个标识或者标识列表
        :param show_hist: 是否显示图像
        :return:
        """

        model_infos = self._get_model_infos(keys)
        if len(model_infos) == 0:
            return {}

        data = {}
        for model_info in model_infos:
            image_infos = model_info.get_image_desc()
            data[model_info.model_key] = [
                image_infos['ng_image_check_num'],
                image_infos['ok_image_check_num'],
                image_infos['image_miss_num'],
                image_infos['image_over_num'],
            ]

        save_folder = create_dir(os.path.join(self.work_space, 'hist_images'))
        show_feature_bar(data,
                         categories=['Check-NG', 'Check-OK', 'Miss', 'Over'],
                         title='图像检出详情',
                         save_path=os.path.join(save_folder, f'{len(model_infos)}_images.jpg'),
                         show_hist=show_hist)
        return data

    def get_defect_detail(self, keys: Union[str, List[str]], show_hist=False, show_detail=False) -> dict:
        """
            获取模型缺陷检出情况

        :param keys: 模型标识，支持输入单个标识或者标识列表
        :param show_hist: 是否显示图像
        :param show_detail: 是否输出检出详情
        :return: 缺陷信息
        """

        if isinstance(keys, str):
            keys = [keys]

        defect_infos = {}
        for ky in keys:
            if self[ky].is_empty:
                continue

            save_folder = create_dir(os.path.join(self.work_space, 'defect_detail'))

            defect_desc = self[ky].get_defect_desc()
            show_defect_detail(defect_desc['check_mrk'], defect_desc['miss'], f'缺陷检出详情 - {ky}',
                               save_path=os.path.join(save_folder, f'{ky}_MarkDetail.jpg'), show_hist=show_hist)
            defect_infos[ky] = defect_desc

            if show_detail:
                checked_rates = {}
                for k, v in defect_desc['total'].items():
                    detail = f"{defect_desc['check_mrk'][k]}/{v}"
                    checked_rates[k] = f"{detail:<10}= {defect_desc['check_mrk'][k] / v:.2%}"
                print(f'模型{ky}的缺陷检出分布 \r\n{dict_to_str(checked_rates)}')

        return defect_infos

    def get_defect_line(self, keys: Union[str, List[str]], show_hist=False, show_detail=False,
                        labels: List[int] = None) -> dict:
        """
            获取缺陷各个维度的对比

            注意：统计推理标签对应的缺陷检出时，确保对比模型的标签含义是否一致，否则对比无意义！！！
        :param keys: 对比缺陷检出的模型标识
        :param show_hist: 是否显示图表
        :param show_detail: 是否输出信息
        :param labels: 缺陷标签，默认统计标注标签对应缺陷的检出；设定为模型推理标签时，表示统计推理标签对应缺陷的检出。
        :return: 模型缺陷维度的检出信息
        """

        if isinstance(keys, str):
            keys = [keys]

        defect_capability = {}
        lbs = []
        for ky in keys:
            if self[ky].is_empty:
                continue

            defect_desc = self[ky].get_defect_desc()
            defect_capability[ky] = []
            if not labels:
                for lb, n in defect_desc['total'].items():
                    defect_capability[ky].append(round(defect_desc['check_mrk'][lb] / n, 4))
                    if len(lbs) < len(defect_desc['total']):
                        lbs.append(lb)
            else:
                lbs = lbs if lbs else sorted(labels)
                for lb in lbs:
                    if lb in defect_desc['check_inf']:
                        defect_capability[ky].append(defect_desc['check_inf'][lb])
                    else:
                        warn(f'Inf label maybe wrong!', UserWarning)
                        defect_capability[ky].append(0)
        data = pd.DataFrame(defect_capability, index=[f'{lb}' for lb in lbs])
        save_folder = create_dir(os.path.join(self.work_space, 'defect_line'))
        show_defect_plot(data, title='缺陷检出变化', y_label='检出数目' if labels else '检出率', show_hist=show_hist,
                         save_path=os.path.join(save_folder, f'{len(defect_capability)}_defect.jpg'))

        if show_detail:
            print(data.to_string())

        return defect_capability

    def _get_image(self, keys: Union[str, List[str]], image_type='miss', region_type='mark', compared=False,
                   draw_label=True, draw_single=False, color=(255, 0, 0), thickness=1, save_format='.bmp',
                   **kwargs) -> None:
        """
            获取图像

        :param keys: 模型标识，支持输入单个标识或者标识列表
        :param image_type: 图像类型，['miss', 'over', 'check', 'all', 'abs-over']
        :param region_type: 绘制区域类型，['miss', 'over', 'check', 'mark', 'inf', 'cmp', 'all', 'score', 'iou', ...]
        :param compared: 输出时，是否左侧添加对比图像
        :param draw_label: 是否绘制标签
        :param draw_single: 是否将每个缺陷作为独立个体绘制在图像上, False:将所有区域绘制在同一张图上, True:将每个区域绘制在独自图像上
        :param color: 区域绘制颜色, region_type='all' 时，指的推理的区域颜色
        :param thickness: 绘制区域边界线条宽度，≥1
        :param save_format: 保存格式，查看局部图时，建议为.bmp格式；查看整图时，建议.jpg格式
        :param kwargs:
        :return: None
        """

        if region_type == 'cmp':
            self.classify_defect(keys, True)
        else:
            model_infos = self._get_model_infos(keys)
            for model_info in model_infos:
                k = model_info.model_key
                if region_type in ['score', 'check-score', 'over-score', 'inf-score', 'iou']:
                    assert 'min_feature' in kwargs and 'max_feature' in kwargs, kwargs
                    features = self.get_score_hist(k, region_type.split('-')[0])[k] if 'score' in region_type \
                        else self.get_iou_hist(k)[k]
                    if len(features) == 0:
                        print(f"Model - [{k}] doesn't have {region_type} infos, "
                              f"maybe {region_type} of defect is too small.")
                        continue
                    if min(features) > kwargs['max_feature']:
                        print(f"Model - [{k}] doesn't have images with a {region_type} of "
                              f"{kwargs['min_feature']} to {kwargs['max_feature']}")
                        continue

                image_infos = model_info.get_images(image_type)
                if len([img for img in image_infos if not img.exists]):
                    warn(f'ImageFolder: {self.image_folder} maybe wrong!', UserWarning)

                if len(image_infos):
                    # TODO 通过多线程/多进程进行优化，但最终结果受SSD和HDD影响
                    # 默认使用单线程处理图像，若图像放在SSD上，推荐使用多线程处理
                    run_mode = 'multi_thread'  # ['single_thread', 'multi_thread', 'multi_process']
                    save_folder = create_dir(os.path.join(self.work_space, 'images', f'{k}_{image_type}'))
                    image_save_folder = r''
                    desc = f'{image_type.capitalize()} Image' if region_type == 'all' else \
                        f'{region_type.capitalize()} Defect'

                    if run_mode == 'multi_process':
                        image_save_folder = mul_process(image_infos, _draw, pro_num=os.cpu_count(),
                                                        desc=f'Model-[{k}] Outputting {desc}',
                                                        save_folder=save_folder, region_type=region_type,
                                                        compared=compared, draw_label=draw_label,
                                                        draw_single=draw_single, color=color, thickness=thickness,
                                                        save_format=save_format, min_iou=model_info.min_iou, **kwargs)
                    elif run_mode == 'multi_thread':
                        image_save_folder = mul_thread(image_infos, _draw, td_num=os.cpu_count(),
                                                       desc=f'Model-[{k}] Outputting {desc}',
                                                       save_folder=save_folder, region_type=region_type,
                                                       compared=compared, draw_label=draw_label,
                                                       draw_single=draw_single, color=color, thickness=thickness,
                                                       save_format=save_format, min_iou=model_info.min_iou, **kwargs)
                    else:
                        pro_bar = tqdm(image_infos, desc=f'Model-[{k}] Outputting {desc}')
                        for image_info in pro_bar:
                            image_save_folder = image_info.draw_regions(save_folder, region_type, compared, draw_label,
                                                                        draw_single, color, thickness, save_format,
                                                                        min_iou=model_info.min_iou, **kwargs)
                        pro_bar.close()

                    if image_save_folder:
                        print(f'{desc} 获取完成，请打开文件夹查看。"{image_save_folder}:0"')

                    time.sleep(0.05)
                else:
                    print(f"Model-[{k}] doesn't have {image_type} image!")

    def classify_defect(self, keys: Union[str, List[str]], save_defect=False) -> dict:
        """
            根据缺陷的检出情况，将标注缺陷进行分类

            分类的类别由输入模型对该缺陷的检出情况决定的，例如：\r\n
            假设输入对比模型个数为4，某个缺陷在模型1和3中能够检出，则该缺陷分类标签为1010
        :param keys: 对比模型的标识，注意：分类标签和模型顺序相关哦
        :param save_defect: 是否保存defect的对比图
        :return: 缺陷的分类信息，得分信息
        """

        model_infos = self._get_model_infos(keys)
        if len(model_infos) < 1 or ((len(model_infos) > 4 or len(model_infos) <= 1) and save_defect):
            print(f"The number of comparison models must be greater than 1 and less than 5! \n"
                  f"The current number of comparison models is {len(model_infos)}")
            return {}

        save_folder = os.path.join(self.work_space, 'images', f'comparison_{get_cur_time()}')
        result = deepcopy(self.mark_data)
        # 删除没有推理的图像信息
        detected_keys = [i.full_name for i in model_infos[0].img_infos]
        for k in list(result.keys()):
            if k not in detected_keys:
                result.pop(k)
        # 分类缺陷
        pro_bar = tqdm(result.keys(), ncols=80, desc='生成缺陷对比图' if save_defect else '获取对比信息', delay=0.1)
        for img_idx, name in enumerate(pro_bar):
            dft_num = len(result[name]['regions'])
            dft_infos = [[] for _ in range(dft_num)]
            dft_scores = [[] for _ in range(dft_num)]
            dft_ctr_scores = [[] for _ in range(dft_num)]
            dft_cls_scores = [[] for _ in range(dft_num)]
            dft_fpn_levels = [[] for _ in range(dft_num)]
            # 分类缺陷
            for model_info in model_infos:
                img_info = model_info.img_infos[img_idx]
                dft_info = ['0' for _ in range(dft_num)]
                dft_score = [0 for _ in range(dft_num)]
                dft_ctr_score = [0 for _ in range(dft_num)]
                dft_cls_score = [0 for _ in range(dft_num)]
                dft_fpn = [-1 for _ in range(dft_num)]
                chk_regions = img_info.get_check_regions(model_info.min_iou)
                for chk_region in chk_regions:
                    dft_info[chk_region.idx] = '1'
                    dft_score[chk_region.idx] = chk_region.IR.score
                    dft_fpn[chk_region.idx] = chk_region.IR.get_feature('fpn_levels')
                    dft_ctr_score[chk_region.idx] = chk_region.IR.get_feature('ctr_score')
                    dft_cls_score[chk_region.idx] = chk_region.IR.get_feature('cls_score')
                for dft_idx, info in enumerate(dft_info):
                    dft_infos[dft_idx].append(info)
                    dft_scores[dft_idx].append(dft_score[dft_idx])
                    dft_ctr_scores[dft_idx].append(dft_ctr_score[dft_idx])
                    dft_cls_scores[dft_idx].append(dft_cls_score[dft_idx])
                    dft_fpn_levels[dft_idx].append(dft_fpn[dft_idx])
            # 更新结果
            for dft_idx, dft_info in enumerate(dft_infos):
                result[name]['regions'][dft_idx]['region_attributes']['detect'] = dft_info
                result[name]['regions'][dft_idx]['region_attributes']['scores'] = dft_scores[dft_idx]
                result[name]['regions'][dft_idx]['region_attributes']['ctr_scores'] = dft_ctr_scores[dft_idx]
                result[name]['regions'][dft_idx]['region_attributes']['cls_scores'] = dft_cls_scores[dft_idx]
                result[name]['regions'][dft_idx]['region_attributes']['fpn'] = dft_fpn_levels[dft_idx]
            # 检出图像是否存在
            img_info = model_infos[0].img_infos[img_idx]
            if not img_info.exists or not save_defect:
                continue
            # 输出缺陷
            src_img = cv2.imdecode(np.fromfile(img_info.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            img_size = src_img.shape[:2]
            for dft_idx, dft_info in enumerate(dft_infos):
                dst_folder = create_dir(os.path.join(save_folder, ''.join(dft_info),
                                                     img_info.mark_regions[dft_idx].label))
                # 绘制当前标注缺陷信息
                drawn_img = draw_via_region(src_img, img_info.mark_regions[dft_idx].json_info,
                                            (0, 0, 255), 1, True, False, inplace=False)
                box, anchor = get_via_region_boundary(img_info.mark_regions[dft_idx].json_info, 50, img_size)
                df_mrk_images = [drawn_img[box[1]:box[3], box[0]:box[2]]]
                df_cmp_img = src_img[box[1]:box[3], box[0]:box[2]]
                for model_idx, info in enumerate(dft_info):
                    if len(df_mrk_images) == 1:
                        drawn_img = draw_via_region(src_img, img_info.mark_regions[dft_idx].json_info,
                                                    (0, 0, 255), 1, False, False, inplace=False)
                    if info != '1':
                        df_mrk_images.append(drawn_img[box[1]:box[3], box[0]:box[2]])
                    else:
                        img_info = model_infos[model_idx].img_infos[img_idx]
                        cmp_img = draw_via_region(drawn_img, img_info.mark_regions[dft_idx].IR.json_info,
                                                  (255, 0, 0), 1, True, True, inplace=False)
                        df_mrk_images.append(cmp_img[box[1]:box[3], box[0]:box[2]])
                border = 2
                df_mrk_num = len(df_mrk_images)
                bg_img = PImage.new('RGB',
                                    ((box[2] - box[0]) * (df_mrk_num + 1) + border * (df_mrk_num + 2),
                                     (box[3] - box[1]) + border * 2),
                                    "white")
                bg_img.paste(PImage.fromarray(df_cmp_img), (border, border))
                for df_mrk_idx, df_mrk_img in enumerate(df_mrk_images):
                    bg_img.paste(PImage.fromarray(df_mrk_img),
                                 ((box[2] - box[0]) * (df_mrk_idx + 1) + border * (df_mrk_idx + 2), border))
                df_mrk_name = f'{os.path.splitext(img_info.full_name)[0]}_{dft_idx}'
                save_file = os.path.join(dst_folder, f'{df_mrk_name}_{"".join(dft_info)}.bmp')
                # PImage和cv2的保存方式不同cv2是bgr,PImage是rgb
                cv2.imencode('.bmp', np.array(bg_img))[1].tofile(save_file)
        pro_bar.close()
        return result

    def save_defect(self, keys: Union[str, List[str]]) -> None:
        """
            保存缺陷对比信息

        :param keys: 对比模型关键词
        :return: None
        """

        defect_infos = self.classify_defect(keys, save_defect=False)
        defect_data = []
        for k, v in defect_infos.items():
            name = os.path.splitext(k)[0]
            for idx, defect_info in enumerate(v['regions']):
                scores = defect_info['region_attributes']['scores']
                ctr_scores = defect_info['region_attributes']['ctr_scores']
                cls_scores = defect_info['region_attributes']['cls_scores']
                detect = [int(i) for i in defect_info['region_attributes']['detect']]
                fpn = defect_info['region_attributes']['fpn']
                region_info = RegionInfo.from_json(defect_info, False)
                data = dict(name=f'{name}_{idx}',
                            mark_label=int(defect_info['region_attributes']['regions']),
                            mark_width=region_info.width,
                            mark_height=region_info.height,
                            model_num=len(scores),
                            mean_score=sum(scores) / len(scores),
                            max_score=max(scores),
                            min_score=min(scores),
                            chk_pro=sum(detect) / len(detect))
                for i, model_info in enumerate(self._get_model_infos(keys)):
                    data[f'{model_info.model_key}_score'] = scores[i]
                for i, model_info in enumerate(self._get_model_infos(keys)):
                    data[f'{model_info.model_key}_ctr_score'] = ctr_scores[i]
                for i, model_info in enumerate(self._get_model_infos(keys)):
                    data[f'{model_info.model_key}_cls_score'] = cls_scores[i]
                for i, model_info in enumerate(self._get_model_infos(keys)):
                    data[f'{model_info.model_key}_fpn_level'] = fpn[i]
                defect_data.append(data)
        cur_time = get_cur_time()
        self._to_excel(defect_data, '缺陷统计信息', sheet_name=f'defect_{cur_time}', show_message=False, freeze_panes='C')

    def save_over_defect(self, keys: Union[str, List[str]]):
        model_infos = self._get_model_infos(keys)
        if len(model_infos) == 0:
            return

        for model_info in model_infos:
            defect_data = []
            for img_info in model_info.get_images(image_type='all'):
                for dft in img_info.get_over_regions(model_info.min_iou):
                    data = dict(name=f'{img_info.name}_{dft.idx}',
                                mark_label=int(dft.label),
                                score=dft.score,
                                ctr_score=dft.get_feature('ctr_score'),
                                cls_score=dft.get_feature('cls_score'),
                                )
                    defect_data.append(data)
            cur_time = get_cur_time()
            self._to_excel(defect_data, f'{model_info.model_key}过检缺陷信息', sheet_name=f'over_defect_{cur_time}')

    def get_image(self, keys: Union[str, List[str]], image_type: str, compared=False, save_format='.jpg') -> None:
        """
            获取指定类型推理图像

        :param keys: 模型标识，支持输入单个标识或者标识列表
        :param image_type: 图像类型，['miss', 'over', 'check', 'all', 'abs-over']
        :param compared: 输出时，是否左侧添加对比图像
        :param save_format: 保存格式
        :return: None
        """

        self._get_image(keys, image_type=image_type, region_type='all', compared=compared, draw_single=False,
                        save_format=save_format)

    def get_defect(self, keys: Union[str, List[str]], defect_type: str, compared=True, save_format='.bmp') -> None:
        """
            获取指定类型缺陷图像

        :param keys: 模型标识，支持输入单个标识或者标识列表
        :param defect_type: 缺陷类型，['miss', 'over', 'check', 'mark', 'inf', 'cmp']
        :param compared: 输出时，是否左侧添加对比图像
        :param save_format: 保存格式
        :return: None
        """

        self._get_image(keys, image_type='all', region_type=defect_type, compared=compared, draw_single=True,
                        save_format=save_format)

    def get_feature_defect(self, keys: Union[str, List[str]], feature_name: str, min_feature: float, max_feature: float,
                           compared=True, save_format='.bmp') -> None:
        """
            获取得分(min_score, max_score)区间内的缺陷

        :param keys: 模型标识
        :param feature_name: 缺陷特征名称, ['check-score', 'over-score', 'inf-score', 'iou', ...]
        :param min_feature: 缺陷特征最小值
        :param max_feature: 缺陷特征最大值
        :param compared: 输出时，是否左侧添加对比图像
        :param save_format: 保存格式
        :return: None
        """

        self._get_image(keys, image_type='all', region_type=feature_name, compared=compared, draw_single=True,
                        save_format=save_format, min_feature=min_feature, max_feature=max_feature)

    def get_custom_defect(self, keys: Union[str, List[str]]):
        model_infos = self._get_model_infos(keys)
        for model_info in model_infos:
            k = model_info.model_key
            image_infos = model_info.get_images('all')
            if len([img for img in image_infos if not img.exists]):
                warn(f'ImageFolder: {self.image_folder} maybe wrong!', UserWarning)

            if len(image_infos):
                save_folder = create_dir(os.path.join(self.work_space, 'images', f'{k}_custom'))
                pro_bar = tqdm(range(len(image_infos)), desc=f'Model-[{k}] Outputting Custom Defect')
                for image_info in image_infos:
                    # 自定义条件
                    regions = [region for region in image_info.inf_regions
                               if region.iou > 0.0001 and region.score < 0.5]
                    image_info.draw(save_folder, regions, region_type='score',
                                    compared=False, draw_label=True, draw_single=True)
                    pro_bar.update()
                pro_bar.close()

    def gen_report(self, keys: Union[str, List[str]], reporter: str) -> None:
        """
            生成评价报告

        :param keys: 模型标识，一般为推荐模型返回值; 标识的先后顺序代表其性能优劣
        :param reporter: 创建人
        :return: None
        """

        model_infos = self._get_model_infos(keys)
        if len(model_infos) == 0:
            print('No model needs to reported!')
            return

        from evalmodel.reporter import gen_model_report
        gen_model_report(self, model_infos, reporter=reporter)

    def get_ambiguous_sample(self, keys: Union[str, List[str]], score_thresh: Tuple[float, float] = (0.3, 0.5),
                             model_prob: Tuple[float, float] = (0.5, 0.9), benchmark: str = None) -> Tuple[dict, dict]:
        """
            获取模棱两可/模糊的数据

            单个输入模型：score_thresh[0]≤score≤score_thresh[1]的缺陷所属的样本

            多个输入模型：基准模型中score_thresh[0]≤score≤score_thresh[1]且
                        model_prob[1]≥多个模型推理检出的概率≥model_prob[0]的缺陷所属的样本
        :param keys: 输入模型的关键词
        :param score_thresh: 基准模型中缺陷的得分过滤阈值
        :param model_prob: 多个模型推理检出的概率阈值
        :param benchmark: 基准模型的关键词，默认第一个输入模型为基准模型
        :return: 模糊样本信息和掩模样本信息
        """

        model_infos = self._get_model_infos(keys)
        ambiguous_samples, mask_samples = {}, {}
        if len(model_infos) == 0:
            return ambiguous_samples, mask_samples

        # 获取基准模型
        benchmark_model = model_infos[0]
        if benchmark and not self[benchmark].is_empty:
            benchmark_model = self[benchmark]

        # 删除没有推理的图像信息
        ambiguous_samples = deepcopy(self.mark_data)
        detected_keys = [i.full_name for i in benchmark_model.img_infos]
        for k in list(ambiguous_samples.keys()):
            if k not in detected_keys:
                ambiguous_samples.pop(k)

        # 获取模糊样本
        pro_bar = tqdm(list(ambiguous_samples.keys()), ncols=100, desc='获取模糊样本')
        for img_idx, name in enumerate(pro_bar):
            img_info = benchmark_model.img_infos[img_idx]
            chk_regions = img_info.get_check_regions(benchmark_model.min_iou)
            # 筛选条件1：筛选基准模型中0.3≤score≤0.5的缺陷
            retain_ids = [chk_region.idx for chk_region in chk_regions
                          if score_thresh[0] <= chk_region.IR.score <= score_thresh[1]]
            # 筛选条件2：筛选0.9≥多模型推理得出概率≥0.5的缺陷
            if len(model_infos) > 1 and len(retain_ids) > 0:
                chk_ids = []
                for model_info in model_infos:
                    img_info = model_info.img_infos[img_idx]
                    chk_regions = img_info.get_check_regions(model_info.min_iou)
                    # 统计检出缺陷的Index
                    chk_ids.extend([chk_region.idx for chk_region in chk_regions])
                for retain_idx in deepcopy(retain_ids):
                    # 计算符合得要要求缺陷所对应的模型检出概率
                    chk_prob = len([i for i in chk_ids if i == retain_idx]) / len(model_infos)
                    if chk_prob < model_prob[0] or chk_prob > model_prob[1]:
                        retain_ids.remove(retain_idx)
            # 更新输出
            if len(retain_ids) == 0:
                ambiguous_samples.pop(name)
            else:
                region_num = len(ambiguous_samples[name]['regions'])
                del_ids = [i for i in range(region_num) if i not in retain_ids]
                del_ids = sorted(del_ids, reverse=True)
                mask_samples[name] = {'filename': name, 'regions': []}
                for idx in del_ids:
                    mask_samples[name]['regions'].append(ambiguous_samples[name]['regions'].pop(idx))
            pro_bar.set_postfix(sample=len(ambiguous_samples),
                                defect=sum([len(v['regions']) for k, v in ambiguous_samples.items()]),
                                mask=sum([len(v['regions']) for k, v in mask_samples.items()]))
        pro_bar.close()

        return ambiguous_samples, mask_samples

    def create_new_sample(self, roi_samples: dict, mask_samples: dict, ps_idx: int) -> None:
        """

        :param roi_samples: 需增强关注的样本
        :param mask_samples: 需掩模忽略的样本
        :param ps_idx: 生成样本的序列标识
        :return: None
        """

        assert len(roi_samples) == len(mask_samples), 'check input data!'
        save_path = create_dir(os.path.join(self.work_space, 'augmentation', get_cur_time()))
        pro_bar = tqdm(list(mask_samples.keys()), ncols=100, desc='生成样本')
        for k in pro_bar:
            img_path = os.path.join(self.image_folder, k)
            img_name, img_ext = get_file_infos(img_path)[2:]
            if os.path.exists(img_path):
                aug_img_name = f'{img_name}_ps_{ps_idx}{img_ext}'
                aug_img_path = os.path.join(save_path, aug_img_name)
                # 创建增强图像
                if len(mask_samples[k]['regions']):
                    # 根据掩模信息，PS缺陷
                    src_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    mark_img = np.zeros(src_img.shape, dtype=np.uint8)
                    for region in mask_samples[k]['regions']:
                        region_shape = region['shape_attributes']
                        region_points = [(int(region_shape['all_points_x'][i]), int(region_shape['all_points_y'][i]))
                                         for i in range(len(region_shape['all_points_x']))]

                        # region_left, region_right = min(region_shape['all_points_x']), \
                        #                             max(region_shape['all_points_x'])
                        # region_top, region_bot = min(region_shape['all_points_y']), max(region_shape['all_points_y'])
                        # region_width = region_right - region_left
                        # region_height = region_bot - region_top
                        # if region_width <= 50 and region_height <= 50:
                        #     roi_top = region_top - region_height - 5
                        #     if roi_top < 0:
                        #         roi_top = region_bot + 5
                        #     roi_bot = roi_top + region_height
                        #     src_img[region_top:region_bot, region_left:region_right] = \
                        #         src_img[roi_top:roi_bot, region_left:region_right]
                        # else:
                        #     temp = np.zeros(src_img.shape, dtype=np.uint8)
                        #     cv2.fillPoly(temp, [np.array(region_points)], color=255)
                        #     split_height = 20
                        #     split_num = region_height // split_height + (1 if region_height % split_height else 0)
                        #     for split_idx in range(split_num):
                        #         split_top = region_top + split_idx * split_height
                        #         split_bot = split_top + split_height
                        #         if split_bot > src_img.shape[0]:
                        #             split_bot = src_img.shape[0]
                        #             split_top = split_bot - split_height
                        #         roi_top = split_top - split_height - 5
                        #         if roi_top < 0:
                        #             roi_top = region_bot + 5
                        #         roi_bot = roi_top + split_height
                        #
                        #         roi = temp[split_top:split_bot, 0:src_img.shape[1]]
                        #         gray_val = ((np.sum(roi, axis=0) / 255).astype(np.int32)).tolist()
                        #         split_left, split_right = -1, -1
                        #         for i, val in enumerate(gray_val):
                        #             if val > 0 and split_left == -1:
                        #                 split_left = i
                        #             if val == 0 and split_left != -1 and split_right == -1:
                        #                 split_right = i
                        #
                        #         src_img[split_top:split_bot, split_left:split_right] = \
                        #             src_img[roi_top:roi_bot, split_left:split_right]

                        cv2.fillPoly(mark_img, [np.array(region_points)], color=255)
                    dst_img = cv2.inpaint(src_img, mark_img, 5, cv2.INPAINT_NS)
                    cv2.imencode(img_ext, np.array(dst_img))[1].tofile(aug_img_path)
                else:
                    shutil.copy(img_path, aug_img_path)
                # 同时更新标注信息
                info = roi_samples.pop(k)
                info['filename'] = aug_img_name
                roi_samples[aug_img_name] = info
            else:
                roi_samples.pop(k)

        save_json(roi_samples, save_path, 'data')

    def get_abnormal_samples(self, keys: Union[str, List[str]],
                             abnormal_score: float = 0.8, abnormal_area: int = 50) -> dict:
        """
            获取异常样本-得分高于abnormal_score的过检数据

        :param keys: 输入模型的关键词
        :param abnormal_score: 异常得分最低阈值
        :param abnormal_area: 异常面积最低阈值
        :return: 异常样本信息
        """

        model_infos = self._get_model_infos(keys)
        abnormal_samples = {}
        if not len(model_infos):
            return abnormal_samples

        detected_keys = [i.full_name for i in model_infos[0].img_infos]
        pro_bar = tqdm(detected_keys, ncols=100, desc='获取异常样本')
        for img_idx, name in enumerate(pro_bar):
            abnormal_samples[name] = {'filename': name, 'regions': []}
            abnormal_regions = []
            for model_info in model_infos:
                img_info = model_info.img_infos[img_idx]
                over_regions = img_info.get_over_regions(model_info.min_iou)
                for over_region in over_regions:
                    if over_region.score >= abnormal_score and over_region.area >= abnormal_area:
                        is_abnormal = True
                        for region in abnormal_regions:
                            if region.cal_iou2(over_region) > 0.3:
                                is_abnormal = False
                                break
                        if is_abnormal:
                            abnormal_regions.append(over_region)
                            abnormal_samples[name]['regions'].append(over_region.json_info)

            if not len(abnormal_samples[name]['regions']):
                abnormal_samples.pop(name)

            pro_bar.set_postfix(sample=len(abnormal_samples),
                                defect=sum([len(v['regions']) for k, v in abnormal_samples.items()]))
        pro_bar.close()

        save_path = create_dir(os.path.join(self.work_space, 'abnormality', get_cur_time()))
        save_json(abnormal_samples, save_path, 'data')
        for k, v in abnormal_samples.items():
            img_file = os.path.join(self.image_folder, k)
            if os.path.exists(img_file):
                shutil.copy(img_file, os.path.join(save_path, k))

        return abnormal_samples

    def split_result(self, key: str, update_label: bool = False) -> None:
        """
            将推理结果分割成过检及检出。检出区域的标签更新为对应的标注标注

        :param key: 模型关键词
        :param update_label: 是否将推理的检出区域标签更新为标注标签
        :return: None
        """
        model_info = self.model_infos[key]
        chk_data, over_data, inf_data = {}, {}, {}
        for img_info in model_info.img_infos:
            if update_label:
                chk_data[img_info.full_name] = dict(filename=img_info.full_name, regions=[], type='inf')
                for chk_region in img_info.get_check_regions(model_info.min_iou, 'inf'):
                    json_info = deepcopy(chk_region.json_info)
                    json_info['region_attributes']['regions'] = chk_region.IR.label
                    chk_data[img_info.full_name]['regions'].append(json_info)
            else:
                chk_data[img_info.full_name] = img_info.get_json('inf_check', model_info.min_iou)
            over_data[img_info.full_name] = img_info.get_json('over', model_info.min_iou)
            inf_data[img_info.full_name] = img_info.get_json('inf_all', model_info.min_iou)
        save_folder = os.path.join(self.work_space, 'split')
        save_json(chk_data, save_folder, f'{key}_chk.json', True)
        save_json(over_data, save_folder, f'{key}_over.json', True)
        save_json(inf_data, save_folder, f'{key}_inf.json', True)

    def __str__(self):
        return f'M{len(self.model_keys)}'

    __repr__ = __str__

    def __len__(self):
        return len(self.model_infos)


if __name__ == "__main__":
    # # 裁切软件的过滤信息
    # clipping(r'D:\Projects\299Black\box\stdimage',
    #          r"D:\Projects\299Black\box\Shield_roi.json",
    #          dst_size=(2048, 2048),
    #          dst_channel=1, special_clip=None)
    # # 查看裁剪的结果
    # common_view(r'D:\Projects\299Black\box\stdimage\clips',
    #             cfg=r"D:\Projects\299Black\box\stdimage\clips\data.json",
    #             key_parse=lambda k: k + '.bmp')

    # 测试数据的标注信息
    mark_json = r"D:\ii\tt\src\mark.json"
    # 测试数据图像文件夹
    images_folder = r"D:\ii\tt\src"
    # 模型推理结果文件夹
    inf_folder = r'D:\ii\tt\inf'
    # 方式一， 批量添加分析数据，分析结果保存在model_folder同级目录‘analyzer’中
    MA = ModelAnalyzer.load(images_folder, mark_json, model_folder=inf_folder,
                            gen_box_func=None,
                            cfg=r"",
                            key_parse=lambda k: '_'.join(k.split('_')[1:]) + '.bmp')
    # # MA = ModelAnalyzer.load(images_folder, mark_json, model_folder=inf_folder)
    # MA.filter_inf_regions(min_score=0.4, min_area=100)
    # 模型评估并推荐
    good_models = MA.recommend_model(top_k=10, min_iou=0.00001, min_chk_rate=0, eva_method=0, over_factor=0)
    print(good_models)
    # MA.split_result(good_models[0], True)
    # 保存模型的性能指标
    MA.to_excel()
    # MA.save_defect(MA.model_keys)
    # MA.get_custom_defect(good_models[0])
    # MA.save_over_defect(MA.model_keys)
    # 输出模型的评估报告
    MA.gen_report(good_models[:5], reporter='Tom')
    # 获取模型的漏检图
    # MA.get_defect(good_models[:3], 'cmp')
    # a = MA.classify_defect(good_models)
    # MA.get_defect(['model_0044999'], 'mark')
    # MA.get_defect(good_models[0], 'miss')
    # MA.get_defect(good_models[0], 'check')
    # MA.get_defect(MA.model_keys[0], 'over')
    # MA.get_image(good_models[:5], 'abs-over', False, '.jpg')
    # MA.get_image(good_models[:5], 'miss', False, '.jpg')
    MA.get_image(good_models[:5], 'check', False, '.jpg')
    # MA.get_defect(good_models, 'over')
    # MA.get_feature_defect(MA.model_keys, 'over-score', 0, 0.4)
    # MA.get_feature_defect(good_models, 'over-score', 0.8, 1)
    # MA.get_feature_defect(good_models, 'iou', 0, 0.4)

    # amb_samples, msk_samples = MA.get_ambiguous_sample(good_models, model_prob=(0.5, 0.9))
    # MA.create_new_sample(amb_samples, msk_samples, 1)
    # abn_samples = MA.get_abnormal_samples(good_models)

    # 分别保存指定标签的检出和漏检信息
    # labels = ['10', '11', '12']
    # chk_data, miss_data = {}, {}
    # for img_data in MA[good_models[0]].img_infos:
    #     c_data = {'filename': img_data.full_name, "regions": []}
    #     m_data = {'filename': img_data.full_name, "regions": []}
    #     for region in img_data.get_check_regions(0.01):
    #         if region.label in labels:
    #             c_data['regions'].append(region.json_info)
    #     for region in img_data.get_miss_regions(0.01):
    #         if region.label in labels:
    #             m_data['regions'].append(region.json_info)
    #     if len(c_data['regions']):
    #         chk_data[img_data.full_name] = c_data
    #     if len(m_data['regions']):
    #         miss_data[img_data.full_name] = m_data
    # save_json(chk_data, images_folder, 'data_chk')
    # save_json(miss_data, images_folder, 'data_miss')

    print('done')
