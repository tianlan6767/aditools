from reportlab.pdfbase import pdfmetrics   # 注册字体
from reportlab.pdfbase.ttfonts import TTFont # 字体类
from reportlab.platypus import Table, SimpleDocTemplate, Paragraph, Image  # 报告内容相关类
from reportlab.lib.pagesizes import letter  # 页面的标志尺寸(8.5*inch, 11*inch)
from reportlab.lib.styles import getSampleStyleSheet  # 文本样式
from reportlab.lib import colors  # 颜色模块
from reportlab.graphics.charts.barcharts import VerticalBarChart  # 图表类
from reportlab.graphics.charts.legends import Legend  # 图例类
from reportlab.graphics.shapes import Drawing  # 绘图工具
from reportlab.lib.units import cm  # 单位：cm
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, Counter
from dpp.common.util import cal_area
from dpp.common.dpp_json import DppJson
import re
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 注册字体(提前准备好字体文件, 如果同一个文件需要多种字体可以注册多个)
pdfmetrics.registerFont(TTFont('SimSun', "./dpp/dataset/visualization/simsunzt/simsun.ttf"))


class Convention_Analysis:
    def __init__(self, json_data, dst='.//', area_range=[100, 2000, 5000]):

        self._dj = DppJson(json_data)
        self.dst = dst

        ncols = 1
        nrows = 3
        figsize = (5, 15)
        self._area_range = list(sorted(area_range))
        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        plt.tight_layout(h_pad=nrows, w_pad=nrows)
        for i in range(nrows * ncols):
            setattr(self, '_ax' + str(i), axes.flatten()[i])

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
        explode = [0.1, 0.1]
        sizes = [self._dj.ng_pcs, self._dj.ok_pcs]
        labels = ['NG_number', 'OK_number']
        self._ax0.pie(sizes, labels=labels, explode=explode, labeldistance=0.8, autopct='%1.1f')
        self._ax0.legend([self._dj.ng_pcs, self._dj.ok_pcs],
                         loc="upper right", fontsize=10, bbox_to_anchor=(1.1, 1.05), borderaxespad=0.3)
        text_1 = 'Image_nums: ' + str(self._dj.ng_pcs + self._dj.ok_pcs)
        text_2 = 'Defect_nums: ' + str(len(self._dj.new_json["all_points_x"]))
        self._ax0.text(0.8, 0.95, text_1, color='red', fontsize=12, fontweight='bold')
        self._ax0.text(0.74, 0.8, text_2, color='red', fontsize=12, fontweight='bold')
        self._ax0.set_title('NG/OK数量分布', fontsize=30)

    def bar_label(self):
        label_keys = [int(item) for item in list(self._dj.labels_dict.keys())]
        label_values = list(self._dj.labels_dict.values())
        label_list = self._dj.new_json["regions"]

        sorted_label_values = [item[1] for item in sorted(dict(zip(label_keys, label_values)).items(), key=lambda x:x[0])]
        sorted_label_keys = sorted(label_keys)

        label_pcs = []
        for label in sorted_label_keys:
            begin = 0
            per_label_list = []
            for i in range(self._dj.labels_dict[str(label)]):
                per_label_list.append(label_list.index(str(label), begin, len(label_list)))
                begin = label_list.index(str(label), begin, len(label_list)) + 1
            label_pcs.append(len(set([self._dj.new_json["filename"][index] for index in per_label_list])))

        self._ax1.bar(np.arange(len(sorted_label_keys))-0.25+np.arange(len(sorted_label_keys))*0.5,
                      sorted_label_values, tick_label=sorted_label_keys, width=0.5, align='center')
        self._ax1.bar(np.arange(len(sorted_label_keys))+0.25+np.arange(len(sorted_label_keys))*0.5,
                      label_pcs, tick_label=sorted_label_keys, width=0.5, align='center')

        for i, v in enumerate(sorted_label_values):
            self._ax1.text(i-0.25+i*0.5, v, str(v), color='blue', fontsize=10, fontweight='bold')
        for i, v in enumerate(label_pcs):
            self._ax1.text(i+0.25+i*0.5, v, str(v), color='red', fontsize=10, fontweight='bold')
        self._ax1.legend(("缺陷数", "图片数"), loc='best', fontsize="20", borderpad=0.1, labelspacing=0.2,
                         columnspacing=0.2, framealpha=0.5)
        self._ax1.set_title('缺陷类别数量分布', fontsize=30)

    def pie_area(self):
        area_list, area_queue = self._area
        labels = list(range(len(area_list)))
        step = len(area_list) // 10
        x_labels = list(range(1, len(area_list), step))
        y_labels = area_queue
        self._ax2.plot(labels, area_list, linewidth=1, color="orange", marker="o", label="Area Distribution Curve")
        # midel = len(area_list) // 2
        # self._ax2.plot(midel, area_list[midel], linewidth=1, color="red", marker="+", linestyle='--', label="50%")
        self._ax2.set_xlabel("Defects Number")
        self._ax2.set_ylabel("Area")
        self._ax2.set_xticks(x_labels)
        self._ax2.set_yticks(y_labels)
        self._ax2.set_title("缺陷面积分布图")

    def __call__(self):
        for func in dir(self):
            if not func.startswith("_"):
                try:
                    getattr(self, func)()
                except:
                    pass
        f_name = self.dst + '//image_1.jpg'
        plt.tight_layout()
        plt.savefig(f_name, dpi=300)
        # plt.show()


class Work_Station_Analysis:
    def __init__(self, json_data, dst='.//'):

        self._dj = DppJson(json_data)
        self.dst = dst

        _, axes = plt.subplots(nrows=1, ncols=1)
        plt.tight_layout(h_pad=1, w_pad=1)


    def dict_updata(self, dict, key):
        value = int(dict.get(key))
        value += 1
        dict[key] = str(value)

    def work_station(self):
        label_keys = [int(item) for item in list(self._dj.labels_dict.keys())]
        label_values = list(self._dj.labels_dict.values())
        labels_list = self._dj.new_json["regions"]

        sorted_label_values = [item[1] for item in sorted(dict(zip(label_keys, label_values)).items(), key=lambda x:x[0])]
        sorted_label_keys = sorted(label_keys)

        label_pcs = []
        for label in sorted_label_keys:
            # label = 5
            begin = 0

            per_label_list = []
            for i in range(self._dj.labels_dict[str(label)]):
                per_label_list.append(labels_list.index(str(label), begin, len(labels_list)))
                begin = labels_list.index(str(label), begin, len(labels_list)) + 1

            per_label_name_list = []
            for index in per_label_list:
                fn = self._dj.new_json["filename"][index]
                if '-' in fn:
                    patten_1 = re.compile(r'-[0-9]_[0-9]')
                    fn_1 = patten_1.findall(fn)
                    fn_2 = fn_1[0].split('-')[1]
                    fn_new = fn_2.replace('_', '/')
                    per_label_name_list.append(fn_new)
                else:
                    patten_1 = re.compile(r'[0-9]_[0-9]_[0-9]*[.a-zA-Z]')
                    fn_1 = patten_1.findall(fn)
                    fn_2 = fn_1[0].split('_')
                    fn_new = fn_2[0] + '/' + fn_2[1]
                    per_label_name_list.append(fn_new)

            label_pcs.append(per_label_name_list)

        static_worker = {}
        for i in range(len(label_pcs)):
            per_class_name_worker = label_pcs[i]
            for j in range(len(per_class_name_worker)):
                woker_name = per_class_name_worker[j]
                if woker_name in static_worker:
                    self.dict_updata(dict=static_worker, key=woker_name)
                else:
                    static_worker[woker_name] = '1'

        return static_worker

    def __call__(self):
        static_worker = self.work_station()
        label_keys = [item for item in static_worker.keys()]
        label_values = [int(item) for item in list(static_worker.values())]
        stored_label_keys = sorted(label_keys)
        plt.bar(range(len(label_values)), label_values, width=0.5)
        plt.xticks(range(len(stored_label_keys)), stored_label_keys)
        for i in range(len(label_values)):
            plt.text(x=i - 0.05, y=label_values[i] + 0.2, s='%d' % label_values[i])
        plt.xlabel("worker number")
        plt.ylabel("defects number")
        plt.title("工位所对应的缺陷数量分布")
        plt.tight_layout()
        f_name = self.dst + '//image_3.jpg'
        plt.savefig(f_name, dpi=300)
        # plt.show()


class Graphs:
    # 绘制标题
    @staticmethod
    def draw_title(title: str):
        # 获取所有样式表
        style = getSampleStyleSheet()
        # 拿到标题样式
        ct = style['Heading1']
        # 单独设置样式相关属性
        ct.fontName = 'SimSun'  # 字体名
        ct.fontSize = 18  # 字体大小
        ct.leading = 50  # 行间距
        ct.textColor = colors.green  # 字体颜色
        ct.alignment = 1  # 居中
        ct.bold = True
        # 创建标题对应的段落，并且返回
        return Paragraph(title, ct)

    # 绘制小标题
    @staticmethod
    def draw_little_title(title: str):
        # 获取所有样式表
        style = getSampleStyleSheet()
        # 拿到标题样式
        ct = style['Normal']
        # 单独设置样式相关属性
        ct.fontName = 'SimSun'  # 字体名
        ct.fontSize = 15  # 字体大小
        ct.leading = 30  # 行间距
        ct.textColor = colors.red  # 字体颜色
        # 创建标题对应的段落，并且返回
        return Paragraph(title, ct)

    # 绘制普通段落内容
    @staticmethod
    def draw_text(text: str):
        # 获取所有样式表
        style = getSampleStyleSheet()
        # 获取普通样式
        ct = style['Normal']
        ct.fontName = 'SimSun'
        ct.fontSize = 12
        ct.wordWrap = 'CJK'  # 设置自动换行
        ct.alignment = 0  # 左对齐
        ct.firstLineIndent = 32  # 第一行开头空格
        ct.leading = 25
        return Paragraph(text, ct)

    # 绘制表格
    @staticmethod
    def draw_table(*args):
        # 列宽度
        col_width = 120
        style = [
            ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),  # 字体
            ('FONTSIZE', (0, 0), (-1, 0), 12),  # 第一行的字体大小
            ('FONTSIZE', (0, 1), (-1, -1), 10),  # 第二行到最后一行的字体大小
            ('BACKGROUND', (0, 0), (-1, 0), '#d5dae6'),  # 设置第一行背景颜色
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # 第一行水平居中
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),  # 第二行到最后一行左右左对齐
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # 所有表格上下居中对齐
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.darkslategray),  # 设置表格内文字颜色
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # 设置表格框线为grey色，线宽为0.5
            # ('SPAN', (0, 1), (0, 2)),  # 合并第一列二三行
            # ('SPAN', (0, 3), (0, 4)),  # 合并第一列三四行
            # ('SPAN', (0, 5), (0, 6)),  # 合并第一列五六行
            # ('SPAN', (0, 7), (0, 8)),  # 合并第一列五六行
        ]
        table = Table(args, colWidths=col_width, style=style)
        return table

    # 创建图表
    @staticmethod
    def draw_bar(bar_data: list, ax: list, items: list):
        drawing = Drawing(500, 250)
        bc = VerticalBarChart()
        bc.x = 45  # 整个图表的x坐标
        bc.y = 45  # 整个图表的y坐标
        bc.height = 200  # 图表的高度
        bc.width = 350  # 图表的宽度
        bc.data = bar_data
        bc.strokeColor = colors.black  # 顶部和右边轴线的颜色
        bc.valueAxis.valueMin = 5000  # 设置y坐标的最小值
        bc.valueAxis.valueMax = 26000  # 设置y坐标的最大值
        bc.valueAxis.valueStep = 2000  # 设置y坐标的步长
        bc.categoryAxis.labels.dx = 2
        bc.categoryAxis.labels.dy = -8
        bc.categoryAxis.labels.angle = 20
        bc.categoryAxis.categoryNames = ax

        # 图示
        leg = Legend()
        leg.fontName = 'SimSun'
        leg.alignment = 'right'
        leg.boxAnchor = 'ne'
        leg.x = 475  # 图例的x坐标
        leg.y = 240
        leg.dxTextSpace = 10
        leg.columnMaximum = 3
        leg.colorNamePairs = items
        drawing.add(leg)
        drawing.add(bc)
        return drawing

    # 绘制图片
    @staticmethod
    def draw_img_large(path):
        img = Image(path)  # 读取指定路径下的图片
        img.drawWidth = 18 * cm  # 设置图片的宽度
        img.drawHeight = 15 * cm  # 设置图片的高度
        return img

    # 绘制图片
    @staticmethod
    def draw_img_small(path):
        img = Image(path)  # 读取指定路径下的图片
        img.drawWidth = 10 * cm  # 设置图片的宽度
        img.drawHeight = 20 * cm  # 设置图片的高度
        return img


def bulid_analysis_report(dst):
    content = list()

    # 添加标题
    content.append(Graphs.draw_title('Data Analysis Report'))

    # 添加图片
    image_1_path = dst + '//image_1.jpg'
    image_3_path = dst + '//image_3.jpg'
    content.append(Graphs.draw_img_small(image_1_path))
    content.append(Graphs.draw_img_large(image_3_path))

    # # 添加段落文字
    # content.append(Graphs.draw_text('众所周知，大数据分析师岗位是香饽饽，近几年数据分析热席卷了整个互联网行业，与数据分析的相关的岗位招聘、培训数不胜数。很多人前赴后继，想要参与到这波红利当中。那么数据分析师就业前景到底怎么样呢？'))
    # # 添加小标题
    # content.append(Graphs.draw_title(''))
    # content.append(Graphs.draw_little_title('不同级别的平均薪资'))
    # # 添加表格
    # data = [
    #     ('职位名称', '平均薪资', '较上年增长率'),
    #     ('数据分析师', '18.5K', '25%'),
    #     ('高级数据分析师', '25.5K', '14%'),
    #     ('资深数据分析师', '29.3K', '10%')
    # ]
    # content.append(Graphs.draw_table(*data))
    #
    # # 生成图表
    # content.append(Graphs.draw_title(''))
    # content.append(Graphs.draw_little_title('热门城市的就业情况'))
    # b_data = [(25400, 12900, 20100, 20300, 20300, 17400), (15800, 9700, 12982, 9283, 13900, 7623)]
    # ax_data = ['BeiJing', 'ChengDu', 'ShenZhen', 'ShangHai', 'HangZhou', 'NanJing']
    # leg_items = [(colors.red, '平均薪资'), (colors.green, '招聘量')]
    # content.append(Graphs.draw_bar(b_data, ax_data, leg_items))

    # 生成pdf文件
    doc_path = dst + '//report.pdf'
    doc = SimpleDocTemplate(doc_path, pagesize=letter)
    doc.build(content)

























