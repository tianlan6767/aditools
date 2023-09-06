import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager
from typing import Union, Sequence, List
from .comm import get_ellipse_box, create_dir


__all__ = ['show_feature_hist', 'show_defect_detail', 'show_feature_bar', 'show_feature_pie', 'draw_via_region',
           'show_defect_plot', 'get_via_region_boundary']

# 解决图像显示中文问题
mpl_fonts = set(f.name for f in FontManager().ttflist)
__chinese_font = 'SimHei'
if 'SimHei' in mpl_fonts:
    __chinese_font = 'SimHei'
elif 'AR PL UKai CN' in mpl_fonts:
    __chinese_font = 'AR PL UKai CN'
else:
    __chinese_font = 'AR PL UMing CN'


def draw_via_region(image, region, color=(255, 0, 0), thickness=1, draw_label=False, draw_score=False, inplace=True):
    """
        绘制via区域信息，直接更新输入图像

    :param image: 图像，进过绘制或发生变化
    :param region: via的区域字段信息
    :param color: 区域颜色, (b,g,r)
    :param thickness: 区域边界宽度
    :param draw_label: 是否同时绘制缺陷标签信息
    :param draw_score: 是否同时绘制缺陷得分信息
    :param inplace: 是否执行原地操作
    :return: 绘制后的图像
    """

    region_shape = region['shape_attributes']['name'] if 'name' in region['shape_attributes'] else 'polygon'
    try:
        region_label = int(region['region_attributes']['regions'])
    except KeyError:
        region_label = 1

    drawn_image = image if inplace else image.copy()

    def draw_text(text, scale=0.5, x_offset=0, y_offset=0):
        font_face, font_scale = cv2.FONT_HERSHEY_SIMPLEX, scale
        # 计算文本的宽高，baseLine
        text_size, base_line = cv2.getTextSize(text, font_face, font_scale, thickness)
        box, anchor = get_via_region_boundary(region, 5, drawn_image.shape[:2])
        if anchor[1] < text_size[1] + base_line:
            anchor[1] = text_size[1] + base_line
            anchor[0] += base_line
        if anchor[0] < text_size[0] + base_line:
            anchor[0] = text_size[0] + base_line
            anchor[1] += base_line
        cv2.putText(drawn_image, text, (anchor[0] + x_offset, anchor[1] + y_offset),
                    font_face, font_scale, color, thickness)

    # 绘制标签
    if region_label and draw_label:
        draw_text(str(region_label), 0.5, x_offset=-10)

    # 绘制得分
    if draw_score:
        region_score = float(region['region_attributes']['score'])
        draw_text(f'{region_score:.2f}', 0.3)

    # 绘制区域
    if region_shape == "circle":
        cx = region['shape_attributes']['cx']
        cy = region['shape_attributes']['cy']
        r = region['shape_attributes']['r']
        cv2.circle(drawn_image, (cx, cy), int(r), color=color, thickness=thickness)
    elif region_shape == "rect":
        left = region['shape_attributes']['x']
        top = region['shape_attributes']['y']
        right = region['shape_attributes']['x'] + region['shape_attributes']['width']
        bottom = region['shape_attributes']['y'] + region['shape_attributes']['height']
        cv2.rectangle(drawn_image, (left, top), (right, bottom), color=color, thickness=thickness)
    elif region_shape == "polygon":
        ptx = region['shape_attributes']['all_points_x']
        pty = region['shape_attributes']['all_points_y']
        points = np.dstack((ptx, pty))
        cv2.polylines(drawn_image, pts=[points], isClosed=True, color=color, thickness=thickness)
    elif region_shape == "ellipse":
        cx = int(region['shape_attributes']['cx'])
        cy = int(region['shape_attributes']['cy'])
        l_x = int(region['shape_attributes']['rx'])
        s_y = int(region['shape_attributes']['ry'])
        theta = int(region['shape_attributes']['theta'])
        cv2.ellipse(drawn_image, (cx, cy), (l_x, s_y), theta, theta, 360, color=color, thickness=thickness)
    else:
        raise ValueError(f'不支持解析[{region_shape}]区域!!!')

    return drawn_image


def get_via_region_boundary(region, extension=0, limit_size=None):
    """
        获取区域的外接矩形

    :param region: 区域信息
    :param extension: 外扩像素数目
    :param limit_size: 限制区域，一般为图像大小(rows, cols)
    :return: 区域外接矩形信息 (left, top, right, bottom)，区域边界离外接矩形最近的点 anchor
    """

    region_shape = region['shape_attributes']['name'] if 'name' in region['shape_attributes'] else 'polygon'
    anchor = [0, 0]

    if region_shape == "circle":
        cx = region['shape_attributes']['cx']
        cy = region['shape_attributes']['cy']
        r = region['shape_attributes']['r']
        box = (cx - r, cy - r, cx + r, cy + r)
    elif region_shape == "rect":
        left = region['shape_attributes']['x']
        top = region['shape_attributes']['y']
        right = region['shape_attributes']['x'] + region['shape_attributes']['width']
        bottom = region['shape_attributes']['y'] + region['shape_attributes']['height']
        box = (left, top, right, bottom)
    elif region_shape == "polygon":
        ptx = region['shape_attributes']['all_points_x']
        pty = region['shape_attributes']['all_points_y']
        min_y = min(pty)
        box = (min(ptx), min_y, max(ptx), max(pty))
        anchor = [ptx[pty.index(min_y)], min_y]
        anchor[0] = 0 if anchor[0] - extension < 0 else anchor[0] - extension
        anchor[1] = 0 if anchor[1] - extension < 0 else anchor[1] - extension
    elif region_shape == "ellipse":
        cx = int(region['shape_attributes']['cx'])
        cy = int(region['shape_attributes']['cy'])
        major_r = int(region['shape_attributes']['rx'])
        minor_r = int(region['shape_attributes']['ry'])
        theta = int(region['shape_attributes']['theta'])
        box = get_ellipse_box(major_r, minor_r, theta, cx, cy)
    else:
        raise ValueError(f'不支持解析[{region_shape}]区域!!!')

    box = [box[0] - extension, box[1] - extension, box[2] + extension, box[3] + extension]
    box[0] = 0 if box[0] < 0 else box[0]
    box[1] = 0 if box[1] < 0 else box[1]

    if limit_size:
        box[2] = limit_size[1] if box[2] > limit_size[1] else box[2]
        box[3] = limit_size[0] if box[3] > limit_size[0] else box[3]

    if region_shape != "polygon":
        anchor = [box[0], box[1]]

    return tuple(box), anchor


def show_feature_hist(feature, bins: Union[int, Sequence[float]] = 10, subtitle='', save_path=None, show_hist=True):
    """
        显示特征分布图

    :param feature: 特征值
    :param bins: Number of histogram bins to be used. default: 10

            If *bins* is an integer, it defines the number of equal-width bins in the range.

            If *bins* is a sequence, it defines the bin edges, including the left edge of the first bin
            and the right edge of the last bin; in this case, bins may be unequally spaced.
            All but the last (righthand-most) bin is half-open. In other words, if *bins* is::

                [1, 2, 3, 4]

            then the first bin is ``[1, 2)`` (including 1, but excluding 2) and
            the second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which
            *includes* 4.

    :param subtitle: 子标题
    :param save_path: 图像保存路径
    :param show_hist: 是否显示图像
    :return: None
    """

    if not isinstance(feature, (pd.Series,)):
        raise ValueError("feature must be pandas.Series")

    feature.plot.hist(grid=False, bins=bins, rwidth=0.5)
    # feature.plot.kde(grid=True, legend=False)
    plt.title(f'{ str(feature.name).capitalize()} Histogram{" - " + subtitle if subtitle else ""}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    if save_path:
        if '.' in os.path.split(save_path)[-1]:
            plt.savefig(save_path)
        else:
            create_dir(save_path, )
            plt.savefig(os.path.join(save_path, f'{feature.name}.jpg'))
    if show_hist:
        plt.show()
    else:
        plt.clf()
        plt.close()


def show_feature_bar(data: dict, categories: List[str], title: str, save_path=None, show_hist=True):
    """
        显示特征信息

    :param data: 每个标签所映射的数据，数据的长度和categories的长度一致
    :param categories: 标签列表
    :param title: 标题
    :param save_path: 图像保存路径
    :param show_hist: 是否显示图像
    :return: None
    """

    labels = list(data.keys())
    data = np.array(list(data.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data.shape[1]))[-1::-1]

    fig, ax = plt.subplots(figsize=(7.6, 5))
    ax.set_title(title, fontproperties=__chinese_font, fontsize=12, loc='right')
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max(initial=0))

    for i, (col_name, color) in enumerate(zip(categories, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=col_name, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        try:
            ax.bar_label(rects, label_type='center', color=text_color)
        except AttributeError:
            _label_bars(ax, rects, label_type='center', bar_ori='horizontal', color=text_color)
            pass
    ax.legend(ncol=len(categories), bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')

    # 解决y轴显示不全的问题
    fig.tight_layout()

    if save_path:
        if '.' in os.path.split(save_path)[-1]:
            plt.savefig(save_path)
        else:
            create_dir(save_path, )
            plt.savefig(os.path.join(save_path, f'{len(data)}_bar.jpg'))
    if show_hist:
        plt.show()
    else:
        plt.clf()
        plt.close()


def show_defect_detail(right_detail: dict, error_detail: dict, title: str, save_path=None, show_hist=True):
    """
        显示缺陷的检出信息

    :param right_detail:
    :param error_detail:
    :param title:
    :param save_path:
    :param show_hist:
    :return:
    """

    fig, ax = plt.subplots()
    ind = np.arange(len(right_detail))
    title = 'Defect Detail' if not title else title

    p1 = ax.bar(ind, tuple(right_detail.values()), width=0.8, color='#15B01A', yerr=None, label='Check')
    p2 = ax.bar(ind, tuple(error_detail.values()), width=0.8, color='#FF6347',
                bottom=tuple(right_detail.values()), yerr=None, label='Miss')

    ax.axhline(0, color='grey', linewidth=1.5)
    ax.set_ylabel('Num')
    ax.set_title(title, fontproperties=__chinese_font, fontsize=12)
    ax.set_xticks(ind)
    ax.set_xticklabels(tuple(right_detail.keys()))
    ax.legend()

    # Label with label_type 'center' instead of the default 'edge'
    try:
        ax.bar_label(p1, label_type='center', fontsize=6)
        ax.bar_label(p2, label_type='center', fontsize=6)
        ax.bar_label(p2, fontsize=9)
    except AttributeError:
        # 升级matplotlib到3.4.1以上版本即可
        _label_bars(ax, p1, label_type='center', fontsize=6)
        _label_bars(ax, p2, label_type='center', fontsize=6)
        _label_bars(ax, p2, label_type='edge', fontsize=9)

    if save_path:
        if '.' in os.path.split(save_path)[-1]:
            plt.savefig(save_path)
        else:
            create_dir(save_path)
            plt.savefig(os.path.join(save_path, f'{title}.jpg'))
    if show_hist:
        plt.show()
    else:
        plt.clf()
        plt.close()


def show_defect_plot(detail: Union[pd.DataFrame, ], title: str, y_label: str,
                     save_path=None, show_hist=True):
    """
        显示缺陷的检出信息

    :param detail:
    :param title:
    :param y_label:
    :param save_path:
    :param show_hist:
    :return:
    """

    if not isinstance(detail, (pd.DataFrame,)):
        raise ValueError("feature must be pandas.DataFrame")

    detail.plot(kind='line', figsize=(7.6, 5))
    plt.title(title, fontproperties=__chinese_font, fontsize=10)
    plt.grid(alpha=0.3, linestyle='-.')
    # 设置图例样式
    # loc代表图例所在的位置，upper right代表右上角
    plt.legend(loc='upper right', prop=__chinese_font)
    # 设置字体，x轴刻度，x轴刻度内容，旋转标签45°，设置中文字体显示
    lbs = detail.index.to_list()
    plt.xticks(np.arange(0, len(lbs), step=1), labels=lbs, rotation=45, fontproperties=__chinese_font)
    plt.ylabel(y_label, fontproperties=__chinese_font, fontsize=8)
    plt.xlabel('标签类别', fontproperties=__chinese_font, fontsize=8)

    if save_path:
        if '.' in os.path.split(save_path)[-1]:
            plt.savefig(save_path)
        else:
            create_dir(save_path)
            plt.savefig(os.path.join(save_path, f'{title}.jpg'))
    if show_hist:
        plt.show()
    else:
        plt.clf()
        plt.close()


def show_feature_pie(data, title: str, save_path=None, show_hist=True):
    """
        以饼图显示分布信息

    :param data: 数据
    :param title: 标题
    :param save_path: 图像保存路径
    :param show_hist: 是否显示图像
    :return: None
    """

    assert isinstance(data, pd.Series), "data must be pandas.Series"

    data.name = ''
    plt.axes(aspect='equal')
    plt.title(title, fontproperties=__chinese_font)
    data.plot(kind='pie',
              autopct='%.1f%%',
              radius=1,
              startangle=180,
              counterclock=False,
              shadow=False,
              # explode=(0, 0, 0, 0, 0),
              wedgeprops={'linewidth': 1, 'edgecolor': 'black', "width": 1},
              textprops={'fontsize': 10, 'color': 'black', 'fontproperties': __chinese_font},
              )

    if save_path:
        if '.' in os.path.split(save_path)[-1]:
            plt.savefig(save_path)
        else:
            create_dir(save_path, )
            plt.savefig(os.path.join(save_path, f'{len(data)}_pie.jpg'))
    if show_hist:
        plt.show()
    else:
        plt.clf()
        plt.close()


def _label_bars(ax, rects, label_type, bar_ori='vertical', **kwargs):
    """
        以饼图显示分布信息

    :param ax: axis对象
    :param rects: bar区域
    :param label_type: 字符放置的位置
    :param bar_ori: 条形图的方向 ['horizontal', 'vertical']
    :param kwargs: Additional kwargs are passed to Text
    :return: None
    """

    for rect in rects:
        width = rect.get_width()
        height = rect.get_height()

        if label_type == 'edge':
            if bar_ori == 'horizontal':
                text = width + rect.get_x()
                xy = (rect.get_x() + width, rect.get_y() + height / 2)
            else:
                text = height + rect.get_y()
                xy = (rect.get_x() + width / 2, rect.get_y() + height)
            xy_text = (0, 1)
        else:
            text = width if bar_ori == 'horizontal' else height
            xy = (rect.get_x() + width / 2, rect.get_y() + height / 2)
            xy_text = (0, -4)

        ax.annotate('{}'.format(text),
                    xy=xy,
                    xytext=xy_text,
                    textcoords="offset points",
                    ha='center', va='bottom', **kwargs)
