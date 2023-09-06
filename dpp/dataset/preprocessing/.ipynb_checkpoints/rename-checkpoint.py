import os
import shutil
from tqdm import tqdm
from dpp.common.util import load_file, read_json
from dpp.common.dpp_json import DppJson
from dpp.dataset.preprocessing.json_convert import via_to_json
from dpp.dataset.preprocessing.json_operate import merge_json
from dpp.common.mylog import Logger


def rename_img(imgs, dst, start):
    for img in tqdm(imgs):
        fn = os.path.basename(img)
        newname = '{}{}.bmp'.format(start, fn.replace(".bmp", ""))
        if dst == os.path.dirname(img):
            os.rename(img, os.path.join(dst, newname))
        else:
            shutil.copy(img, os.path.join(dst, newname))


def reanme_img_json(imgs, json_data, dst, start):
    new_json = {}
    for img in tqdm(imgs):
        filename = os.path.basename(img)
        newname = '{}{}.bmp'.format(start, filename.replace(".bmp", ""))
        try:
            regions = json_data[filename]['regions']
        except:
            raise ValueError("json键值与图片名不匹配: {}".format(os.path.dirname(img)))
        new_json[newname] = {}
        new_json[newname]['filename'] = newname
        new_json[newname]['regions'] = regions
        shutil.copy(img, os.path.join(dst, newname))
    return new_json


def rename_one_folder(imgs, json_data, dst, start):
    """
    Args:
      rename: 是否重命名
      start: 重命名前缀
      end: 重命名后缀
      {HG299B_0812_LP}_{1-1-1_1_1}_{OK}.bmp
      产品型号 日期 类型 批次 图片名 NG/OK
    """
    dj = DppJson(json_data)
    if 'ok' in dj.keys:
        pass
        # mylog.warning("空标注=>{}".format(dj.new_json["ok"]))
    if dj.json_format.startswith('VIA'):
        json_data = via_to_json(json_data)
    rename_json = reanme_img_json(imgs, json_data, dst, start)
    return rename_json


def rename_classify_folder(src, dst, start):
    """
    按缺陷类别采集的文件夹，手动修改下需要文件夹命名;
    请确保一个文件夹以多张图片及一个json组成
    """
    json_list = []
    for name in os.listdir(src):  # name:缺陷类型缩写KL
        path = os.path.join(src, name)
        imgs = load_file(path)
        jf = load_file(path, format="json")
        assert len(jf) == 1, "一个文件夹只有有一个标注<{}>".format(path)
        json_data = read_json(jf[0])
        dj = DppJson(json_data)
        if 'ok' in dj.keys:
            Logger.warning("空标注=>{}".format(dj.new_json["ok"]))
        if dj.json_format.startswith('VIA'):
            json_data = via_to_json(json_data)
        _start = "{}{}_".format(start, name)
        rename_json = reanme_img_json(
            imgs, json_data, dst, _start)
        json_list.append(rename_json)
    new_json = merge_json(json_list)
    return new_json


def rename_product_folder(src, dst, start):
    json_list = []
    for name in os.listdir(src):    # name:产品编号
        path = os.path.join(src, name)
        imgs = load_file(path)
        jf = load_file(path, format="json")
        assert len(jf) == 1, "一个文件夹只有有一个标注"
        json_data = read_json(jf[0])
        dj = DppJson(json_data)
        if 'ok' in dj.keys:
            Logger.warning("空标注=>{}".format(dj.new_json["ok"]))
        if dj.json_format.startswith('VIA'):
            json_data = via_to_json(json_data)
        _start = "{}{}-".format(start, name)
        rename_json = reanme_img_json(
            imgs, json_data, dst, _start)
        json_list.append(rename_json)
    new_json = merge_json(json_list)
    return new_json
