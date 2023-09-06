import os
import glob
import math
import time
import torch


def validate_weights(cfg):
    if os.path.isdir(cfg.MODEL.WEIGHTS):
        return sorted(glob.glob(os.path.join(cfg.MODEL.WEIGHTS, '*.pth')))
    elif os.path.isfile(cfg.MODEL.WEIGHTS):
        return [cfg.MODEL.WEIGHTS]
    else:
        raise ValueError("权重模型参数配置错误")


def checkpoints_weights(cfg, weights):
    weights_dir = os.path.dirname(weights[0])
    finished_weights = [os.path.join(weights_dir, os.path.basename(weight).replace(
        ".json", ".pth")) for weight in glob.glob(os.path.join(cfg.WEIGHTS_JSON_DIR, '*.json'))]
    continue_weights = list(set(weights).difference(set(finished_weights)))
    return sorted(continue_weights)


def filter_weights(weights,step = 1 ,drop = 0):
    if drop>=1 or drop<0:
        raise ValueError("权重舍弃比率只能(0,1]之间的小数")
    start = math.floor(len(weights)*drop)
    return weights[start:][::-1][::step][::-1]

def cal_time(gpu_nums,img_size,imgs,weights,inf_type="single"):
    if inf_type == "single":
      pre_time = round(len(weights)*len(imgs)*(img_size/2048)/7/len(gpu_nums)+len(weights)*len(gpu_nums)*3, 2)
    else:
      pre_time = round(len(weights)*len(imgs)*(img_size/2048)/(7*len(gpu_nums))+len(weights)*len(gpu_nums)*3, 2)
    return pre_time
    
    
def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()