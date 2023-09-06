import time
import warnings
import multiprocessing as mp
from dpp.common.util import load_file
from dpp.common.mylog import Logger
from inf.config import init_config
from inf.default import multi_weights_inf
from inf.default_async import single_weight_inf
from inf.validate import validate_weights, checkpoints_weights,cal_time,filter_weights
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    start = time.time()
    # 初始化配置cfg
    cfg = init_config()
    weights = filter_weights(validate_weights(
        cfg), step=cfg.WEIGHTS_INF_STEP, drop=cfg.WEIGHTS_INF_DROP)     # 验证筛选模型
    continue_weights = checkpoints_weights(cfg, weights)                # 筛选已经推理过的模型
    imgs = load_file(cfg.TEST_IMG_DIR, format="img")

    mp.set_start_method("spawn", force=True)
    if len(continue_weights) < len(cfg.GPU_NUMS):       # 多个GPU推理一个模型
        predict_time = cal_time(cfg.GPU_NUMS,cfg.INPUT.MIN_SIZE_TEST,imgs,continue_weights,inf_type="multi")
        Logger.info('本次推理{}个模型，{}张图片，预计{}s完成'.format(len(continue_weights), len(imgs), predict_time))
        single_weight_inf(cfg, imgs, continue_weights)
    else:                               # 每个GPU推理一个模型
        predict_time = round(len(continue_weights)*len(imgs) / 7/len(cfg.GPU_NUMS)+len(continue_weights)*3, 2)
        Logger.info('本次推理{}个模型，{}张图片，预计{}s完成'.format(len(continue_weights), len(imgs), predict_time))
        multi_weights_inf(cfg, imgs, continue_weights)
    Logger.info("推理完成。本次推理{}个模型，{}张小图，总耗时：{}s".format(len(continue_weights), len(imgs), round((time.time()-start), 2)))
