import cv2
import atexit
import bisect
import torch
import os
import multiprocessing as mp
from collections import deque
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from inf.default import DefaultPredictor
from copy import deepcopy
from tqdm import tqdm
from inf.parse import model_to_json
from dpp.common.util import save_json


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, gpus_list):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_gpus = len(gpus_list)
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(
                gpus_list[gpuid]) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(
                    cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()

        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 3


class Visualization(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=True):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.cfg = cfg
        self.parallel = parallel
        if parallel:
            self.predictor = AsyncPredictor(cfg, gpus_list=cfg.GPU_NUMS)
        else:
            self.predictor = DefaultPredictor(cfg)

    def inf(self, im, save=False):
        predictions = self.predictor(im)["instances"].to(self.cpu_device)
        if save:
            visualizer = Visualizer(cv2.cvtColor(
                im, cv2.COLOR_GRAY2RGB), self.metadata, instance_mode=self.instance_mode)
            vis_output = visualizer.draw_instance_predictions(
                predictions=predictions)
            # vis_output = cv2.cvtColor(vis_output.get_image()[:,:,::-1], cv2.COLOR_RGB2BGR)
            vis_output = vis_output.get_image()
            return predictions, vis_output
        return predictions

    def async_inf(self, imgs, save=False):
        vis_kwargs = {"metadata": {1: 0}, "instance_mode": self.instance_mode}

        def process_predictions(im, predictions, kwargs, save):
            predictions = predictions["instances"].to(self.cpu_device)
            if save:
                visualizer = Visualizer(cv2.cvtColor(
                    im, cv2.COLOR_GRAY2RGB)[:, :, ::-1], **kwargs)
                vis_output = visualizer.draw_instance_predictions(predictions)
                vis_output = vis_output.get_image()
                return predictions, vis_output
            return predictions

        buffer_size = self.predictor.default_buffer_size
        im_data = deque()
        for cnt, img in enumerate(imgs):
            if self.cfg.INPUT.FORMAT == 'L':
                im = cv2.imread(img, 0)
            else:
                im = cv2.imread(img, 1)
            im_data.append(im)
            self.predictor.put(im)

            if cnt >= buffer_size:
                im = im_data.popleft()
                preds = self.predictor.get()
                yield process_predictions(im, preds, vis_kwargs, save)

        while len(im_data):
            im = im_data.popleft()
            preds = self.predictor.get()
            yield process_predictions(im, preds, vis_kwargs, save)


def single_weight_inf(cfg, imgs, weights):
    from inf.default_async import Visualization  # 不能放在最上面
    # mp.set_start_method("spawn", force=True)
    if os.path.isfile(cfg.MODEL.WEIGHTS):
        json_dir = os.path.dirname(cfg.MODEL.WEIGHTS)+"_json"
    else:
        json_dir = cfg.WEIGHTS_JSON_DIR
    for id in range(len(weights)):
        async_cfg = deepcopy(cfg)
        async_cfg.MODEL.WEIGHTS = weights[id]
        demo = Visualization(async_cfg)
        # 推理NG图片，输出图片维度，缺陷维度检出率
        result = demo.async_inf(imgs)
        new_json = {}
        bar = tqdm(imgs)
        bar.set_description("推理模型{}".format(async_cfg.MODEL.WEIGHTS))
        for img in bar:
            predictions = next(result)
            filename = os.path.basename(img)
            regions = model_to_json(predictions)
            new_json[filename] = {}
            new_json[filename]["filename"] = filename
            new_json[filename]["regions"] = regions
            new_json[filename]["type"] = "inf"
        json_name = os.path.basename(async_cfg.MODEL.WEIGHTS).split(".")[0]
        save_json(new_json, json_dir, json_name)
