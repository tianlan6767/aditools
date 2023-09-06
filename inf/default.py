import torch,os,copy,cv2
import multiprocessing as mp
from tqdm import tqdm
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from inf.parse import model_to_json
from dpp.common.util import save_json,make_dir


COLORS = ["red","yellow","green","blue"]



class DefaultPredictor:
    
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        # self.aug = T.ResizeShortestEdge(
        #     [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        # )

        self.input_format = cfg.INPUT.FORMAT
        for module in self.model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        self.model.eval()
        
    def __call__(self, original_image):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            height, width = original_image.shape[:2]
            if self.cfg.INPUT.FORMAT == 'L':
                image = torch.as_tensor(original_image.astype("float32"))
            else:
                if len(original_image.shape)==2:
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
                image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}  
            # import time
            # start = time_synchronized()
            predictions = self.model([inputs])[0]
            # print("all:",time_synchronized()-start)
            return predictions


def worker(s,q,idx,cfg,weights,imgs):
    s.acquire()
    cuda_id = q.get()
    cfg.MODEL.DEVICE = "cuda:{}".format(cuda_id)
    model_name = os.path.basename(cfg.MODEL.WEIGHTS)
    make_dir(cfg.WEIGHTS_JSON_DIR)
    new_json = {}
    predictor = DefaultPredictor(cfg)
    for img in tqdm(imgs,desc='({}/{})cuda:{}推理{}'.format(idx+1,len(weights),cuda_id,model_name),ncols=100,colour=COLORS[int(cuda_id)],
                    position=int(cuda_id),mininterval=1):
        filename = os.path.basename(img)
        reult = predictor(cv2.imread(img,0))
        predictions = reult["instances"].to("cpu")
        regions = model_to_json(predictions)
        new_json[filename] = {}
        new_json[filename]["filename"] = filename
        new_json[filename]["regions"] = regions
        new_json[filename]["type"] = "inf"
    
    json_name = os.path.basename(cfg.MODEL.WEIGHTS).split(".")[0]
    save_json(new_json, cfg.WEIGHTS_JSON_DIR, json_name)
    q.put(cuda_id)
    s.release()


def multi_weights_inf(cfg,imgs,weights):
    ps = []
    s = mp.Semaphore(len(cfg.GPU_NUMS))
    q = mp.Queue(maxsize=len(cfg.GPU_NUMS))
    for i in cfg.GPU_NUMS:
        q.put(i)
    for idx in range(len(weights)): 
        pool_cfg = copy.deepcopy(cfg)
        pool_cfg.MODEL.WEIGHTS = weights[idx]
        ps.append(mp.Process(target=worker, args=(s,q,idx,pool_cfg,weights,imgs)))

    for p in ps:
        p.start()

    for p in ps:
        p.join()
    






