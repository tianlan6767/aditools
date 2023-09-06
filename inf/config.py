from adet.config import get_cfg

def init_config():
    """
    绘图时需要
    cfg.TRAIN_JSON = "/media/ps/E80EDA380EDA000C/LB/train/resnet34/annotations"
    register_coco_instances("phone", {}, coco_dir, images_dir)
    fruits_nuts_metadata = MetadataCatalog.get("phone")
    dataset_dicts = DatasetCatalog.get("phone") coco_dir,images_dir=''
    cfg.DATASETS.TEST = ("phone", )
    """
    cfg = get_cfg()
    config_file = '/home/ps/adet/AdelaiDet/configs/BlendMask/R_50_3x.yaml'
    cfg.merge_from_file(config_file)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    # cfg.INPUT.FORMAT = 'L'
    cfg.GPU_NUMS = [0,1,2]   #torch.cuda.device_count()
    ######模型参数################
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 25
    cfg.MODEL.FCOS.NUM_CLASSES = 25
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.24
    # ## 855b
    # cfg.MODEL.PIXEL_MEAN = [77.857]
    # cfg.MODEL.PIXEL_STD = [54.068]
    # 827 808 784 819
    # cfg.MODEL.PIXEL_MEAN = [77.74]
    # cfg.MODEL.PIXEL_STD = [55.246]
    #299b
    # cfg.MODEL.PIXEL_MEAN = [113.131]
    # cfg.MODEL.PIXEL_STD = [79.277]
    # 093
    # cfg.MODEL.PIXEL_MEAN = [65.548]
    # cfg.MODEL.PIXEL_STD = [58.738]
    # 829xc
    # cfg.MODEL.PIXEL_MEAN = [67.111]
    # cfg.MODEL.PIXEL_STD = [55.246]
    # xian
    # cfg.MODEL.PIXEL_MEAN = [31.292]
    # cfg.MODEL.PIXEL_STD = [28.981]
    # # 684
    # cfg.MODEL.PIXEL_MEAN = [70.387]
    # cfg.MODEL.PIXEL_STD = [45.558]

    # 855g
    # cfg.MODEL.PIXEL_MEAN = [113.048, 114.843, 115.812]
    # cfg.MODEL.PIXEL_STD = [61.348, 59.195, 61.814]

    # laptop
    # cfg.MODEL.PIXEL_MEAN = [54.214]
    # cfg.MODEL.PIXEL_STD = [45.527]

     # laptopA
    cfg.MODEL.PIXEL_MEAN = [87.178]
    cfg.MODEL.PIXEL_STD = [40.895]
    
    # zk21
    # cfg.MODEL.PIXEL_MEAN = [28.82]
    # cfg.MODEL.PIXEL_STD = [29.03]
    # zk22
    # cfg.MODEL.PIXEL_MEAN = [64.11]
    # cfg.MODEL.PIXEL_STD = [65.40]
    # zk1232
    # cfg.MODEL.PIXEL_MEAN = [110.29]
    # cfg.MODEL.PIXEL_STD = [88.31]
    # zk1131
    # cfg.MODEL.PIXEL_MEAN = [59.109]
    # cfg.MODEL.PIXEL_STD = [55.331]
    # lxzk
    # cfg.MODEL.PIXEL_MEAN = [56]
    # cfg.MODEL.PIXEL_STD = [53]
    

    # 22v19
    # cfg.MODEL.PIXEL_MEAN = [96.943, 90.513, 86.63]
    # cfg.MODEL.PIXEL_STD = [83.149, 84.579, 84.473]

    cfg.MODEL.RESNETS.DEPTH = 34
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64

    


    ######测试图片参数##############
    cfg.TEST_IMG_DIR = "/media/ps/244e88e1-d2e1-477f-9e37-7b9cb43b842a/LB/val/laptopa/24ok"#"/home/ps/LB/code/coco-annotator/mask/HG827"      # 测试图片路径
    cfg.TEST_IMG_FULL = False                                    # "整图推理/分割图推理"
    cfg.TEST_IMG_IS_SEGMENT = False                             # 测试图片是否需要分割
    
    cfg.TEST_IMG_JSON = ""
    cfg.TEST_IMG_SEGMENT_DIR = cfg.TEST_IMG_DIR + '_segment'    # 测试图片分割路径
    cfg.INPUT.MIN_SIZE_TEST = 2048                        # 测试图片大小
    cfg.INPUT.MAX_SIZE_TEST = cfg.INPUT.MIN_SIZE_TEST
    ######权重参数################
    cfg.MODEL.WEIGHTS = "/home/ps/LB/train/laptopa/weights"        # 权重路径或文件名
    cfg.WEIGHTS_JSON_DIR = cfg.MODEL.WEIGHTS+"_json/24ok"            # 权重推理后保存路径
    cfg.WEIGHTS_INF_STEP = 1                                    # 权重筛选步长
    cfg.WEIGHTS_INF_DROP = 0                                    # 权重舍弃比率(0,1)
    


    # cfg.MODEL.RESNETS.DEPTH = 34
    # cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    # cfg.MODEL.BACKBONE.FREEZE_AT = 0
    # cfg.MODEL.FCOS.CENTER_SAMPLE = 'center'
    # cfg.MODEL.FPN.OUT_CHANNELS = 64
    # cfg.MODEL.FCOS.NUM_CLS_CONVS = 0
    # cfg.MODEL.FCOS.NUM_BOX_CONVS = 0

    # cfg.MODEL.FCOS.P2_NUM_CLS_CONVS = 0
    # cfg.MODEL.FCOS.P2_NUM_BOX_CONVS = 0
    # cfg.MODEL.FCOS.NUM_SHARE_CONVS =2

    # cfg.MODEL.FCOS.TOP_LEVELS = 2
    # cfg.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]      
    # cfg.MODEL.BASIS_MODULE.COMMON_STRIDE = 8

    # cfg.MODEL.FCOS.IN_FEATURES = ['p2', 'p3', 'p4', 'p5', 'p6', 'p7']
    # cfg.MODEL.FCOS.FPN_STRIDES = [4, 8, 16, 32, 64, 128]
    # cfg.MODEL.FCOS.SIZES_OF_INTEREST = [32, 64, 128, 256, 512]
    # cfg.MODEL.RESNETS.OUT_FEATURES = ['res2', 'res3', 'res4', 'res5']
    # cfg.MODEL.FPN.IN_FEATURES = ['res2', 'res3', 'res4', 'res5']

    # vgg
    # cfg.MODEL.REPVGG = CfgNode({})
    # cfg.MODEL.REPVGG.NAME = "RepVGG_w6"
    # cfg.MODEL.BACKBONE.NAME = "build_fcos_repvgg_fpn_backbone"
    # cfg.MODEL.REPVGG.OUT_FEATURES = ["res2","res3","res4","res5"]
    # cfg.MODEL.FPN.IN_FEATURES = ["res3","res4","res5"]
    # cfg.MODEL.FCOS.IN_FEATURES = ["p3","p4","p5","p6","p7"]
    # cfg.MODEL.FCOS.TOP_LEVELS =2
    # cfg.MODEL.FPN.OUT_CHANNELS = 256

    return cfg



    # cfg.MODEL.FCOS.USE_COMPRESSION = True
    # cfg.MODEL.FCOS.CMPRS_WIDTH = 32
    # cfg.MODEL.FCOS.CMPRS_ASPECT_RATIO = 0.25
    
    # cfg.MODEL.FPN.OUT_CHANNELS = 64
    # cfg.MODEL.FCOS.NUM_CLS_CONVS = 0
    # cfg.MODEL.FCOS.NUM_BOX_CONVS = 0

    # cfg.MODEL.FCOS.P2_NUM_CLS_CONVS = 0
    # cfg.MODEL.FCOS.P2_NUM_BOX_CONVS = 0
    # cfg.MODEL.FCOS.NUM_SHARE_CONVS = 2

    # cfg.MODEL.FCOS.TOP_LEVELS = 2            # 0:不使用p6p7, 1:使用p6, 2:使用p7
    # cfg.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]      
    # cfg.MODEL.BASIS_MODULE.COMMON_STRIDE = 8

    # cfg.MODEL.FCOS.IN_FEATURES = ["p2","p3","p4", "p5", "p6", "p7"]  # p2------p7
    # cfg.MODEL.FCOS.FPN_STRIDES = [4, 8, 16, 32, 64, 128]              # p2: 4, p3:8, p4：16, p5:32, p6:64, p7:128
    # cfg.MODEL.FCOS.SIZES_OF_INTEREST = [32, 64, 128, 256, 512]        # p2:(0, 32), p3:(32, 64), p4:(64, 128), p5:(128, 256), p6:(256, 512), p7:(512, 正无穷)
    # cfg.MODEL.RESNETS.OUT_FEATURES = ["res2","res3","res4", "res5"]   # 未使用p2, 可以删除"res2"
    # cfg.MODEL.FPN.IN_FEATURES = ["res2","res3","res4", "res5"]        # 未使用p2, 可以删除"res2"


    # CONVNEXT
    # cfg.MODEL.CONVNEXT = CfgNode({})
    # cfg.MODEL.BACKBONE.NAME = "build_fcos_convnext_fpn_backbone"
    # cfg.MODEL.CONVNEXT.NAME = "ConvNext-T"  # ["ConvNext-T","ConvNext-S","ConvNext-B","ConvNext-L","ConvNext-XL"]
    # cfg.MODEL.CONVNEXT.OUT_FEATURES = ["res2","res3","res4","res5"]
    # cfg.MODEL.FPN.IN_FEATURES = ["res3","res4","res5"]
    # cfg.MODEL.FCOS.IN_FEATURES = ["p3","p4","p5"]
    # cfg.MODEL.FPN.OUT_CHANNELS = 96
    

    