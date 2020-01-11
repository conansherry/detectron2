from my_imports import *
from my_detectron_main import *

def train(net_name):
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file(NETS[net_name]['config_file']))
    cfg.DATASETS.TRAIN = ("bus_train",)
    cfg.DATASETS.TEST = ("bus_val",)
    cfg.DATALOADER.NUM_WORKERS = 1


    cfg.MODEL.WEIGHTS = NETS[net_name]['coco_weights']

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.MAX_ITER = 300
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def eval(metadata, net_name, pictures_dir, split_lines_eval):
    cfg = get_cfg()

    cfg.merge_from_file(get_config_file(NETS[net_name]['config_file']))
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.DATASETS.TEST = ("bus_val",)
    predictor = DefaultPredictor(cfg)

    eval_dataset_dict = get_bus_dicts(pictures_dir, split_lines_eval)
    k=0
    for d in random.sample(eval_dataset_dict, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=0.2,
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2_imshow(v.get_image()[:, :, ::-1])
        cv2.imwrite('c' + str(k) + '.jpg', (v.get_image()[:, :, ::-1]))
        k+=1