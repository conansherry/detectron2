# Lior Magram - I.D 316113422, Yuval Noam Feinstein - I.D 206197816
# import some common libraries
import os
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import itertools
import torch, torchvision

# import some common detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.model_zoo.model_zoo import get_config_file


# def run1(myAnnFileName, buses):
#
#     annFileNameGT = os.path.join(os.getcwd(),'annotationsTrain.txt')
#     writtenAnnsLines = {}
#     annFileEstimations = open(myAnnFileName, 'w+')
#     annFileGT = open(annFileNameGT, 'r')
#     writtenAnnsLines['Ground_Truth'] = (annFileGT.readlines())
#
#     for k, line_ in enumerate(writtenAnnsLines['Ground_Truth']):
#
#         line = line_.replace(' ','')
#         imName = line.split(':')[0]
#         anns_ = line[line.index(':') + 1:].replace('\n', '')
#         anns = ast.literal_eval(anns_)
#         if (not isinstance(anns, tuple)):
#             anns = [anns]
#         corruptAnn = [np.round(np.array(x) + np.random.randint(low = 0, high = 100, size = 5)) for x in anns]
#         corruptAnn = [x[:4].tolist() + [anns[i][4]] for i,x in enumerate(corruptAnn)]
#         strToWrite = imName + ':'
#         if(3 <= k <= 5):
#             strToWrite += '\n'
#         else:
#             for i, ann in enumerate(corruptAnn):
#                 posStr = [str(x) for x in ann]
#                 posStr = ','.join(posStr)
#                 strToWrite += '[' + posStr + ']'
#                 if (i == int(len(anns)) - 1):
#                     strToWrite += '\n'
#                 else:
#                     strToWrite += ','
#         annFileEstimations.write(strToWrite)
#     return

def load_Base_RCNN_FPN():
    cfg = get_cfg()
    #cfg.merge_from_file(get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # Adding Base-RCNN-FPN configurations
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]  # One size for each in feature map
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000 # Per FPN Level
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000 # Per FPN Level
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV = 4
    cfg.MODEL.ROI_MASK_HEAD.POOLR_RESOLUTION = 14
    cfg.DATASETS.TRAIN = ("coco_2017_train",)
    cfg.DATASETS.TEST = ("coco_2017_val",)
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.STEPS = (60000, 80000)
    cfg.SOLVER.MAX_ITER = 90000
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    return cfg


def buses_predict(pred_obj, img):
    im_outputs = pred_obj(img)
    num_instances = len(im_outputs["instances"])
    boxes = im_outputs["instances"].get("pred_boxes").tensor
    classes = im_outputs["instances"].get("pred_classes")
    return boxes, classes, num_instances


def get_img_str(im_outputs, img_name, parse_cfg=None):
    '''
    This function gets the outputs of specific image from the preditor, and based on the cfg parses the outputs
    by the given format [X1, Y1, W, H, Class].
    :param im_outputs: The networks outputs after applying the image on the predictor.
    :param img_name: The name of the given image.
    :param parse_cfg: The configuration of the parsing, containing 3 settings: shifiting, XYXY or XYWH and integer
                      output of the location:
                    cls_shifted -   Our network is predicting color class between 0 to 5. if this setting is True, the
                                    parsing will shift the class to be in the range 1 to 6.
                    is_XYXY     -   This setting will determine if the given prediction output from the NN is in
                                    [X1,Y1,X2,Y2] format or in [X1,Y1,W,H] format. if is_XYXY is True it will convert
                                    it to [X1,Y1,W,H] format, else it will keep it in [X1,Y1,W,H] format.
                    is_int      -   Will determind if the [X1,Y1,W,H] will be parsed as integer or as floats. If it's
                                    True, it will round the [X1,Y1,W,H] values, else it will parse it as floats (As the
                                    predictor outputs get from the predictor function).
    :return: to_prnt -  A string which contains the estimated annotation of the given image from our predictor in the
                        Course's format of [X1, Y1, W, H, Class].
    '''
    to_prnt = ''

    # Loading the parsing configuration if given.
    if parse_cfg == None:  # Default parsing configuration
        cls_shifted = True  # Change from class range of 0-5 to a class range of 1-6
        is_XYXY = False  # The predictor output format is [X1,Y1,W,H]
        is_int = True  # Parse to integers representation
    else:
        cls_shifted, is_XYXY, is_int = parse_cfg  # Customize parsing configuration

    img_name = img_name.upper()
    to_prnt = to_prnt + img_name + ':'
    boxes, classes, num_instances = im_outputs
    if is_int:
        boxes = torch.round(boxes)
    if cls_shifted:
        classes = classes + 1  # Updating to range 1 to 6 clases
    if not is_XYXY:
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    for k in range(num_instances):
        obj_str = ''
        box = boxes[k]
        cl = classes[k]
        x1_bus, y1_bus, w_bus, h_bus = box
        if is_int:
            obj_str = '[%d,%d,%d,%d,%d],' % (x1_bus, y1_bus, w_bus, h_bus, cl)
        else:
            obj_str = '[%f,%f,%f,%f,%d],' % (x1_bus, y1_bus, w_bus, h_bus, cl)
        to_prnt = to_prnt + obj_str

    if num_instances:
        to_prnt = to_prnt[:-1]
    return to_prnt


def run(myAnnFileName, bus_dir):
    '''
    run -   This function gets the myAnnFilename (with path) and buses directory path and calculates the buses
            estimation for each image in the buses directory and saves all of the estimation into the an annotation
            file (myAnnFileName) with the given format: the name of the image in the beginning of the line,
             and then for each bus estimation will be presented as  [X1, Y1, W, H, Class].
             The run function uses the buses_predict(img_path) function from our detectron file:
                buses_predict(img_path) -   The function located at our detectron2 file. This function get an image path
                                            and eval the buses taggings in the image. it returns the network outputs, in
                                            the network predictor format.
    :param myAnnFileName:   The path for the annotation file of the estimated images.
    :param bus_dir:         The path for the buses images directory.
    :return:                None.
    '''

    setup_logger()

    cfg = load_Base_RCNN_FPN()

    # Adding COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" configurations
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.SOLVER.MAX_ITER = 270000
    cfg.SOLVER.STEPS = (210000, 250000)

    # Adding additional non-default configurations
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "weights_f1_1.000000_lr_0.008000_iter_450_ther_0.800000.pth")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.DATASETS.TEST = ("bus_val",)
    # cfg.TEST.AUG.ENABLED = True

    predictor = DefaultPredictor(cfg)
    annFileEstimations = open(myAnnFileName, 'w+')
    for file_name in os.listdir(bus_dir):
        if file_name.endswith(".JPG"):
            img_path = os.path.join(bus_dir, file_name)
            file_name = file_name.upper()
            print(file_name)
            img = cv2.imread(img_path)
            pred_outputs = buses_predict(predictor, img)
            pred_parsed = get_img_str(pred_outputs, file_name)
            annFileEstimations.write(pred_parsed + '\n')
    annFileEstimations.close()

    return

# if __name__ == '__main__':
#     ann_file = os.path.join('test_output.txt')
#     bus_dir = os.path.join("data","val")
#     # bus_dir = os.path.join("..","toyData","val")
#     run(ann_file, bus_dir)
