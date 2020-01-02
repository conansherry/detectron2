# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
import torch, torchvision

setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.model_zoo.model_zoo import get_config_file
from detectron2.utils.visualizer import ColorMode

import os
import json
from detectron2.structures import BoxMode
import itertools
import re

IMAGE_HEIGHT = 2736
IMAGE_WIDTH = 3648

def find_line_index(lines, name):
    '''
    This function finds the index of last line that contains the given name. if
    there is not one - the function will return -1.
    Inputs:
        lines - An arrays of strings
        name - name to be search
    output:
        k0 - An integer which indicates in which line the last name appears. if
            it doesn't appear at all it will return -1.
    '''
    k0 = -1
    for k in range(len(lines)):
        line = lines[k]
        if name in line:
            k0 = k
    return k0


def unwrap_tags(line, is_xyxy):
    '''
        This function gets the string of a line in the tags txt file, and
        returns the line bounding-box 2 coordinates (x1,y1,x2,y2) and the class
        of each bounding box tag. if it invalid line it returns Null.
        Input:
            line - string, which contain the bounding boxes.
        Ouput:
            boxes - Numpy-array Nx4 which contain for N bounding boxes taggings
                    in the picture the 2-coordinates for the bounding box
                    (x1, y1, x2, y2) as floats.
            tag_class - returns Nx1 array which contains the class of each box.
    '''
    # spec_line = re.sub(img_name+img_suffix+':','', bbox_lines[line]).split()[-1]
    if line.find(':') == -1:
        print('ERROR')
        return
    line = line[line.find(':') + 1:-1].split()[-1]
    # print(line)
    pre_boxes = line.split('],[')
    boxes = np.zeros((len(pre_boxes), 5))
    for k in range(len(pre_boxes)):
        pre_boxes[k] = re.sub('[[]', '', pre_boxes[k])
        pre_boxes[k] = re.sub('[]]', '', pre_boxes[k])
        tmp = np.array(pre_boxes[k].split(',')).astype(int)
        boxes[k] = tmp

    if is_xyxy:
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    tag_class = boxes[:, 4].astype(int)
    return boxes[:, :-1], tag_class

def get_balloon_dicts(img_dir):
    print('balloon')
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def get_bus_dicts(img_dir):

    annotations_path = os.path.join(img_dir, 'originalAnnotationsTrain.txt')

    # Load tagging files
    f_original = open(annotations_path, 'r')

    original_lines = f_original.readlines()

    images_list = []
    image_names = []
    dataset_dicts = []

    for line_idx, line in enumerate(original_lines):
        annotations = []

        image_filename = line.split(':')[0]
        image_id = line_idx + 1

        image_annotations_list = []

        tags, tag_class = unwrap_tags(original_lines[line_idx], is_xyxy=False)

        # for i in range(len(tag_class)):
        #     annotation = {
        #         "segmentation": [],
        #         "area": tags[i][2] * tags[i][3],
        #         "iscrowd": 0,
        #         "image_id": image_id,
        #         "bbox": list(tags[i]),
        #         "category_id": int(tag_class[i]),
        #         "id": i
        #     }
        #     image_annotations_list += [annotation]

        record = {}

        full_filename = os.path.join(img_dir, image_filename)
        # height, width = cv2.imread(full_filename).shape[:2]

        record["file_name"] = full_filename
        record["image_id"] = image_id
        record["height"] = IMAGE_HEIGHT
        record["width"] = IMAGE_WIDTH


        objs = []
        for j in range(len(tag_class)):

            obj = {
                "bbox": tags[j],
                "bbox_mode": BoxMode.XYWH_ABS,
                # "segmentation": [],
                "category_id": int(tag_class[j]), #TODO return tu tag_class[j]
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def verification(metadata, type):
    if type == 'balloon':
        dataset_dicts = get_balloon_dicts(os.path.join("..", "..", "balloon_dataset", "balloon", "train"))
    else:
        dataset_dicts = get_bus_dicts(os.path.join("..", "data", "originalData", "pictures"))

    l = 0
    for d in random.sample(dataset_dicts, 10):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite('a' + str(l) + '.jpg', (vis.get_image()[:, :, ::-1]))
        l+=1



def train():
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("bus_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1

    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"

    # cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/fast_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl"  # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = "model_final_b275ba.pkl"  # Let training initialize from model zoo

    cfg.MODEL.LOAD_PROPOSALS = False
    # cfg.DATASETS.PROPOSAL_FILES_TRAIN = "coco_2017_train_box_proposals_21bc3a.pkl"

    # cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/fast_rcnn_R_50_FPN_1x/137849600/model_final_f10217.pklmodel_final_b275ba.pkl"  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 30    # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # only has one class (balloon)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

def eval(metadata):
    cfg = get_cfg()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # TODO change back to 0.7 set the testing threshold for this model
    cfg.DATASETS.TEST = ("bus_train",)
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_bus_dicts(os.path.join("..", "data", "originalData", "pictures"))
    k=0
    for d in random.sample(dataset_dicts, 10):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=0.8,
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2_imshow(v.get_image()[:, :, ::-1])
        cv2.imwrite('b' + str(k) + '.jpg', (v.get_image()[:, :, ::-1]))
        k+=1


def main():
    # for balloon
    # for d in ["train", "val"]:
    #     DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("../../balloon_dataset/balloon/" + d))
    #     MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
    # balloon_metadata = MetadataCatalog.get("balloon_train")

    # verification(balloon_metadata, 'balloon')


    for d in ["train"]:
        DatasetCatalog.register("bus_" + d, lambda d=d: get_bus_dicts(os.path.join("..", "data", "originalData", "pictures")))
        # MetadataCatalog.get("bus_" + d).set(thing_classes=['bla', 'a', 'b', 'c', 'd', 'e', 'f'])
        MetadataCatalog.get("bus_" + d).set(thing_classes=['0', '1', '2', '3', '4', '5', '6'])
    buses_metadata = MetadataCatalog.get("bus_train")

    verification(buses_metadata, 'bus')

    train()
    print('finished training')
    print('started training')
    eval(buses_metadata)
    print('finished evaluation')


if __name__ == '__main__':
    main()