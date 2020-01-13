# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
# from google.colab.patches import cv2_imshow
import os
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.model_zoo.model_zoo import get_config_file
import matplotlib.pyplot as plt
from detectron2_tutorial import *


def main():
    # im = cv2.imread(os.path.join("data", "originalData", "pictures", "DSCF1013.JPG"))


    cfg = get_cfg()

    model = 'detection'

    if model == 'instance-segmentation':
        pass
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        # cfg.MODEL.MASK_ON = False
        cfg.MODEL.WEIGHTS = "model_final_f10217.pkl"
    elif model == 'detection':
        cfg.merge_from_file(get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = "model_final_68b088.pkl"

        # cfg.merge_from_file(get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        # cfg.MODEL.WEIGHTS = "model_final_280758.pkl"

        # cfg.merge_from_file(get_config_file("COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml"))
        # cfg.MODEL.WEIGHTS = "model_final_b275ba.pkl"

        # cfg.MODEL.LOAD_PROPOSALS = False
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the detectron2:// shorthand
    predictor = DefaultPredictor(cfg)


    # We can use `Visualizer` to draw the predictions on the image.
    pictures_path = os.path.join("..", "data", "originalData", "pictures")
    images_filenames = os.listdir(pictures_path)
    output_path = os.path.join("..", "data", "originalData", "output_pictures")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for idx, image_filename in enumerate(images_filenames):
        im = cv2.imread(os.path.join("data", 'originalData', 'pictures', image_filename))

        outputs = predictor(im)

        # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
        outputs["instances"].pred_classes
        outputs["instances"].pred_boxes

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        image_output_path = os.path.join(output_path, 'pred_x101_' + image_filename)
        plt.imsave(image_output_path, v.get_image()[:, :, ::-1])
    # cv2.imsave('funny.jpg', v.get_image()[:, :, ::-1])

if __name__ == '__main__':
    main()