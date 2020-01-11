SPLIT_RATE = 0.66

NETS = {}

NETS['faster_rcnn_R_50_FPN_3x'] = { 'config_file'    :"COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                                    'coco_weights'   :'model_final_280758.pkl'}