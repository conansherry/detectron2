from my_utils import *
from net_methods import *

def get_bus_dicts(img_dir, relevant_lines):

    dataset_dicts = []

    for line_idx, line in enumerate(relevant_lines):

        image_filename = line.split(':')[0]
        image_id = line_idx + 1

        tags, tag_class = unwrap_tags(relevant_lines[line_idx], is_xyxy=False)

        record = {}

        full_filename = os.path.join(img_dir, image_filename)
        image_height, image_width = cv2.imread(full_filename).shape[:2]

        record["file_name"] = full_filename
        record["image_id"] = image_id
        record["height"] = image_height
        record["width"] = image_width

        objs = []
        for j in range(len(tag_class)):

            obj = {
                "bbox": tags[j],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": int(tag_class[j] - 1),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def verification(metadata, pictures_dir, annotation_lines):

    dataset_dicts = get_bus_dicts(pictures_dir, annotation_lines)

    l = 0
    for d in random.sample(dataset_dicts, 10):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.2)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite('a' + str(l) + '.jpg', (vis.get_image()[:, :, ::-1]))
        l+=1



def train_and_eval(split_rate):

    max_iter = 400
    lr = 0.01
    pictures_dir = os.path.join("..", "data", "originalData", "pictures")
    annotations_path = os.path.join(pictures_dir, 'originalAnnotationsTrain.txt')

    split_lines = get_split_lines(annotations_path, split_rate)

    buses_metadata = {}
    for d in ['train', 'val']:
        DatasetCatalog.register("bus_" + d, lambda d=d: get_bus_dicts(img_dir=pictures_dir, relevant_lines=split_lines[d]))
        MetadataCatalog.get("bus_" + d).set(thing_classes=['0', '1', '2', '3', '4', '5'])
        buses_metadata[d] = MetadataCatalog.get("bus_" + d)

    ver_type = 'val'
    verification(buses_metadata[ver_type], pictures_dir, split_lines[ver_type])

    net_name = 'faster_rcnn_R_50_FPN_3x'

    print('started training')
    train(net_name, True, max_iter, lr)
    print('finished training')


    eval_run_type = 'val'
    print('starting evaluation')
    eval(buses_metadata[eval_run_type], net_name, pictures_dir, split_lines[eval_run_type])
    print('finished evaluation')


def get_split_lines(annotations_path, split_rate):
    # Load tagging files
    gt_file = open(annotations_path, 'r')
    gt_lines = gt_file.readlines()
    random.shuffle(gt_lines)
    split_index = int(len(gt_lines) * split_rate) + 1
    split_lines = {'train': gt_lines[:split_index], 'val': gt_lines[split_index:]}

    return split_lines


if __name__ == '__main__':
    train_and_eval(split_rate=SPLIT_RATE)