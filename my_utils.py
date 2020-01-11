from my_imports import *

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