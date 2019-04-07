import main
import json

from mean_average_precision.detection_map import DetectionMAP
from mean_average_precision.utils.show_frame import show_frame
import numpy as np
import matplotlib.pyplot as plt
import unicodedata


def get_IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_top1(pred_bb, pred_cl, gt_bb, gt_cl, countT, countF):
    countF += len(gt_bb)
    for i in range(len(pred_bb)):
        maxIOU = 0
        c = False
        for j in range(len(gt_bb)):
            IOU = get_IOU(pred_bb[i], gt_bb[j])
            if (IOU > maxIOU):
                maxIOU = IOU
                c = pred_cl[i] == gt_cl[j]
        if maxIOU >= 0.5 and c == True:
            countT += 1
    return countT, countF

frames = []

imgs_path = "/home/maxgod/Downloads/photo/photo"
json_file = "/home/maxgod/Downloads/photo/project1_new.json"

with open(json_file, "r") as read_file:
    data = json.load(read_file)

data = data['_via_img_metadata']
count = 0

classes = {'MXG': 0, 'Sanaken': 1, 'Zofinka': 2, 'Toalk': 3, 'Zissxzirsziiss': 4, 'kiasummer': 5, 'Unknown': 6}


countT = 0
countF = 0

for key in data.keys():
    pred_bb = []
    pred_cl = []
    pred_conf = []

    gt_bb = []
    gt_cl = []
    count += 1
    img_file = data[key]['filename']

    regions = data[key]['regions']
    for region in regions:
        shape = region['shape_attributes']
        gt_bb.append([shape['x'], shape['y'], shape['x'] + shape['width'], shape['y'] + shape['height']])
        name = region['region_attributes']['class']
        #name = unicodedata.normalize('NFKD', name).encode('ascii','ignore')
        gt_cl.append(classes[name])

    predicted = main.main('metr', imgs_path + "/" + img_file)

    for pred in predicted:
        pred_bb.append([pred['x1'], pred['y1'], pred['x2'], pred['y2']])
        pred_cl.append(classes[pred['class']])
        pred_conf.append(pred['conf'])

    pred_bb = np.array(pred_bb)
    pred_cl = np.array(pred_cl)
    pred_conf = np.array(pred_conf)

    gt_bb = np.array(gt_bb)
    gt_cl = np.array(gt_cl)

    frames.append((pred_bb, pred_cl, pred_conf, gt_bb, gt_cl))
    countT, countF = get_top1(pred_bb, pred_cl, gt_bb, gt_cl, countT, countF)
    #if count > 3:
    #    break

n_class = 7

mAP = DetectionMAP(n_class)
for i, frame in enumerate(frames):
    print("Evaluate frame {}".format(i))
    #show_frame(*frame)
    mAP.evaluate(*frame)

countT = countT + 0.0
print (countT)
print (countF)
print ("top1 error = " + str(countT/countF))

mAP.plot()
plt.show()
