import main

import sys
import subprocess
import os
import json
import numpy as np


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

def get_top1(pred_bb, pred_cl, pred_conf, gt_bb, gt_cl, countT, countF, countFaces, countFounded):
    countFaces += len(gt_cl)
    countF += len(gt_cl)
    for i in range(len(gt_bb)):
        maxIOU = 0
        c = False
        for j in range(len(pred_bb)):
            IOU = get_IOU(pred_bb[j], gt_bb[i])
            if (IOU > maxIOU):
                maxIOU = IOU
                c = pred_cl[j] == gt_cl[i]
                conf = pred_conf[j]
                if c == False:
                    print("pred = " + str(pred_cl[j]) + " gt = " + str(gt_cl[i]))
        if maxIOU > 0 and c == True:
            countT += 1
        if conf >= 0.5 and maxIOU > 0:
            countFounded += 1
        if maxIOU < 0.5:
            countF -= 1
    return countT, countF, countFaces, countFounded

persons = ["otherPeople", "Sanaken", "kiasummer", "MXG", "toalk", "zofinka", "zissxzirsziiss"]
mode = "test"

out_path = "mAP/input"

print("Recount vects?[yes/no]")
ans = sys.stdin.readline()


if ans == "yes\n":
    for the_file in os.listdir(out_path + "/ground-truth/"):
        file_path = os.path.join(out_path + "/ground-truth/", the_file)
        if os.path.isfile(file_path):
            os.unlink(file_path)

    for the_file in os.listdir(out_path + "/detection-results/"):
        file_path = os.path.join(out_path + "/detection-results/", the_file)
        if os.path.isfile(file_path):
            os.unlink(file_path)


    countT = 0
    countF = 0
    countFaces = 0
    countFounded = 0
    numImgs = 0

    for person in persons:
        imgs_path = "/home/afr/Downloads/datatset/" + person

        with open(imgs_path + "/" + person + "_" + mode + "_via_region_data.json", "r") as read_file:
            data = json.load(read_file)


        count = 0

        if person == "Sanaken" or person == "otherPeople":
            data = data['_via_img_metadata']

        for key in data.keys():
            numImgs += 1
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
                gt_cl.append(name)
                with open(out_path + "/ground-truth/" + img_file.split(".")[0] + ".txt", "a") as w_file:
                    w_file.write("face " + str(shape['x']) + " " + str(shape['y']) + " " + str(
                        shape['x'] + shape['width']) + " " + str(shape['y'] + shape['height']) + "\n")

            predicted = main.main('metr', imgs_path + "/" + mode + "/" + img_file, 'rec')

            for pred in predicted:
                with open(out_path + "/detection-results/" + img_file.split(".")[0] + ".txt", "a") as w_file:
                    w_file.write("face " + str(pred['conf']) + " " + str(pred['x1']) + " " + str(pred['y1']) + " " + str(
                        pred['x2']) + " " + str(pred['y2']) + "\n")
                pred_bb.append([pred['x1'], pred['y1'], pred['x2'], pred['y2']])
                pred_cl.append(pred['class'])
                pred_conf.append(pred['conf'])

            pred_bb = np.array(pred_bb)
            pred_cl = np.array(pred_cl)

            gt_bb = np.array(gt_bb)
            gt_cl = np.array(gt_cl)

            countT, countF, countFaces, countFounded = get_top1(pred_bb, pred_cl, pred_conf, gt_bb, gt_cl, countT, countF, countFaces, countFounded)


    countT = countT + 0.0
    #print(countT)
    #print(countF)
    print("top1 error = " + str(countT/countF))

    #print(str(countFaces) + " " + str(countFounded))
    print("detected faces = ", + str(countFounded/countFaces))


subprocess.call("python " + out_path + "/../main.py --no-animation", shell=True)
