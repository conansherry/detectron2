import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import ast

import imageio


fontdict = {'fontsize':15, 'weight':'bold'}
plt.switch_backend('Qt5Agg')

class IMAGE:

    def __init__(self):
        self.ROIS = []

    def set_image(self, im):
        self.I = imageio.imread(im)
        self.imShape = self.I.shape

    def clear_ROIS(self):
        self.ROIS = []

    def add_ROI(self, pos):
        self.ROIS.append(pos)

    def show_ROI(self, title, edgecolor, numGT, text, saveDir = None):
        fig, ax = plt.subplots(1)
        ax.imshow(self.I)
        if(not isinstance(edgecolor,list) and len(self.ROIS) > 0):
            edgecolor = [edgecolor] * len(self.ROIS)
        for i in range(0,numGT):
            ROI = self.ROIS[i]
            rect = patches.Rectangle((ROI[0], ROI[1]), ROI[2], ROI[3], linewidth = 1, edgecolor = edgecolor[i], facecolor='none')
            ax.add_patch(rect)
        for i in range(numGT,len(self.ROIS)):
            ROI = self.ROIS[i]
            rect = patches.Rectangle((ROI[0], ROI[1]), ROI[2], ROI[3], linewidth = 1, edgecolor = edgecolor[i], facecolor='none', linestyle = '--')
            ax.add_patch(rect)
        if(saveDir is None):
            ax.text(15,160,text, fontdict = fontdict, bbox={'facecolor':'yellow', 'edgecolor':'yellow','alpha':0.5, 'pad':2})
        else:
            ax.text(15, 300, text, fontdict=fontdict,bbox={'facecolor': 'yellow', 'edgecolor': 'yellow', 'alpha': 0.5, 'pad': 2})
        plt.title(title)
        if(not saveDir is None):
            plt.savefig(os.path.join(saveDir, title), dpi = 500)
        plt.close()

    def close(self):
        plt.close()

def IOU(boxAList, boxBList):
    Th = 0.69
    iou = []
    matches = {}
    tp = 0
    fp = len(boxBList)
    missed = len(boxAList)
    for i in range(len(boxAList)):
        boxA = boxAList[i][:4]
        iou_ = []
        for j in range(len(boxBList)):
            boxB = boxBList[j][:4]
            if(not ((boxB[0] <= boxA[0] <= boxB[0] + boxB[2]) or (boxA[0] <= boxB[0] <= boxA[0] + boxA[2]))):
                iou_.append(0.0)
                continue
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
            yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
            interArea = (xB - xA + 1) * (yB - yA + 1)
            boxAArea = (boxA[2] + 1)*(boxA[3] + 1)
            boxBArea = (boxB[2] + 1)*(boxB[3] + 1)
            iou_.append(interArea / float(boxAArea + boxBArea - interArea))
        maxIou = max(iou_)
        maxIouIndex = iou_.index(max(iou_))
        iou.append(maxIou)
        if (maxIouIndex in matches and maxIou > iou[matches[maxIouIndex]]):
            if (iou[matches[maxIouIndex]] > Th and boxAList[matches[maxIouIndex]][4] == boxBList[maxIouIndex][4]):
                pass
            elif(maxIou > Th and boxAList[i][4] == boxBList[maxIouIndex][4]):
                tp += 1
                missed -= 1
                fp -= 1
            matches[maxIouIndex] = i
        if(not maxIouIndex in matches):
            matches[maxIouIndex] = i
            if(maxIou > Th and boxAList[i][4] == boxBList[maxIouIndex][4]):
                tp += 1
                missed -= 1
                fp -= 1
    return tp, fp, missed, iou

def runTest(annFileNameGT, myAnnFileName, busDir , saveDir = None, elapsed = None):

    image = IMAGE()
    objectsColors = {'g':'1', 'y':'2', 'w':'3', 's':'4', 'b':'5', 'r':'6'}
    objectsColorsInv = {v: k for k, v in objectsColors.items()}
    objectsColorsForShow = {'g':'g', 'y':'y', 'w':'w', 's':'tab:gray', 'b':'b', 'r':'r'}

    writtenAnnsLines = {}
    annFileEstimations = open(myAnnFileName, 'r')
    annFileGT = open(annFileNameGT, 'r')
    writtenAnnsLines['Ground_Truth'] = (annFileGT.readlines())
    writtenAnnsLines['Estimation'] = (annFileEstimations.readlines())

    TP = 0
    FP = 0
    MISS = 0

    for i in range(len(writtenAnnsLines['Ground_Truth'])):

        lineGT = writtenAnnsLines['Ground_Truth'][i].replace(' ','')
        colors = []
        imName = lineGT.split(':')[0]
        lineE = [x for x in writtenAnnsLines['Estimation'] if imName == x.split(':')[0]]
        if(len(lineE) == 0):
            lineE = imName + ':'
        else:
            lineE = lineE[0]
        bus = os.path.join(busDir, imName)
        image.set_image(bus)
        image.clear_ROIS()
        annsGT = lineGT[lineGT.index(':') + 1:].replace('\n', '')
        annsE = lineE[lineE.index(':') + 1:].replace('\n', '')
        annsGT = ast.literal_eval(annsGT)
        if (not isinstance(annsGT, tuple)):
            annsGT = [annsGT]
        for ann in annsGT:
            image.add_ROI(ann[:4])
            colorTag = objectsColorsInv[str(ann[4])]
            colors.append(objectsColorsForShow[colorTag])
        numGT = len(annsGT)
        if('[' in lineE):
            annsE = ast.literal_eval(annsE)
            if (not isinstance(annsE, tuple)):
                annsE = [annsE]
            for ann in annsE:
                image.add_ROI(ann[:4])
                colorTag = objectsColorsInv[str(ann[4])]
                colors.append(objectsColorsForShow[colorTag])
            tp, fp, missed, iou = IOU(annsGT, annsE)
        else:
            tp = 0
            fp = 0
            numGT = 0
            missed = len(annsGT)
            iou = []
        TP += tp
        FP += fp
        MISS += missed
        print('Image : {}, TP: {} FP: {} MISSED : {}'.format(imName, tp, fp, missed))
        iouStr = ','.join(['{0:.2f}'.format(x) for x in iou])
        text = 'IOU Scores : ' + iouStr + '\nTP = {}, FP = {}, Missed = {} '.format(tp, fp, missed)
        image.show_ROI(edgecolor = colors, title = imName, numGT = numGT , text = text, saveDir = saveDir)

    if(TP == 0):
        F1Score = 0
    else:
        precision = TP/(TP + FP)
        recall = TP/(TP + MISS)
        F1Score = 2*(precision * recall)/(precision + recall)
    strToWrite = 'Total detections = {}/{}\nTotal False Positives = {}\nTotal missed = {}'.format(TP, TP+MISS, FP, MISS)
    strToWrite += '\nF1 SCORE : {0:.3f}'.format(F1Score)
    if(not elapsed is None):
        strToWrite += '\nTime elapsed : {0:.2f} seconds'.format(elapsed)
    fig, ax = plt.subplots(1)
    plt.title('Results', fontdict = {'fontsize':20})
    im = np.zeros((10,26,3), dtype=np.uint8)
    im[:,:,2] = 221
    im[:,:,1] = 114
    im[:,:,0] = 102
    ax.imshow(im)
    ax.text(4,7, strToWrite, style='italic', fontdict = {'fontsize':50, 'weight':'bold'})
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    plt.close()
    FIG, ax = plt.subplots(1)
    plt.title('Results', fontdict = {'fontsize':20})
    ax.imshow(im)
    ax.text(2,8, strToWrite, style='italic', fontdict = {'fontsize':20, 'weight':'bold'})
    if(saveDir is None):
        saveDir = os.path.join(os.getcwd(), 'Output')
        if(not os.path.exists(saveDir)):
            os.mkdir(saveDir)
    plt.savefig(os.path.join(saveDir,'Results.png'), dpi = 600)
