import glob
import numpy as np
import os
import random
def generateKittySemanticSplit(datasetLoc, splitFileLoc):
    val_fineList = list()
    for imagePath in glob.glob(os.path.join(datasetLoc, "training", "image_2", "*")):
        val_fineList.append(imagePath)
    random.shuffle(val_fineList)
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    fileTrain.close()

    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for index, imagePath in enumerate(val_fineList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileVal.writelines(writel)
    fileVal.close()
def generateKittiSemanDepthSplit(mappingFileLoc, splitFileLoc):
    with open('/media/shengjie/other/sceneUnderstanding/monodepth2/splits/train_mapping.txt') as f:
        mapping = f.readlines()

    mapping = [x.strip() for x in mapping]
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    for line in mapping:
        if len(line) > 1:
            lineComp = line.split(' ')
            writel = lineComp[0] + '/' + lineComp[1] + ' ' + str(int(lineComp[2])) + ' ' + 'l\n'
            fileTrain.writelines(writel)
    fileTrain.close()

    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for line in mapping:
        if len(line) > 1:
            lineComp = line.split(' ')
            writel = lineComp[0] + '/' + lineComp[1] + ' ' + str(int(lineComp[2])) + ' ' + 'l\n'
            fileVal.writelines(writel)
    fileVal.close()

def generateKittiToyExaple(mappingFileLoc, splitFileLoc):
    repeatTime = 5000
    with open('/media/shengjie/other/sceneUnderstanding/monodepth2/splits/train_mapping.txt') as f:
        mapping = f.readlines()

    mapping = [x.strip() for x in mapping]
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    for i in range(repeatTime):
        line = mapping[2]
        lineComp = line.split(' ')
        writel = lineComp[0] + '/' + lineComp[1] + ' ' + str(int(lineComp[2])) + ' ' + 'l\n'
        fileTrain.writelines(writel)
    fileTrain.close()

    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for i in range(repeatTime):
        line = mapping[2]
        lineComp = line.split(' ')
        writel = lineComp[0] + '/' + lineComp[1] + ' ' + str(int(lineComp[2])) + ' ' + 'l\n'
        fileVal.writelines(writel)
    fileVal.close()

if __name__ == "__main__":
    # datasetLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_semantics"
    # splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/kitti_semantic_eval"
    # generateKittiSemanDepthSplit(datasetLoc, splitFileLoc)

    # mappingFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_semantics"
    # splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/kitti_seman_mapped2depth"
    # generateKittiSemanDepthSplit(mappingFileLoc, splitFileLoc)

    mappingFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_semantics"
    splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/kitti_seman_mapped_toy"
    generateKittiToyExaple(mappingFileLoc, splitFileLoc)