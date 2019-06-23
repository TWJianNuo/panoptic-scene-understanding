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
if __name__ == "__main__":
    datasetLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_semantics"
    splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/kitti_semantic_eval"
    generateKittySemanticSplit(datasetLoc, splitFileLoc)