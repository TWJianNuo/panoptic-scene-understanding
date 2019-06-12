import glob
import numpy as np
import os
def generateCityScapeSplit(datasetLoc, splitFileLoc):
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train", "*")):
        index = 0
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            if np.random.random(1) > 0.5:
                writeComp3 = 'l'
            else:
                writeComp3 = 'r'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileTrain.writelines(writel)
            index = index + 1
    fileTrain.close()

    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "val", "*")):
        index = 0
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            if np.random.random(1) > 0.5:
                writeComp3 = 'l'
            else:
                writeComp3 = 'r'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileVal.writelines(writel)
            index = index + 1
    fileVal.close()

if __name__ == "__main__":
    datasetLoc = "/media/shengjie/other/cityscapesData"
    splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/cityscape"
    generateCityScapeSplit(datasetLoc, splitFileLoc)