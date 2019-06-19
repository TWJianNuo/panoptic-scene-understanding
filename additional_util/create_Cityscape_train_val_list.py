import glob
import numpy as np
import os
import random
def generateCityScapeSplit(datasetLoc, splitFileLoc):
    kittiLength = 45200 # maintain the other file is around 10000
    to_expand = 10000
    fineList = list()
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train", "*")):
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            fineList.append(imagePath)
    boostTime = np.ceil(to_expand / len(fineList))
    blendedList = fineList * np.int(boostTime)
    random.shuffle(blendedList)
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    for index, imagePath in enumerate(blendedList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileTrain.writelines(writel)
    fileTrain.close()
    # fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    # for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train", "*")):
    #     index = 0
    #     for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
    #         split_comp = imagePath.split("/")
    #         writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
    #         writeComp2 = split_comp[-1]
    #         if np.random.random(1) > 0.5:
    #             writeComp3 = 'l'
    #         else:
    #             writeComp3 = 'r'
    #         writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
    #             writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
    #         fileTrain.writelines(writel)
    #         index = index + 1
    # fileTrain.close()


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
def generateCityScape_scaleVerification_split(datasetLoc, splitFileLoc):
    # This is only served for generating validation file for scale adjustment
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train", "*")):
        index = 0
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            # if np.random.random(1) > 0.5:
            #     writeComp3 = 'l'
            # else:
            #     writeComp3 = 'r'
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileTrain.writelines(writel)
            index = index + 1
    fileTrain.close()

def generateCityScapeSplitExtra(datasetLoc, splitFileLoc):
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    # Balance fine and coarse label

    fineList = list()
    coarseList = list()
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train_extra", "*")):
        index = 0
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            coarseList.append(imagePath)
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train", "*")):
        # index = 0 # continues index
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            fineList.append(imagePath)
    boostTime = np.ceil(len(coarseList) / len(fineList))
    blendedList = fineList * np.int(boostTime) + coarseList
    random.shuffle(blendedList)

    for index, imagePath in enumerate(blendedList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileTrain.writelines(writel)
    fileTrain.close()

    # for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train_extra", "*")):
    #     index = 0
    #     for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
    #         split_comp = imagePath.split("/")
    #         writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
    #         writeComp2 = split_comp[-1]
    #         writeComp3 = 'l'
    #         writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
    #             writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
    #         fileTrain.writelines(writel)
    #         index = index + 1
    # for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train", "*")):
    #     for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
    #         split_comp = imagePath.split("/")
    #         writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
    #         writeComp2 = split_comp[-1]
    #         writeComp3 = 'l'
    #         writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
    #             writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
    #         fileTrain.writelines(writel)
    #         index = index + 1
    # fileTrain.close()
    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "val", "*")):
        index = 0
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileVal.writelines(writel)
            index = index + 1
    fileVal.close()
def generateCityScapeSplit_riginalSize(datasetLoc, splitFileLoc):
    kittiLength = 45200 # maintain the other file is around 10000
    to_expand = 10000
    fineList = list()
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train", "*")):
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            fineList.append(imagePath)
    boostTime = 1
    blendedList = fineList * np.int(boostTime)
    random.shuffle(blendedList)
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    for index, imagePath in enumerate(blendedList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileTrain.writelines(writel)
    fileTrain.close()
    # fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    # for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train", "*")):
    #     index = 0
    #     for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
    #         split_comp = imagePath.split("/")
    #         writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
    #         writeComp2 = split_comp[-1]
    #         if np.random.random(1) > 0.5:
    #             writeComp3 = 'l'
    #         else:
    #             writeComp3 = 'r'
    #         writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
    #             writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
    #         fileTrain.writelines(writel)
    #         index = index + 1
    # fileTrain.close()


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
    # datasetLoc = "/media/shengjie/other/cityscapesData"
    # splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/cityscape"
    # generateCityScapeSplit(datasetLoc, splitFileLoc)

    # datasetLoc = "/media/shengjie/other/cityscapesData"
    # splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/cityscape_verification_scale"
    # generateCityScape_scaleVerification_split(datasetLoc, splitFileLoc)

    # datasetLoc = "/media/shengjie/other/cityscapesData"
    # splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/cityscapeExtra"
    # generateCityScapeSplitExtra(datasetLoc, splitFileLoc)

    datasetLoc = "/media/shengjie/other/cityscapesData"
    splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/cityscape_original"
    generateCityScapeSplit_riginalSize(datasetLoc, splitFileLoc)