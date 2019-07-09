
imgpath1 = '/home/shengjie/Desktop/konigswinter_000000_000098_rightImg8bit.png';
imgpath2 = '/home/shengjie/Desktop/erlangen_000000_000136_rightImg8bit.png';
img1 = imread(imgpath1);
img2 = imread(imgpath2);
img1 = img1(:,:,1) - img1(:,:,2) - img1(:,:,2);
img2 = img2(:,:,1) - img2(:,:,2) - img2(:,:,2);
binmask = (img1 > 250) | (img2 > 250);
imshow(binmask);
imwrite(binmask, '/media/shengjie/other/sceneUnderstanding/monodepth2/assets/cityscapemask_right.png')


ctsRoot = '/media/shengjie/other/cityscapesData';
leftFolder = 'leftImg8bit';
rightFolder = 'rightImg8bit';
meanImg = computeCtsMean(ctsRoot, leftFolder);
varImg = cimputeCtsVar(meanImg, ctsRoot, leftFolder);
sumvar = sum(varImg, 3);
binimg = sumvar < 1.3e3;
imshow(binimg)
function meanImg = computeCtsMean(root, folder)
    folders = dir(fullfile(root, folder));
    meanImg = zeros(1024, 2048, 3);
    imgNums = 0;
    for i = 3 : size(folders, 1)
        tmpfold = fullfile(folders(i).folder, folders(i).name);
        imgdirs = dir(tmpfold);
        for j = 3 : size(imgdirs, 1)
            imgpaths = dir(fullfile(imgdirs(j).folder, imgdirs(j).name));
            for k = 3 : 20 : size(imgpaths, 1)
                imgpath = fullfile(imgpaths(k).folder, imgpaths(k).name);
                tmpimg = double(imread(imgpath));
                meanImg = meanImg + tmpimg;
                imgNums = imgNums + 1;
                sprintf('Mean : %d th img finished.\n', imgNums)
            end
        end 
    end
    meanImg = meanImg / imgNums;
end

function varImg = cimputeCtsVar(meanImg, root, folder)
    folders = dir(fullfile(root, folder));
    varImg = zeros(1024, 2048, 3);
    imgNums = 0;
    for i = 3 : size(folders, 1)
        tmpfold = fullfile(folders(i).folder, folders(i).name);
        imgdirs = dir(tmpfold);
        for j = 3 : size(imgdirs, 1)
            imgpaths = dir(fullfile(imgdirs(j).folder, imgdirs(j).name));
            for k = 3 : 20 : size(imgpaths, 1)
                imgpath = fullfile(imgpaths(k).folder, imgpaths(k).name);
                tmpimg = double(imread(imgpath));
                varImg = varImg + (tmpimg - meanImg).^2;
                imgNums = imgNums + 1;
                sprintf('Var : %d th img finished.\n', imgNums)
            end
        end 
    end
    varImg = varImg / imgNums;
end