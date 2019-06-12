from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from utils import set_axes_equal
from utils import reconstruct3dPts
from utils import project3dPts
from utils import diffICP
from kitti_utils import readCamParam
from kitti_utils import readSeman

import pykitti
from PIL import Image
import kitti_semantics_util.labels
from globalInfo import acGInfo
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline
from scipy.misc import imresize

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()



def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            np.save(name_dest_npy, scaled_disp.cpu().numpy())


            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            name_dest_im = os.path.join(output_directory, "{}_disp.jpg".format(output_name))
            plt.imsave(name_dest_im, disp_resized_np, cmap='magma', vmax=vmax)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

            # scaled_disp = scaled_disp.squeeze()
            scaled_disp = torch.nn.functional.interpolate(
                scaled_disp, (original_height, original_width), mode="bilinear", align_corners=False)
            scaled_disp = scaled_disp.squeeze().cpu().numpy()

    print('-> Done!')
    return scaled_disp




gInfo = acGInfo()
STEREO_SCALE_FACTOR = gInfo['STEREO_SCALE_FACTOR']

mapFilepath = gInfo['mapFilepath']
rawData_baseDir = gInfo['rawData_baseDir']
intemSvPath = os.path.join(gInfo['internalRe_add'], 'icpLoss')

exArgs = '--image_path assets/test_image.jpg --model_name mono+stereo_1024x320'
args = parse_args()

interestedType = [26, 27, 28, 29, 30, 31, 32, 33] # [car, truck, bus, caravan, trailer, train, motorcycle, bicycle]
globalCounts = 0
beforeError = list()
afterError = list()
with open(mapFilepath) as f:
    content = f.readlines()
    for idx, line in enumerate(content):
        if len(line) > 1:
            comp = line.split(' ')
            date = comp[0]
            drive = comp[1].split('_')[4]
            frame = int(comp[2])

            data = pykitti.raw(rawData_baseDir, date, drive, frames=[frame])
            velo = data.get_velo(0)

            img2path = os.path.join(rawData_baseDir, comp[0], comp[1], 'image_02', 'data', comp[2][:-1:] + '.png')
            img3path = os.path.join(rawData_baseDir, comp[0], comp[1], 'image_03', 'data', comp[2][:-1:] + '.png')
            img2 = mpimg.imread(img2path)
            img3 = mpimg.imread(img3path)
            instance_semantic_seg = readSeman(idx)
            instance_seg = instance_semantic_seg % 256
            semantic_seg = instance_semantic_seg // 256
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(velo[0::10, 0], velo[0::10,1], velo[0::10,2], s = 0.1)


            # read velo, rgb image and prepare #
            # intrinsic = data.calib.P_rect_20
            # extrinsic = data.calib.R_rect_00 @ data.calib[1]

            velo_fore = velo[:, 0:3]
            velo_fore = velo_fore[velo_fore[:, 0] > 0.1]
            velo_fore = np.insert(velo_fore, 3, 1, axis=1)


            # cam 2
            cam = 2
            intrinsic2, extrinsic2 = readCamParam(data, cam)
            velo_projected = (intrinsic2 @ extrinsic2 @ velo_fore.T).T
            velo_projected[:, 0] = velo_projected[:, 0] / velo_projected[:, 2]
            velo_projected[:, 1] = velo_projected[:, 1] / velo_projected[:, 2]

            args.image_path = img2path
            imgShape = img2.shape[0:2]
            img2Depth = test_simple(args)
            img2Depth = 1 / img2Depth * STEREO_SCALE_FACTOR
            img2Depth = imresize(img2Depth, imgShape, mode='F')
            # img2Depth = np.array(Image.fromarray(img2Depth).resize([imgShape[1], imgShape[0]], mode='F'))

            testPos = np.round(velo_projected[:, 0:2]).astype(np.intc)
            se1 = np.logical_and(testPos[:, 0] >= 0, testPos[:, 0] < imgShape[1])
            se2 = np.logical_and(testPos[:, 1] >= 0, testPos[:, 1] < imgShape[0])
            se = np.logical_and(se1, se2)
            testPos = testPos[se, :]


            predictedDepth = img2Depth[testPos[:, 1], testPos[:, 0]]
            recon3Pts2 = np.stack((testPos[:, 0] * predictedDepth, testPos[:, 1] * predictedDepth, predictedDepth, predictedDepth / predictedDepth), axis = 1)
            recon3Pts2 = (np.linalg.inv(intrinsic2 @ extrinsic2) @ recon3Pts2.T).T

            gtDepth = velo_projected[se, 2]
            gtRecon3Pts = np.stack((testPos[:, 0] * gtDepth, testPos[:, 1] * gtDepth, gtDepth, gtDepth / gtDepth), axis=1)
            gtRecon3Pts = (np.linalg.inv(intrinsic2 @ extrinsic2) @ gtRecon3Pts.T).T

            gtDepthMap1 = np.ones(imgShape) * np.inf
            for m in range(testPos.shape[0]):
                if gtDepth[m] < gtDepthMap1[testPos[m, 1], testPos[m, 0]]:
                    gtDepthMap1[testPos[m, 1], testPos[m, 0]] = gtDepth[m]
            # ---------------------------------------------------------------------------------------------#
            cam = 3
            intrinsic3, extrinsic3 = readCamParam(data, cam)
            velo_projected = (intrinsic3 @ extrinsic3 @ velo_fore.T).T
            velo_projected[:, 0] = velo_projected[:, 0] / velo_projected[:, 2]
            velo_projected[:, 1] = velo_projected[:, 1] / velo_projected[:, 2]

            args.image_path = img3path
            img3Depth = test_simple(args)
            img3Depth = 1 / img3Depth * STEREO_SCALE_FACTOR
            img3Depth = imresize(img3Depth, imgShape, mode='F')

            testPos = np.round(velo_projected[:, 0:2]).astype(np.intc)
            se1 = np.logical_and(testPos[:, 0] >= 0, testPos[:, 0] < imgShape[1])
            se2 = np.logical_and(testPos[:, 1] >= 0, testPos[:, 1] < imgShape[0])
            se = np.logical_and(se1, se2)
            testPos = testPos[se, :]



            predictedDepth = img3Depth[testPos[:, 1], testPos[:, 0]]
            recon3Pts3 = np.stack((testPos[:, 0] * predictedDepth, testPos[:, 1] * predictedDepth, predictedDepth, predictedDepth / predictedDepth), axis = 1)
            recon3Pts3 = (np.linalg.inv(intrinsic3 @ extrinsic3) @ recon3Pts3.T).T

            gtDepth = velo_projected[se, 2]
            gtRecon3Pts = np.stack((testPos[:, 0] * gtDepth, testPos[:, 1] * gtDepth, gtDepth, gtDepth / gtDepth), axis=1)
            gtRecon3Pts = (np.linalg.inv(intrinsic3 @ extrinsic3) @ gtRecon3Pts.T).T

            # fig = plt.figure()
            # ax = fig.add_subplot((111), projection='3d')
            # ax.scatter(recon3Pts3[::3, 0], recon3Pts3[::3, 1], recon3Pts3[::3, 2], s=0.1, c='g')
            # ax.scatter(recon3Pts2[::3, 0], recon3Pts2[::3, 1], recon3Pts2[::3, 2], s=0.1, c='r')
            # set_axes_equal(ax)
            # fig.show()


            for k in interestedType:
                intey, intex = np.where(semantic_seg == k)
                totIns = instance_seg[intey, intex]
                totUniqIns = np.unique(instance_seg[intey, intex])
                for p in totUniqIns:
                    valPosy = intey[totIns == p]
                    valPosx = intex[totIns == p]
                    valMask = np.zeros(imgShape)
                    valMask[valPosy, valPosx] = 1
                    depth2 = img2Depth[valPosy, valPosx]
                    recon3Pts = reconstruct3dPts(depth2, valPosx, valPosy, intrinsic2, extrinsic2)
                    recon2Pts3 = project3dPts(recon3Pts, intrinsic3, extrinsic3)
                    validMask, affM = diffICP(img3Depth, recon3Pts, intrinsic3, extrinsic3, svInd=globalCounts)

                    imgInt = np.concatenate((img2, img3), axis = 0)
                    fig = plt.figure()
                    plt.imshow(imgInt)
                    plt.xlim([0, imgInt.shape[1]])
                    plt.ylim([imgInt.shape[0], 0])
                    plt.scatter(valPosx[::4], valPosy[::4], s = 0.3, c = 'r')
                    if validMask is not None:
                        plt.scatter(recon2Pts3[validMask, 0], recon2Pts3[validMask, 1] + img2.shape[0], s=0.3, c = 'b')
                        tmp_bfErr = 0
                        tmp_afErr = 0
                        count_bfErr = 0
                        count_afErr = 0
                        for m in range(valPosy.shape[0]):
                            if gtDepthMap1[valPosy[m], valPosx[m]] != np.inf:
                                tmp_bfErr = tmp_bfErr + (depth2[m] - gtDepthMap1[valPosy[m], valPosx[m]]) ** 2
                                count_bfErr = count_bfErr + 1
                        if count_bfErr > 0:
                            tmp_bfErr = np.sqrt(tmp_bfErr / count_bfErr)

                        recon3Pts_transformed = (affM @ recon3Pts.T).T
                        recon2Pts3_new, depth_new = project3dPts(recon3Pts_transformed, intrinsic2, extrinsic2, isDepth = True)
                        depth_new[np.logical_not(validMask)] = depth2[np.logical_not(validMask)]
                        for m in range(valPosy.shape[0]):
                            if gtDepthMap1[valPosy[m], valPosx[m]] != np.inf:
                                tmp_afErr = tmp_afErr + (depth_new[m] - gtDepthMap1[valPosy[m], valPosx[m]]) ** 2
                                count_afErr = count_afErr + 1
                        if count_afErr > 0:
                            tmp_afErr = np.sqrt(tmp_afErr / count_afErr)
                        if count_bfErr > 0 and count_afErr > 0:
                            beforeError.append(tmp_bfErr)
                            afterError.append(tmp_afErr)
                    else:
                        plt.scatter(recon2Pts3[:, 0], recon2Pts3[:, 1] + img2.shape[0], s=0.3, c='b')
                    plt.savefig(os.path.join(intemSvPath, str(globalCounts) + '_2dMask'), dpi = 300)
                    plt.close()

                    globalCounts = globalCounts + 1

                    """
                    disparity2 = (STEREO_SCALE_FACTOR / depth2)
                    valPosy_2 = valPosy
                    valPosx_2 = valPosx - disparity2
                    valPosy_2_r = np.round(valPosy_2)
                    valPosx_2_r = np.round(valPosx_2)

                    imgInt = np.concatenate((img2, img3), axis = 0)
                    fig = plt.figure()
                    plt.imshow(imgInt)
                    plt.xlim([0, imgInt.shape[1]])
                    plt.ylim([imgInt.shape[0], 0])
                    plt.scatter(valPosx[::4], valPosy[::4], s = 0.3)
                    plt.scatter(recon2Pts3[::4, 0], recon2Pts3[::4, 1] + img2.shape[0], s=0.3)


                    carSelector = (valMask[testPos[:, 1], testPos[:, 0]] == 1) & (gtDepth < 17)
                    furtherSelect = depth2 < 17
                    fig = plt.figure()
                    ax = fig.add_subplot((111), projection='3d')
                    ax.scatter(recon3Pts[furtherSelect, 0], recon3Pts[furtherSelect, 1], recon3Pts[furtherSelect, 2], s=0.1, c='r')
                    ax.scatter(gtRecon3Pts[carSelector, 0], gtRecon3Pts[carSelector, 1], gtRecon3Pts[carSelector, 2], s=0.1, c='b')
                    set_axes_equal(ax)
                    fig.show()


                    fig = plt.figure()
                    plt.imshow(img3)
                    plt.xlim([0, img3.shape[1]])
                    plt.ylim([img3.shape[0], 0])
                    plt.scatter(recon2Pts3[::4, 0], recon2Pts3[::4, 1], s=0.3)


                    interX = np.arange(0, imgShape[1], 1)
                    interY = np.arange(0, imgShape[0], 1)
                    yy, xx = np.where(img3Depth != 1e10)
                    gridPos = np.stack((yy, xx), axis=1)
                    depth3interp = griddata(gridPos, img3Depth.flatten(), (recon2Pts3[:, 1], recon2Pts3[:, 0]), method='linear')
                    recon3Pts_r = reconstruct3dPts(depth3interp, recon2Pts3[:, 0], recon2Pts3[:, 1], intrinsic3, extrinsic3)
                    r_selector = depth3interp < 17
                    fig = plt.figure()
                    ax = fig.add_subplot((111), projection='3d')
                    ax.scatter(recon3Pts[furtherSelect, 0], recon3Pts[furtherSelect, 1], recon3Pts[furtherSelect, 2], s=0.1, c='r')
                    ax.scatter(recon3Pts_r[r_selector, 0], recon3Pts_r[r_selector, 1], recon3Pts_r[r_selector, 2], s=0.1, c='b')
                    ax.scatter(gtRecon3Pts[carSelector, 0], gtRecon3Pts[carSelector, 1], gtRecon3Pts[carSelector, 2],
                               s=0.1, c='g')
                    set_axes_equal(ax)
                    plt.legend(['Left predicted pts', 'Right predicted pts', 'Gt pts'])
                    fig.show()

                    interp_spline = RectBivariateSpline(interY, interX, img3Depth)
                    Z2 = interp_spline.ev(recon2Pts3[:, 1], recon2Pts3[:, 0])
                    np.mean(np.abs(Z2 - depth3interp))

                    discret_recon2Pts3 = np.round(recon2Pts3).astype((np.intc))
                    approxDepth = img3Depth[discret_recon2Pts3[:,1], discret_recon2Pts3[:,0]]
                    np.mean(np.abs(approxDepth - depth3interp))
                    """
            a = 1

beforeError_np = np.array(beforeError)
afterError_np = np.array(afterError)

boostM = np.mean(beforeError_np) - np.mean(afterError_np)
re = np.sum(np.abs(beforeError_np - afterError_np))
a = 1