from __future__ import absolute_import, division, print_function
import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from layers import *
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import cityscapesscripts.helpers.labels
from utils import *
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import *
from cityscapesscripts.helpers.labels import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

splits_dir = os.path.join(os.path.dirname(__file__), "splits")
STEREO_SCALE_FACTOR = 5.4

def tensor2rgb(tensor, ind):
    slice = (tensor[ind, :, :, :].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    # pil.fromarray(slice).show()
    return pil.fromarray(slice)

def tensor2semantic(tensor, ind):
    slice = tensor[ind, :, :, :]
    slice = F.softmax(slice, dim=0)
    slice = torch.argmax(slice, dim=0).cpu().numpy()
    # visualize_semantic(slice).show()
    return visualize_semantic(slice)

def tensor2disp(tensor, ind):
    # slice = tensor[]
    # plt.imsave(name_dest_im, disp_resized_np, cmap='magma', vmax=vmax)
    slice = tensor[ind, 0, :, :].cpu().numpy()
    vmax = np.percentile(slice, 95)
    slice = slice / vmax
    cm = plt.get_cmap('magma')
    slice = (cm(slice) * 255).astype(np.uint8)
    # pil.fromarray(slice).show()
    return pil.fromarray(slice)


class Tensor23dPts:
    def __init__(self):
        self.height = 1024
        self.width = 2048
        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        self.xx = xx.flatten()
        self.yy = yy.flatten()
        objType = 19
        self.colorMap = np.zeros((objType + 1, self.xx.shape[0], 3), dtype=np.uint8)
        for i in range(objType):
            if i == objType:
                k = 255
            else:
                k = i
            self.colorMap[i, :, :] = np.repeat(np.expand_dims(np.array(trainId2label[k].color), 0), self.xx.shape[0], 0)
        self.colorMap = self.colorMap.astype(np.float)
        self.colorMap = self.colorMap / 255


    def visualize3d(self, tensor, ind, intrinsic, extrinsic, gtmask = None, gtdepth = None, semanticMap = None):
        assert tensor.shape[1] == 1, "please input single channel depth map"
        self.height = 1024
        self.width = 2048
        tensor = F.interpolate(tensor, [self.height, self.width], mode="bilinear", align_corners=False)
        intrinsic = intrinsic.cpu().numpy()
        extrinsic = extrinsic.cpu().numpy()
        slice = tensor[ind, 0, :, :].cpu().numpy()


        # xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        # xx = xx.flatten()
        # yy = yy.flatten()
        depthFlat = slice.flatten()
        oneColumn = np.ones(self.height * self.width)
        pixelLoc = np.stack([self.xx * depthFlat, self.yy * depthFlat, depthFlat, oneColumn], axis=1)
        cam_coord = (np.linalg.inv(intrinsic) @ pixelLoc.T).T
        # mask = np.sum(np.square(cam_coord[:,0:3]), axis=1) < 100
        veh_coord = (np.linalg.inv(extrinsic) @ cam_coord.T).T
        # xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        # veh_coord = veh_coord[:,2].reshape(xx.shape)
        colors = None

        if gtmask is not None and gtdepth is not None:
            # mask = gtmask == 1
            # gtmask = gtmask.cpu().numpy()
            gtdepth = gtdepth.cpu().numpy()
            mask = gtdepth > 0
            mask = mask.flatten()
            depthFlat = gtdepth.flatten()
            oneColumn = np.ones(gtdepth.shape[0] * gtdepth.shape[1])
            pixelLoc = np.stack([self.xx[mask] * depthFlat[mask], self.yy[mask] * depthFlat[mask], depthFlat[mask], oneColumn[mask]], axis=1)
            cam_coord = (np.linalg.inv(intrinsic) @ pixelLoc.T).T
            veh_coord_2 = (np.linalg.inv(extrinsic) @ cam_coord.T).T

            veh_coord = veh_coord[mask, :]
        if semanticMap is not None:
            semanticMap = semanticMap.cpu().numpy()
            semanticMap = semanticMap.flatten()
            semanticMap[semanticMap == 255] = 19
            colors = self.colorMap[semanticMap, np.arange(self.xx.shape[0]), :]
            if mask is not None:
                colors = colors[mask, :]


        camPos = (np.linalg.inv(extrinsic) @ np.array([0,0,0,1]).T).T
        camDir = (np.linalg.inv(extrinsic) @ np.array([1,0,0,1]).T).T
        veh_coord[:, 0:3] = veh_coord[:, 0:3] - np.repeat(np.expand_dims(camPos, 0)[:,0:3], veh_coord.shape[0], 0)
        veh_coord_2[:, 0:3] = veh_coord_2[:, 0:3] - np.repeat(np.expand_dims(camPos, 0)[:,0:3], veh_coord_2.shape[0], 0)

        # camDir = camDir - camPos
        # radius = np.sqrt(np.sum(np.square(camPos[0:3])))
        # theta = np.arccos(camPos[2] / radius)
        # phi = np.arctan2(camPos[1], camPos[0])
        tmpImgName = 'tmp1.png'
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(elev=6., azim=170)
        ax.dist = 4
        # ax = fig.add_subplot(111, projection='3d')
        if colors is None:
            ax.scatter(veh_coord[0::100,0], veh_coord[0::100,1], veh_coord[0::100,2], s=0.1, c = 'b')
            ax.scatter(veh_coord_2[0::100, 0], veh_coord_2[0::100, 1], veh_coord_2[0::100, 2], s=0.1, c='r')
        else:
            # ax.scatter(veh_coord[0::50,0], veh_coord[0::50,1], veh_coord[0::50,2], s=0.5, c = colors[0::50, :])
            ax.scatter(veh_coord_2[0::50, 0], veh_coord_2[0::50, 1], veh_coord_2[0::50, 2], s=0.5, c = colors[0::50, :])
            # ax.plot_surface(xx, yy, veh_coord)
        ax.scatter(camPos[0], camPos[1], camPos[2], s=10, c='g')
        ax.set_zlim(-10, 10)
        plt.ylim([-10, 10])
        plt.xlim([10, 16])
        set_axes_equal(ax)
        fig.savefig(tmpImgName)
        plt.close(fig)

        img1 = pil.open(tmpImgName)
        # time.sleep(1)

        tmpImgName = 'tmp2.png'
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(elev=6., azim=170)
        ax.dist = 4
        # ax = fig.add_subplot(111, projection='3d')
        if colors is None:
            ax.scatter(veh_coord[0::100,0], veh_coord[0::100,1], veh_coord[0::100,2], s=0.1, c = 'b')
            ax.scatter(veh_coord_2[0::100, 0], veh_coord_2[0::100, 1], veh_coord_2[0::100, 2], s=0.1, c='r')
        else:
            ax.scatter(veh_coord[0::50,0], veh_coord[0::50,1], veh_coord[0::50,2], s=0.5, c = colors[0::50, :])
            # ax.scatter(veh_coord_2[0::50, 0], veh_coord_2[0::50, 1], veh_coord_2[0::50, 2], s=0.5, c = colors[0::50, :])
            # ax.plot_surface(xx, yy, veh_coord)
        ax.scatter(camPos[0], camPos[1], camPos[2], s=10, c='g')
        ax.set_zlim(-10, 10)
        plt.ylim([-10, 10])
        plt.xlim([10, 16])
        set_axes_equal(ax)
        fig.savefig(tmpImgName)
        plt.close(fig)
        # time.sleep(1)
        img2 = pil.open(tmpImgName)
        # time.sleep(1)
        img = Image.fromarray(np.concatenate([np.array(img1)[:,:,0:3], np.array(img2)[:,:,0:3]], axis=1))

        return img





def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.split, "val_files.txt"))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    if opt.dataset == 'cityscape':
        dataset = datasets.CITYSCAPERawDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False, tag=opt.dataset)
    elif opt.dataset == 'kitti':
        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False, tag=opt.dataset)
    else:
        raise ValueError("No predefined dataset")
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=True)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    if opt.switchMode == 'on':
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, isSwitch=True, isMulChannel=opt.isMulChannel)
    else:
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()




    ##--------------------Visualization parameter here----------------------------##
    sfx = torch.nn.Softmax(dim=1)
    mergeDisp = Merge_MultDisp(opt.scales, batchSize = opt.batch_size, isMulChannel = opt.isMulChannel)
    svRoot = '/media/shengjie/other/sceneUnderstanding/monodepth2/internalRe/figure_visual'
    index = 0
    isvisualize = True
    isHist = False
    height = 256
    width = 512
    tensor23dPts = Tensor23dPts()
    if isHist:
        rec = np.zeros((19,100))

    if opt.isMulChannel:
        app = os.path.join('mulDispOn', opt.model_name)
    else:
        app = os.path.join('mulDispOff', opt.model_name)

    dirpath = os.path.join(svRoot, app)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            input_color = inputs[("color", 0, 0)].cuda()
            features = encoder(input_color)
            outputs = dict()
            outputs.update(depth_decoder(features, computeSemantic=True, computeDepth=False))
            outputs.update(depth_decoder(features, computeSemantic=False, computeDepth=True))
            if isHist:
                mulDisp = outputs[('mul_disp', 0)]
                scaled_disp, mulDepth = disp_to_depth(mulDisp, 0.1, 100)
                mulDepth = mulDepth.cpu()
                for i in range(mulDisp.shape[1]):
                    rec[i,:] += torch.histc(mulDepth[:,i,:,:],bins=100,min=0,max=100).numpy()
            if isvisualize:
                mergeDisp(inputs, outputs, eval=True)

                dispMap = outputs[('disp', 0)]
                scaled_disp, mulDepth = disp_to_depth(dispMap, 0.1, 100)
                mulDepth = mulDepth * STEREO_SCALE_FACTOR

                fig_seman = tensor2semantic(outputs[('seman', 0)], ind=index)
                fig_rgb = tensor2rgb(inputs[('color', 0, 0)], ind=index)
                fig_disp = tensor2disp(outputs[('disp', 0)], ind=index)
                fig_3d = tensor23dPts.visualize3d(mulDepth, ind = index, intrinsic= inputs['cts_meta']['intrinsic'][index, :, :], extrinsic= inputs['cts_meta']['extrinsic'][index, :, :], gtmask=inputs['cts_meta']['mask'][index, :, :], gtdepth=inputs['cts_meta']['depthMap'][index, :, :], semanticMap=inputs['seman_gt_eval'][index, :, :])
                combined = [np.array(fig_disp)[:,:,0:3], np.array(fig_seman), np.array(fig_rgb)]
                combined = np.concatenate(combined, axis=1)
                fig = pil.fromarray(combined)
                fig.save(os.path.join(dirpath, str(idx) + '.png'))
                fig_3d.save(os.path.join(dirpath, str(idx) + '_fig3d.png'))
                # for k in range(10):
                #     fig_disp = tensor2disp(outputs[('disp', 0)], ind=k)
                #     fig_rgb = tensor2rgb(inputs[('color', 0, 0)], ind=k)
                #     combined = [np.array(fig_disp)[:, :, 0:3], np.array(fig_rgb)]
                #     combined = np.concatenate(combined, axis=1)
                #     fig = pil.fromarray(combined)
                #     fig.save(
                #         os.path.join('/media/shengjie/other/sceneUnderstanding/monodepth2/internalRe/MoredispOrg' + str(k) + '.png'))



                # fig_rgb.save(os.path.join(svRoot, app, 'rgb' + str(idx) + '.png'))
                # fig_seman.save(os.path.join(svRoot, app, 'semantic'+ str(idx) + '.png'))
                # fig_disp.save(os.path.join(svRoot, app, 'disp'+ str(idx) + '.png'))
                # a = inputs['seman_gt_eval']
                # scaled_disp, _ = disp_to_depth(outputs[('disp', 0)], 0.1, 100)
                print("%dth saved" % idx)
                if idx == 40:
                    a =1



    # If compute the histogram
    if isHist:
        svPath = '/media/shengjie/other/sceneUnderstanding/monodepth2/internalRe/mul_channel_depth'
        carId = 13
        prob = copy.deepcopy(rec)
        ind = np.arange(prob.shape[1] * 2)
        for i in range(prob.shape[0]):
            prob[i,:] = prob[i,:] / np.sum(prob[i,:])
        for i in range(prob.shape[0]):
            trainStr = trainId2label[i][0]
            fig, ax = plt.subplots()
            rects1 = ax.bar(ind[0::2], prob[carId, :], label='obj:car')
            rects2 = ax.bar(ind[1::2], prob[i, :], label='obj:' + trainStr)
            ax.set_ylabel('Meter in percentile')
            ax.set_xlabel('Meters')
            ax.set_title('Scale Changes between scale car and scale %s' % trainStr)
            ax.legend()
            plt.savefig(os.path.join(svPath, str(i)), dpi=200)
            plt.close(fig)



if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
