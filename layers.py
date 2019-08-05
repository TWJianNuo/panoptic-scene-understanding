# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from copy import deepcopy
from utils import *
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from cityscapesscripts.helpers.labels import trainId2label, id2label
from PIL import Image

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

class SwitchBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(SwitchBlock, self).__init__()

        self.conv_pos = Conv3x3(in_channels, out_channels)
        self.conv_neg = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x, switch_on = False):
        pos = self.nonlin(self.conv_pos(x))
        neg = self.nonlin(self.conv_neg(x))
        if switch_on:
            out = pos - neg
        else:
            out = pos + neg
        out = self.nonlin(out)
        # out = self.conv(x)
        return out
class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords))

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width))

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1))

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    repeat_time = disp.shape[1]
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True).repeat(1,repeat_time,1,1)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True).repeat(1,repeat_time,1,1)

    # grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    # grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

class Merge_MultDisp(nn.Module):
    def __init__(self, scales, semanType = 19, batchSize = 6, isMulChannel = False):
        # Merge multiple channel disparity to single channel according to semantic
        super(Merge_MultDisp, self).__init__()
        self.scales = scales
        self.semanType = semanType
        self.batchSize = batchSize
        self.sfx = nn.Softmax(dim=1).cuda()
        self.isMulChannel = isMulChannel
        # self.weights_time = 0

    def forward(self, inputs, outputs, eval = False):
        height = inputs[('color_aug', 0, 0)].shape[2]
        width = inputs[('color_aug', 0, 0)].shape[3]
        outputFormat = [self.batchSize, self.semanType + 1, height, width]

        if ('seman', 0) in outputs:
            for scale in self.scales:
                se_gt_name = ('seman', scale)
                seg = F.interpolate(outputs[se_gt_name], size=[height, width], mode='bilinear', align_corners=False)
                outputs[se_gt_name] = seg

        if self.isMulChannel:
            for scale in self.scales:
                disp_pred_name = ('mul_disp', scale)
                # disp = F.interpolate(outputs[disp_pred_name], [height, width], mode="bilinear", align_corners=False)
                disp = outputs[disp_pred_name]
                disp = torch.cat([disp, torch.mean(disp, dim=1, keepdim=True)], dim=1)
                outputs[disp_pred_name] = disp

            if 'seman_gt' in inputs and not eval:
                indexRef = deepcopy(inputs['seman_gt'])
                outputs['gtMask'] = indexRef != 255
                indexRef[indexRef == 255] = self.semanType
                disp_weights = torch.zeros(outputFormat).permute(0, 2, 3, 1).contiguous().view(-1, outputFormat[1]).cuda()
                indexRef = indexRef.permute(0, 2, 3, 1).contiguous().view(-1, 1)
                disp_weights[torch.arange(disp_weights.shape[0]), indexRef[:, 0]] = 1
                disp_weights = disp_weights.view(outputFormat[0], outputFormat[2], outputFormat[3],
                                                 outputFormat[1]).permute(0, 3, 1, 2)
                for scale in self.scales:
                    disp_weights = F.interpolate(disp_weights, [int(height / (2 ** scale)), int(width / (2 ** scale))],
                                                 mode="nearest")
                    outputs[('disp_weights', scale)] = disp_weights
            elif ('seman', 0) in outputs:
                # indexRef = torch.argmax(self.sfx(outputs[('seman', 0)]), dim=1, keepdim=True)
                disp_weights = torch.cat([self.sfx(outputs[('seman', 0)]),torch.zeros(outputFormat[0], outputFormat[2], outputFormat[3]).unsqueeze(1).cuda()], dim=1)
                for scale in self.scales:
                    disp_weights = F.interpolate(disp_weights, [int(height / (2 ** scale)), int(width / (2 ** scale))],
                                                 mode="bilinear", align_corners=False)
                    outputs[('disp_weights', scale)] = disp_weights

            # outputs['disp_weights'] = disp_weights
            for scale in self.scales:
                ref_name = ('mul_disp', scale)
                outputs[('disp', scale)] = torch.sum(outputs[ref_name] * outputs[('disp_weights', scale)], dim=1, keepdim=True)
        else:
            for scale in self.scales:
                ref_name = ('mul_disp', scale)
                if ref_name in outputs:
                    outputs[('disp', scale)] = outputs[ref_name]

class Compute_SemanticLoss(nn.Module):
    def __init__(self, classtype = 19, min_scale = 3):
        super(Compute_SemanticLoss, self).__init__()
        self.scales = list(range(4))[0:min_scale+1]
        # self.cen = nn.CrossEntropyLoss(reduction = 'none')
        self.cen = nn.CrossEntropyLoss(ignore_index = 255)
        self.classtype = classtype # default is cityscape setting 19
    def reorder(self, input, clssDim):
        return input.permute(2,3,1,0).contiguous().view(-1, clssDim)
    def forward(self, inputs, outputs, use_sep_semant_train = False):
        # height = inputs['seman_gt'].shape[2]
        # width = inputs['seman_gt'].shape[3]
        if not use_sep_semant_train:
            label = inputs['seman_gt']
        else:
            label = inputs['seperate_seman_gt']
        # Just for check
        # s = inputs['seman_gt'][0, 0, :, :].cpu().numpy()
        # visualize_semantic(s).show()
        # img = pil.fromarray((inputs[("color_aug", 0, 0)].permute(0,2,3,1)[0,:,:,:].cpu().numpy() * 255).astype(np.uint8))
        # img.show()
        # visualize_semantic(pred[0,:,:]).show()
        loss_toshow = dict()
        loss = 0
        for scale in self.scales:
            entry = ('seman', scale)
            scaled = outputs[entry]
            # scaled = F.interpolate(outputs[entry], size = [height, width], mode = 'bilinear')
            # rearranged = self.reorder(scaled, self.classtype)
            # cenl = self.cen(rearranged[mask[:,0], :], label[mask])
            cenl = self.cen(scaled, label.squeeze(1))
            loss_toshow["loss_seman/{}".format(scale)] = cenl
            loss = loss + cenl
            # just for check
            # m1 = rearranged[mask[:,0], :]
            # m2 = label[mask]
            # m3 = m1.gather(1, m2.view(-1,1))
            # loss_self = -log()
        loss = loss / len(self.scales)
        return loss, loss_toshow


class ComputeSurfaceNormal(nn.Module):
    def __init__(self, height, width, batch_size):
        super(ComputeSurfaceNormal, self).__init__()
        self.height = height
        self.width = width
        self.batch_size = batch_size
        # self.surnormType = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic sign', 'terrain']
        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        xx = xx.flatten().astype(np.float32)
        yy = yy.flatten().astype(np.float32)
        self.pix_coords = np.expand_dims(np.stack([xx, yy, np.ones(self.width * self.height).astype(np.float32)], axis=1), axis=0).repeat(self.batch_size, axis=0)
        self.pix_coords = torch.from_numpy(self.pix_coords).permute(0,2,1)
        self.ones = torch.ones(self.batch_size, 1, self.height * self.width)
        self.pix_coords = self.pix_coords.cuda()
        self.ones = self.ones.cuda()
        self.init_gradconv()

    def init_gradconv(self):
        weightsx = torch.Tensor([
                                [-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
                                [1., 2., 1.],
                                [0., 0., 0.],
                                [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)
        self.convx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, bias=False)
        self.convy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, bias=False)

        self.convx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.convy.weight = nn.Parameter(weightsy,requires_grad=False)


    def forward(self, depthMap, invcamK):
        depthMap = depthMap.view(self.batch_size, -1)
        cam_coords = self.pix_coords * torch.stack([depthMap, depthMap, depthMap], dim=1)
        cam_coords = torch.cat([cam_coords, self.ones], dim=1)
        veh_coords = torch.matmul(invcamK, cam_coords)
        veh_coords = veh_coords.view(self.batch_size, 4, self.height, self.width)
        veh_coords = veh_coords
        changex = torch.cat([self.convx(veh_coords[:, 0:1, :, :]), self.convx(veh_coords[:, 1:2, :, :]), self.convx(veh_coords[:, 2:3, :, :])], dim=1)
        changey = torch.cat([self.convy(veh_coords[:, 0:1, :, :]), self.convy(veh_coords[:, 1:2, :, :]), self.convy(veh_coords[:, 2:3, :, :])], dim=1)
        surfnorm = torch.cross(changex, changey, dim=1)
        surfnorm = F.normalize(surfnorm, dim = 1)
        return surfnorm

    def visualize(self, depthMap, invcamK, orgEstPts = None, gtEstPts = None, viewindex = 0):
        # First compute 3d points in vehicle coordinate system
        depthMap = depthMap.view(self.batch_size, -1)
        cam_coords = self.pix_coords * torch.stack([depthMap, depthMap, depthMap], dim=1)
        cam_coords = torch.cat([cam_coords, self.ones], dim=1)
        veh_coords = torch.matmul(invcamK, cam_coords)
        veh_coords = veh_coords.view(self.batch_size, 4, self.height, self.width)
        veh_coords = veh_coords
        changex = torch.cat([self.convx(veh_coords[:, 0:1, :, :]), self.convx(veh_coords[:, 1:2, :, :]), self.convx(veh_coords[:, 2:3, :, :])], dim=1)
        changey = torch.cat([self.convy(veh_coords[:, 0:1, :, :]), self.convy(veh_coords[:, 1:2, :, :]), self.convy(veh_coords[:, 2:3, :, :])], dim=1)
        surfnorm = torch.cross(changex, changey, dim=1)
        surfnorm = F.normalize(surfnorm, dim = 1)

        # check
        # ckInd = 22222
        # x = self.pix_coords[viewindex, 0, ckInd].long()
        # y = self.pix_coords[viewindex, 1, ckInd].long()
        # ptsck = veh_coords[viewindex, :, y, x]
        # projecteck = torch.inverse(invcamK)[viewindex, :, :].cpu().numpy() @ ptsck.cpu().numpy().T
        # x_ = projecteck[0] / projecteck[2]
        # y_ = projecteck[1] / projecteck[2] # (x, y) and (x_, y_) should be equal

        # colorize this figure
        surfacecolor = surfnorm / 2 + 0.5
        img = surfacecolor[viewindex, :, :, :].permute(1,2,0).detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        # pil.fromarray(img).show()

        # objreg = ObjRegularization()
        # varloss = varLoss(scale=1, windowsize=7, inchannel=surfnorm.shape[1])
        # var = varloss(surfnorm)
        # if orgEstPts is not None and gtEstPts is not None:
        #     testPts = veh_coords[viewindex, :, :].permute(1,0).cpu().numpy()
        #     fig = plt.figure()
        #     ax = Axes3D(fig)
        #     ax.view_init(elev=6., azim=170)
        #     ax.dist = 4
        #     ax.scatter(orgEstPts[0::100, 0], orgEstPts[0::100, 1], orgEstPts[0::100, 2], s=0.1, c='b')
        #     ax.scatter(testPts[0::10, 0], testPts[0::10, 1], testPts[0::10, 2], s=0.1, c='g')
        #     ax.scatter(gtEstPts[0::100, 0], gtEstPts[0::100, 1], gtEstPts[0::100, 2], s=0.1, c='r')
        #     ax.set_zlim(-10, 10)
        #     plt.ylim([-10, 10])
        #     plt.xlim([10, 16])
        #     set_axes_equal(ax)
        return pil.fromarray(img)

class varLoss(nn.Module):
    def __init__(self, windowsize = 3, inchannel = 3):
        super(varLoss, self).__init__()
        assert windowsize % 2 != 0, "pls input odd kernel size"
        # self.scale = scale
        self.windowsize = windowsize
        self.inchannel = inchannel
        self.initkernel()
    def initkernel(self):
        # kernel is for mean value calculation
        weights = torch.ones((self.inchannel, 1, self.windowsize, self.windowsize))
        weights = weights / (self.windowsize * self.windowsize)
        self.conv = nn.Conv2d(in_channels=self.inchannel, out_channels=self.inchannel, kernel_size=self.windowsize, padding=0, bias=False, groups=self.inchannel)
        self.conv.weight = nn.Parameter(weights, requires_grad=False)
        # self.conv.cuda()
    def forward(self, input):
        pad = int((self.windowsize - 1) / 2)
        scaled = input[:,:, pad : -pad, pad : -pad] - self.conv(input)
        loss = scaled * scaled
        # loss = torch.mean(scaled * scaled)
        # check
        # ckr = self.conv(input)
        # exptime = 100
        # for i in range(exptime):
        #     batchind = torch.randint(0, ckr.shape[0], [1]).long()[0]
        #     chanind = torch.randint(0, ckr.shape[1], [1]).long()[0]
        #     xind = torch.randint(pad, ckr.shape[2]-pad, [1]).long()[0]
        #     yind = torch.randint(pad, ckr.shape[3]-pad, [1]).long()[0]
        #     ra = torch.mean(input[batchind, chanind, xind -pad : xind + pad + 1, yind - pad : yind + pad + 1])
        #     assert torch.abs(ra - ckr[batchind, chanind, xind - pad, yind - pad]) < 1e-5, "wrong"
        return loss
    def visualize(self, input):
        pad = int((self.windowsize - 1) / 2)
        scaled = input[:,:, pad : -pad, pad : -pad] - self.conv(input)
        errMap = scaled * scaled
        # loss = torch.mean(scaled * scaled)
        return errMap

class SelfOccluMask(nn.Module):
    def __init__(self, maxDisp = 21):
        super(SelfOccluMask, self).__init__()
        self.maxDisp = maxDisp
        self.pad = self.maxDisp
        self.init_kernel()
        self.boostfac = 400
    def init_kernel(self):
        # maxDisp is the largest disparity considered
        # added with being compated pixels
        convweights = torch.zeros(self.maxDisp, 1, 3, self.maxDisp + 2)
        for i in range(0, self.maxDisp):
            convweights[i, 0, :, 0:2] = 1/6
            convweights[i, 0, :, i+2:i+3] = -1/3
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=self.maxDisp, kernel_size=(3,self.maxDisp + 2), stride=1, padding=self.pad, bias=False)
        self.conv.bias = nn.Parameter(torch.arange(self.maxDisp).type(torch.FloatTensor) + 1, requires_grad=False)
        self.conv.weight = nn.Parameter(convweights, requires_grad=False)

        # convweights_opp = torch.flip(convweights, dims=[1])
        # self.conv_opp = torch.nn.Conv2d(in_channels=1, out_channels=self.maxDisp, kernel_size=(3,self.maxDisp + 2), stride=1, padding=self.pad, bias=False)
        # self.conv_opp.bias = nn.Parameter(torch.arange(self.maxDisp).type(torch.FloatTensor), requires_grad=False)
        # self.conv_opp.weight = nn.Parameter(convweights_opp, requires_grad=False)

        # self.weightck = (torch.sum(torch.abs(self.conv.weight)) + torch.sum(torch.abs(self.conv.bias)))
        # self.gausconv = get_gaussian_kernel(channels = 1, padding = 1)
        # self.gausconv.cuda()

        self.detectWidth = 19  # 3 by 7 size kernel
        # self.detectWidth = 41
        self.detectHeight = 3
        convWeightsLeft = torch.zeros(1, 1, self.detectHeight, self.detectWidth)
        convWeightsRight = torch.zeros(1, 1, self.detectHeight, self.detectWidth)
        convWeightsLeft[0, 0, :, :int((self.detectWidth + 1) / 2)] = 1
        convWeightsRight[0, 0, :, int((self.detectWidth - 1) / 2):] = 1
        self.convLeft = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                        kernel_size=(self.detectHeight, self.detectWidth), stride=1,
                                        padding=[1, int((self.detectWidth - 1) / 2)], bias=False)
        self.convRight = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                         kernel_size=(self.detectHeight, self.detectWidth), stride=1,
                                         padding=[1, int((self.detectWidth - 1) / 2)], bias=False)
        self.convLeft.weight = nn.Parameter(convWeightsLeft, requires_grad=False)
        self.convRight.weight = nn.Parameter(convWeightsRight, requires_grad=False)
    def forward(self, dispmap, bsline):
        # dispmap = self.gausconv(dispmap)

        # assert torch.abs(self.weightck - (torch.sum(torch.abs(self.conv.weight)) + torch.sum(torch.abs(self.conv.bias)))) < 1e-2, "weights changed"
        with torch.no_grad():
            maskl = self.computeMask(dispmap, direction='l')
            maskr = self.computeMask(dispmap, direction='r')
            lind = bsline < 0
            rind = bsline > 0
            mask = torch.zeros_like(dispmap)
            mask[lind,:, :, :] = maskl[lind,:, :, :]
            mask[rind, :, :, :] = maskr[rind, :, :, :]

            # viewInd = 3
            # cm = plt.get_cmap('magma')
            # viewSSIMMask = mask[viewInd, 0, :, :].detach().cpu().numpy()
            # vmax = np.percentile(viewSSIMMask, 95)
            # viewSSIMMask = (cm(viewSSIMMask / vmax) * 255).astype(np.uint8)
            # pil.fromarray(viewSSIMMask).show()
            return mask
    def computeMask(self, dispmap, direction):
        with torch.no_grad():
            width = dispmap.shape[3]
            if direction == 'l':
                # output = self.conv(dispmap)
                # output = torch.min(output, dim=1, keepdim=True)[0]
                # output = output[:,:,self.pad-1:-(self.pad-1):,-width:]
                # mask = torch.tanh(-output * self.boostfac)
                # mask = mask.masked_fill(mask < 0.9, 0)
                output = self.conv(dispmap)
                output = torch.clamp(output, max=0)
                output = torch.min(output, dim=1, keepdim=True)[0]
                output = output[:, :, self.pad - 1:-(self.pad - 1):, -width:]
                output = torch.tanh(-output)
                mask = (output > 0.05).float()
                # mask = (mask > 0.05).float()
            elif direction == 'r':
                # dispmap_opp = torch.flip(dispmap, dims=[3])
                # output_opp = self.conv(dispmap_opp)
                # output_opp = torch.min(output_opp, dim=1, keepdim=True)[0]
                # output_opp = output_opp[:, :, self.pad - 1:-(self.pad - 1):, -width:]
                # mask = torch.tanh(-output_opp * self.boostfac)
                # mask = mask.masked_fill(mask < 0.9, 0)
                # mask = torch.flip(mask, dims=[3])
                dispmap_opp = torch.flip(dispmap, dims=[3])
                output_opp = self.conv(dispmap_opp)
                output_opp = torch.clamp(output_opp, max=0)
                output_opp = torch.min(output_opp, dim=1, keepdim=True)[0]
                output_opp = output_opp[:, :, self.pad - 1:-(self.pad - 1):, -width:]
                output_opp = torch.tanh(-output_opp)
                mask = (output_opp > 0.05).float()
                mask = torch.flip(mask, dims=[3])

                # viewInd = 0
                # cm = plt.get_cmap('magma')
                # viewSSIMMask = mask[viewInd, 0, :, :].detach().cpu().numpy()
                # vmax = np.percentile(viewSSIMMask, 95)
                # viewSSIMMask = (cm(viewSSIMMask / vmax) * 255).astype(np.uint8)
                # pil.fromarray(viewSSIMMask).show()


                # viewdisp = dispmap[viewInd, 0, :, :].detach().cpu().numpy()
                # vmax = np.percentile(viewdisp, 90)
                # viewdisp = (cm(viewdisp / vmax) * 255).astype(np.uint8)
                # pil.fromarray(viewdisp).show()
            return mask
    def visualize(self, dispmap, viewind = 0):
        cm = plt.get_cmap('magma')

        width = dispmap.shape[3]
        output = self.conv(dispmap)
        output = torch.clamp(output, max=0)
        # output = torch.abs(output + 1)
        output = torch.min(output, dim=1, keepdim=True)[0]
        output = output[:,:,self.pad-1:-(self.pad-1):,-width:]
        output = torch.tanh(-output)
        mask = output
        mask = mask > 0.1
        # a = output[0,0,:,:].detach().cpu().numpy()
        # mask = torch.tanh(-output) + 1
        # mask = torch.tanh(-output * self.boostfac)
        # mask = mask.masked_fill(mask < 0.9, 0)

        dispmap_opp = torch.flip(dispmap, dims=[3])
        output_opp = self.conv(dispmap_opp)
        output_opp = torch.min(output_opp, dim=1, keepdim=True)[0]
        output_opp = output_opp[:,:,self.pad-1:-(self.pad-1):,-width:]
        # output = output[:,:,pad:-pad, pad:-pad]
        mask_opp = torch.tanh(-output_opp * self.boostfac)
        # mask_opp = torch.clamp(mask_opp, min=0)
        # mask_opp = mask_opp.masked_fill(mask_opp < 0.8, 0)
        mask_opp = mask_opp.masked_fill(mask_opp < 0.9, 0)
        mask_opp = torch.flip(mask_opp, dims=[3])

        # mask = (mask + mask_opp) / 2
        # mask[mask < 0] = 0

        binmask = mask > 0.1
        viewbin = binmask[viewind, 0, :, :].detach().cpu().numpy()
        # pil.fromarray((viewbin * 255).astype(np.uint8)).show()
        #
        # binmask_opp = mask_opp > 0.3
        # viewbin = binmask_opp[viewind, 0, :, :].detach().cpu().numpy()
        # pil.fromarray((viewbin * 255).astype(np.uint8)).show()

        viewmask = mask[viewind, 0, :, :].detach().cpu().numpy()
        viewmask = (cm(viewmask)* 255).astype(np.uint8)
        # pil.fromarray(viewmask).show()

        viewmask_opp = mask_opp[viewind, 0, :, :].detach().cpu().numpy()
        viewmask_opp = (cm(viewmask_opp)* 255).astype(np.uint8)
        # pil.fromarray(viewmask_opp).show()

        # dispmap = dispmap * (1 - mask)
        viewdisp = dispmap[viewind, 0, :, :].detach().cpu().numpy()
        vmax = np.percentile(viewdisp, 90)
        viewdisp = (cm(viewdisp / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewdisp).show()

        dispmap_sup = dispmap * (1 - mask.float())
        view_dispmap_sup = dispmap_sup[viewind, 0, :, :].detach().cpu().numpy()
        vmax = np.percentile(view_dispmap_sup, 90)
        view_dispmap_sup = (cm(view_dispmap_sup / vmax) * 255).astype(np.uint8)
        # pil.fromarray(view_dispmap_sup).show()

        # viewdisp_opp = dispmap_opp[viewind, 0, :, :].detach().cpu().numpy()
        # vmax = np.percentile(viewdisp_opp, 90)
        # viewdisp_opp = (cm(viewdisp_opp / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewdisp_opp).show()
        return pil.fromarray(viewmask), pil.fromarray(viewdisp)

    def betterLeftRigthOccluMask(self, occluMask, foregroundMask, direction):
        with torch.no_grad():
            if direction == 'l':
                mask = occluMask.clone()
                coSelected = (self.convLeft(occluMask) > 0) * (self.convRight(foregroundMask) > 0)
                mask[coSelected] = 1
                mask = mask * (1 - foregroundMask)
            elif direction == 'r':
                mask = occluMask.clone()
                coSelected = (self.convRight(occluMask) > 0) * (self.convLeft(foregroundMask) > 0)
                mask[coSelected] = 1
                mask = mask * (1 - foregroundMask)
            return mask
    def betterSelfOccluMask(self, occluMask, foregroundMask, bsline, dispPred = None):
        with torch.no_grad():
            maskl = self.betterLeftRigthOccluMask(occluMask, foregroundMask, direction='l')
            maskr = self.betterLeftRigthOccluMask(occluMask, foregroundMask, direction='r')
            lind = bsline < 0
            rind = bsline > 0
            mask = torch.zeros_like(occluMask)
            mask[lind,:, :, :] = maskl[lind,:, :, :]
            mask[rind, :, :, :] = maskr[rind, :, :, :]

            # viewInd = 0
            # cm = plt.get_cmap('magma')
            # viewOccMask = mask[viewInd, 0, :, :].detach().cpu().numpy()
            # viewForegroundMask = foregroundMask[viewInd, 0, :, :].detach().cpu().numpy()
            #
            # viewMaskAndGt = np.zeros([dispPred.shape[2], dispPred.shape[3], 3])
            # viewMaskAndGt[:, :, 0] = viewOccMask
            # viewMaskAndGt[:, :, 1] = viewForegroundMask
            # viewMaskAndGt = (viewMaskAndGt * 255).astype(np.uint8)
            #
            # viewDispPred = dispPred[viewInd, 0, :, :].detach().cpu().numpy()
            # vmax = np.percentile(viewDispPred, 95)
            # viewDispPred = (cm(viewDispPred / vmax)* 255).astype(np.uint8)
            # toshow = np.concatenate([viewMaskAndGt, viewDispPred[:,:,0:3]], axis=0)
            # pil.fromarray(toshow).show()

            # cm = plt.get_cmap('magma')
            # viewInd = 0
            # viewOccMask = occluMask[viewInd, 0, :, :].detach().cpu().numpy()
            # viewOccMask = (cm(viewOccMask)* 255).astype(np.uint8)
            # pil.fromarray(viewOccMask).show()
            #
            # viewForegroundMask = foregroundMask[viewInd, 0, :, :].detach().cpu().numpy()
            # viewForegroundMask = (cm(viewForegroundMask)* 255).astype(np.uint8)
            # pil.fromarray(viewForegroundMask).show()
            #
            # viewDispPred = dispPred[viewInd, 0, :, :].detach().cpu().numpy()
            # vmax = np.percentile(viewDispPred, 95)
            # viewDispPred = (cm(viewDispPred / vmax)* 255).astype(np.uint8)
            # pil.fromarray(viewDispPred).show()
            #
            # occluMask = occluMask * (1 - foregroundMask)
            # viewOccMaskNew = occluMask[viewInd, 0, :, :].detach().cpu().numpy()
            # viewOccMaskNew = (cm(viewOccMaskNew)* 255).astype(np.uint8)
            # pil.fromarray(viewOccMaskNew).show()
            #
            # self.detectWidth = 19 # 3 by 7 size kernel
            # self.detectHeight = 3
            # convWeightsLeft = torch.zeros(1, 1, self.detectHeight, self.detectWidth)
            # convWeightsRight = torch.zeros(1, 1, self.detectHeight, self.detectWidth)
            # convWeightsLeft[0, 0, :, :int((self.detectWidth + 1)/2)] = 1
            # convWeightsRight[0, 0, :, int((self.detectWidth - 1)/2):] = 1
            # self.convLeft = torch.nn.Conv2d(in_channels=1, out_channels=1,
            #                                 kernel_size=(self.detectHeight, self.detectWidth), stride=1,
            #                                 padding=[1, int((self.detectWidth - 1) / 2)], bias=False)
            # self.convRight = torch.nn.Conv2d(in_channels=1, out_channels=1,
            #                                 kernel_size=(self.detectHeight, self.detectWidth), stride=1,
            #                                 padding=[1, int((self.detectWidth - 1) / 2)], bias=False)
            # self.convLeft.weight = nn.Parameter(convWeightsLeft, requires_grad=False)
            # self.convRight.weight = nn.Parameter(convWeightsRight, requires_grad=False)
            # self.cuda()
            # coSelected = (self.convLeft(occluMask) > 0) * (self.convRight(foregroundMask) > 0)
            # occluMask[coSelected] = 1
            # occluMask = occluMask * (1 - foregroundMask)
            # viewOccMaskNew = occluMask[viewInd, 0, :, :].detach().cpu().numpy()
            # viewOccMaskNew = (cm(viewOccMaskNew)* 255).astype(np.uint8)
            # pil.fromarray(viewOccMaskNew).show()



            return mask

class ComputeDispUpLoss(nn.Module):
    def __init__(self):
        super(ComputeDispUpLoss, self).__init__()
        self.init_kernel()
    def init_kernel(self):
        # noinspection PyArgumentList
        weights = torch.Tensor([
                                [-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.conv.weight = nn.Parameter(weights,requires_grad=False)
        # self.conv.cuda()
    def visualize(self, dispImage, viewindex = 0):
        cm = plt.get_cmap('magma')

        xdispgrad = self.conv(dispImage)
        xdispgrad = torch.clamp(xdispgrad, min=0)
        # xdispgrad[xdispgrad < 0] = 0

        viewmask = xdispgrad[viewindex, 0, :, :].detach().cpu().numpy()
        vmax = np.percentile(viewmask, 98)
        viewmask = (cm(viewmask / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewmask).show()

        viewdisp = dispImage[viewindex, 0, :, :].detach().cpu().numpy()
        vmax = np.percentile(viewdisp, 90)
        viewdisp = (cm(viewdisp / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewdisp).show()
        return pil.fromarray(viewmask)

    def forward(self, dispImage):
        xdispgrad = self.conv(dispImage)
        xdispgrad = torch.clamp(xdispgrad, min=0)
        # graduploss = torch.mean(xdispgrad)
        return xdispgrad

class ComputeSmoothLoss(nn.Module):
    def __init__(self):
        super(ComputeSmoothLoss, self).__init__()
        self.init_kernel()
    def init_kernel(self):
        weightsx = torch.Tensor([
                                [-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
                                [ 1.,  2.,  1.],
                                [ 0.,  0.,  0.],
                                [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)

        self.convDispx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convDispy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convRgbx = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False, groups=3)
        self.convRgby = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False, groups=3)

        self.convDispx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.convDispy.weight = nn.Parameter(weightsy,requires_grad=False)
        self.convRgbx.weight = nn.Parameter(weightsx.repeat(3,1,1,1), requires_grad=False)
        self.convRgby.weight = nn.Parameter(weightsy.repeat(3,1,1,1), requires_grad=False)

        gaussWeights = get_gaussian_kernel_weights()
        self.gaussConv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False, groups=3)
        self.gaussConv.weight = nn.Parameter(gaussWeights.repeat(3,1,1,1), requires_grad=False)

        self.convDispx.cuda()
        self.convDispy.cuda()
        self.convRgbx.cuda()
        self.convRgby.cuda()
        self.gaussConv.cuda()
    def visualize(self, rgb, disp, viewindex = 0):
        rgb = self.gaussConv(rgb)

        rgbx = self.convRgbx(rgb)
        rgbx = torch.mean(torch.abs(rgbx), dim=1, keepdim=True)
        rgby = self.convRgby(rgb)
        rgby = torch.mean(torch.abs(rgby), dim=1, keepdim=True)

        dispx = torch.abs(self.convDispx(disp))
        dispy = torch.abs(self.convDispy(disp))

        dispgrad_before = dispx + dispy

        dispx *= torch.exp(-rgbx)
        dispy *= torch.exp(-rgby)

        dispgrad_after = dispx + dispy

        viewRgb = rgb[viewindex, :, :, :].permute(1,2,0).cpu().numpy()
        viewRgb = (viewRgb * 255).astype(np.uint8)
        # pil.fromarray(viewRgb).show()

        rgbGrad = rgbx + rgby
        viewRgbGrad = rgbGrad[viewindex, 0, :, :].cpu().numpy()
        vmax = np.percentile(viewRgbGrad, 98)
        viewRgbGrad[viewRgbGrad > vmax] = vmax
        viewRgbGrad = viewRgbGrad / vmax
        viewRgbGrad = (viewRgbGrad * 255).astype(np.uint8)
        # pil.fromarray(viewRgbGrad).show()
        # viewRgbGrad = (cm(viewRgbGrad / vmax)* 255).astype(np.uint8)


        viewDispBefore = dispgrad_before[viewindex, 0, :, :].cpu().numpy()
        vmax = np.percentile(viewDispBefore, 95)
        viewDispBefore[viewDispBefore > vmax] = vmax
        viewDispBefore = viewDispBefore / vmax
        viewDispBefore = (viewDispBefore * 255).astype(np.uint8)
        # pil.fromarray(viewDispBefore).show()

        viewDispAfter = dispgrad_after[viewindex, 0, :, :].cpu().numpy()
        vmax = np.percentile(viewDispAfter, 95)
        viewDispAfter[viewDispAfter > vmax] = vmax
        viewDispAfter = viewDispAfter / vmax
        viewDispAfter = (viewDispAfter * 255).astype(np.uint8)
        # pil.fromarray(viewDispAfter).show()


        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(rgb[:, :, :, :-1] - rgb[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(rgb[:, :, :-1, :] - rgb[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)
        grad_goard = grad_disp_x[:,:,:-1,:] + grad_disp_y[:,:,:,:-1]
        view_gradgoard = grad_goard[viewindex, 0, :, :].cpu().numpy()
        vmax = np.percentile(view_gradgoard, 95)
        view_gradgoard[view_gradgoard > vmax] = vmax
        view_gradgoard = view_gradgoard / vmax
        view_gradgoard = (view_gradgoard * 255).astype(np.uint8)
        view_gradgoard = pil.fromarray(view_gradgoard).resize((viewDispAfter.shape[1], viewDispAfter.shape[0]))
        view_gradgoard = np.array(view_gradgoard)
        # pil.fromarray(view_gradgoard).show()

        combined = np.concatenate([viewRgb, np.repeat(np.expand_dims(viewRgbGrad, axis=2), 3, axis=2), np.repeat(np.expand_dims(viewDispBefore, axis=2), 3, axis=2), np.repeat(np.expand_dims(viewDispAfter, axis=2), 3, axis=2), np.repeat(np.expand_dims(view_gradgoard, axis=2), 3, axis=2)])
        # pil.fromarray(combined).show()
        return pil.fromarray(combined)


class ObjRegularization(nn.Module):
    def __init__(self, windowsize = 5):
        super(ObjRegularization, self).__init__()
        self.skyDepth = 80
        self.windowsize = windowsize
        self.varloss = varLoss(windowsize = self.windowsize, inchannel = 3)
        self.largePad = nn.MaxPool2d(kernel_size=11, padding=5, stride = 1)
        self.smallPad = nn.MaxPool2d(kernel_size=5, padding=2, stride = 1)
        # self.bdDir = torch.Tensor([0, 0, 1]).cuda()
        # self.roadDir = torch.Tensor([0, 0, 1]).cuda()
    def regularizeSky(self, depthMap, skyMask):
        regLoss = 0
        skyMask = 1 - self.largePad((1 - skyMask).float()).byte()
        if torch.sum(skyMask) > 100:
            regLoss = torch.mean((depthMap[skyMask] - self.skyDepth) ** 2)
        # Sky should be a definite longest value
        return regLoss
    def regularizeBuildingRoad(self, surfNorm, bdMask, rdMask):
        # Suppose surfNorm is bts x 3 x H x W
        # height = bdMask.shape[2]
        # width = bdMask.shape[3]

        # bdErrMap = torch.zeros(bdMask.shape).cuda()
        # rdErrMap = torch.zeros(bdMask.shape).cuda()

        bdMask = 1 - self.largePad((1 - bdMask).float()).byte()
        rdMask = 1 - self.largePad((1 - rdMask).float()).byte()

        bdRegLoss = 0
        rdRegLoss = 0
        for i in range(surfNorm.shape[0]):
            tmpSurNorm = surfNorm[i, :, :, :].contiguous().view(3, -1).permute(1, 0)
            # crossProduct = torch.sum(tmpSurNorm * self.bdDir.repeat([tmpSurNorm.shape[0], 1]), dim=1, keepdim=True)

            tmpBdMask = bdMask[i, :, :, :].contiguous().view(1, -1).permute(1, 0).squeeze(1)
            if torch.sum(tmpBdMask) > 50:
                bdRegLoss += torch.var(torch.abs(tmpSurNorm[tmpBdMask, 2]))


            tmpRdMask = rdMask[i, :, :, :].contiguous().view(1, -1).permute(1, 0).squeeze(1)
            if torch.sum(tmpRdMask) > 50:
                partialRoadVec = torch.abs(tmpSurNorm[tmpRdMask, 2])
                rdRegLoss += torch.mean((partialRoadVec - torch.mean(partialRoadVec)) ** 2 * torch.exp(partialRoadVec - 1))
            # Check
            # torch.sum(torch.abs(tmpSurNorm[tmpBdMask, :]), dim=0)
            # a = torch.abs(tmpSurNorm[tmpBdMask, :]).detach().cpu().numpy()

            # tmpBdErr = torch.zeros(tmpBdMask.shape).cuda()
            # tmpBdErr[tmpBdMask] = (torch.abs(tmpSurNorm[tmpBdMask, 2]) - torch.mean(
            #     torch.abs(tmpSurNorm[tmpBdMask, 2]))) ** 2
            # bdErrMap[i, :, :, :] = tmpBdErr.view(1, height, width)

            # tmpRdErr = torch.zeros(tmpRdMask.shape).cuda()
            # tmpRdErr[tmpRdMask] = (partialRoadVec - torch.mean(partialRoadVec)) ** 2 * torch.exp(partialRoadVec - 1)
            # rdErrMap[i, :, :, :] = tmpRdErr.view(1, height, width)

        bdRegLoss = bdRegLoss / surfNorm.shape[0]
        rdRegLoss = rdRegLoss / surfNorm.shape[0]
        return bdRegLoss, rdRegLoss

    def regularizePoleSign(self, surfNorm, mask):
        mask = 1 - self.smallPad((1 - mask).float()).byte()
        surfNormLoss = 0
        if torch.sum(mask) > 100:
            surfNormLoss = self.varloss(surfNorm)
            surfNormLoss = torch.mean(surfNormLoss, dim=1, keepdim=True)
            surfNormLoss = torch.mean(surfNormLoss[mask])
        return surfNormLoss


    def visualize_regularizeSky(self, depthMap, skyMask, viewInd = 0):
        # Pad sky boundary
        skyMask_shrink = 1 - self.largePad((1 - skyMask).float()).byte()

        regLoss = torch.mean((depthMap[skyMask] - self.skyDepth) ** 2)

        skyErrRec = torch.zeros_like(depthMap)
        skyErrRec[skyMask] = (depthMap[skyMask] - self.skyDepth) ** 2


        viewMask = (skyMask[viewInd, 0, :, :].cpu().numpy() * 255).astype(np.uint8)
        viewShrinkMask = (skyMask_shrink[viewInd, 0, :, :].cpu().numpy() * 255).astype(np.uint8)
        viewShrinkBorder = ((skyMask - skyMask_shrink)[viewInd, 0, :, :].cpu().numpy() * 255).astype(np.uint8)
        viewMaskTot = np.concatenate([viewMask, viewShrinkMask, viewShrinkBorder], axis=0)
        # pil.fromarray(viewMaskTot).show()

        cm = plt.get_cmap('magma')
        viewdisp = 1 / depthMap[viewInd, 0, :, :].detach().cpu().numpy()
        vmax = np.percentile(viewdisp, 90)
        viewdisp = (cm(viewdisp / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewdisp).show()

        viewErr = skyErrRec[viewInd, 0, :, :].detach().cpu().numpy()
        vmax = viewErr.max() * 0.9
        viewErr = (cm(viewErr / vmax) * 255).astype(np.uint8)
        # pil.fromarray(viewErr).show()



        return pil.fromarray(viewErr)
    def visualize_regularizeBuildingRoad(self, surfNorm, bdMask, rdMask, dispMap, viewInd = 0):
        # Suppose surfNorm is bts x 3 x H x W

        bdMask_shrink = 1 - self.largePad((1 - bdMask).float()).byte()
        rdMask_shrink = 1 - self.largePad((1 - rdMask).float()).byte()


        height = bdMask.shape[2]
        width = bdMask.shape[3]

        bdErrMap = torch.zeros(bdMask.shape).cuda()
        rdErrMap = torch.zeros(bdMask.shape).cuda()

        bdRegLoss = 0
        rdRegLoss = 0
        for i in range(surfNorm.shape[0]):
            tmpSurNorm = surfNorm[i, :, :, :].contiguous().view(3, -1).permute(1,0)
            tmpBdMask = bdMask[i, :, :, :].contiguous().view(1, -1).permute(1,0).squeeze(1)
            tmpRdMask = rdMask[i, :, :, :].contiguous().view(1, -1).permute(1,0).squeeze(1)
            # crossProduct = torch.sum(tmpSurNorm * self.bdDir.repeat([tmpSurNorm.shape[0], 1]), dim=1, keepdim=True)


            bdRegLoss += torch.var(torch.abs(tmpSurNorm[tmpBdMask, 2]))
            partialRoadVec = torch.abs(tmpSurNorm[tmpRdMask, 2])
            rdRegLoss += torch.mean((partialRoadVec - torch.mean(partialRoadVec))**2 * torch.exp(partialRoadVec - 1))
            # Check
            # torch.sum(torch.abs(tmpSurNorm[tmpBdMask, :]), dim=0)
            # a = torch.abs(tmpSurNorm[tmpBdMask, :]).detach().cpu().numpy()

            tmpBdErr = torch.zeros(tmpBdMask.shape).cuda()
            tmpBdErr[tmpBdMask] = (torch.abs(tmpSurNorm[tmpBdMask, 2]) - torch.mean(torch.abs(tmpSurNorm[tmpBdMask, 2])))**2
            bdErrMap[i, :, :, :] = tmpBdErr.view(1, height, width)

            tmpRdErr = torch.zeros(tmpRdMask.shape).cuda()
            tmpRdErr[tmpRdMask] = (partialRoadVec - torch.mean(partialRoadVec))**2 * torch.exp(partialRoadVec - 1)
            rdErrMap[i, :, :, :] = tmpRdErr.view(1, height, width)

        bdRegLoss = bdRegLoss / surfNorm.shape[0]
        rdRegLoss = rdRegLoss / surfNorm.shape[0]

        surfacecolor = surfNorm / 2 + 0.5
        surfimg = surfacecolor[viewInd, :, :, :].detach().permute(1,2,0).cpu().numpy()
        surfimg = (surfimg * 255).astype(np.uint8)
        # pil.fromarray(surfimg).show()

        cm = plt.get_cmap('magma')
        viewdisp = dispMap[viewInd, 0, :, :].detach().cpu().numpy()
        vmax = np.percentile(viewdisp, 90)
        viewdisp = (cm(viewdisp / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewdisp).show()

        viewBdMask = bdMask[viewInd, :, :, :].squeeze(0).detach().cpu().numpy()
        viewBdMask = (viewBdMask * 255).astype(np.uint8)
        viewbdMask_shrink = (bdMask_shrink[viewInd, 0, :, :].cpu().numpy() * 255).astype(np.uint8)
        viewbdMask_ShrinkBorder = (viewBdMask - viewbdMask_shrink)
        viewbdMaskMaskTot = np.concatenate([viewBdMask, viewbdMask_shrink, viewbdMask_ShrinkBorder], axis=0)
        # pil.fromarray(viewbdMaskMaskTot).show()


        viewRdMask = rdMask[viewInd, :, :, :].squeeze(0).detach().cpu().numpy()
        viewRdMask = (viewRdMask * 255).astype(np.uint8)
        viewRdMask_shrink = (rdMask_shrink[viewInd, 0, :, :].cpu().numpy() * 255).astype(np.uint8)
        viewRdMask_ShrinkBorder = (viewRdMask - viewRdMask_shrink)
        viewRdMaskMaskTot = np.concatenate([viewRdMask, viewRdMask_shrink, viewRdMask_ShrinkBorder], axis=0)
        # pil.fromarray(viewRdMaskMaskTot).show()

        viewBdErr = bdErrMap[viewInd, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = 0.15
        viewBdErr = viewBdErr / vmax
        viewBdErr = (cm(viewBdErr / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewBdErr).show()

        viewRdErr = rdErrMap[viewInd, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = 0.15
        viewRdErr = viewRdErr / vmax
        viewRdErr = (cm(viewRdErr / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewRdErr).show()

        return pil.fromarray(viewBdErr), pil.fromarray(viewRdErr)

    def visualize_regularizePoleSign(self, surfNorm, mask, dispMap, viewInd = 0):
        mask_shrink = 1 - self.smallPad((1 - mask).float()).byte()

        surfNormLoss = self.varloss(surfNorm)
        surfNormVar = self.varloss.visualize(surfNorm)
        surfNormVar = torch.mean(surfNormVar, dim=1, keepdim=True)
        surfNormVar = surfNormVar.masked_fill((1-mask), 0)



        surfacecolor = surfNorm / 2 + 0.5
        surfimg = surfacecolor[viewInd, :, :, :].detach().permute(1,2,0).cpu().numpy()
        surfimg = (surfimg * 255).astype(np.uint8)
        # pil.fromarray(surfimg).show()

        cm = plt.get_cmap('magma')
        viewdisp = dispMap[viewInd, 0, :, :].detach().cpu().numpy()
        vmax = np.percentile(viewdisp, 90)
        viewdisp = (cm(viewdisp / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewdisp).show()

        viewSurVar = surfNormVar[viewInd, 0, :, :].detach().cpu().numpy()
        vmax = 0.15
        viewSurVar = (cm(viewSurVar / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewSurVar).show()

        viewMask = (mask[viewInd, 0, :, :].cpu().numpy() * 255).astype(np.uint8)
        viewShrinkMask = (mask_shrink[viewInd, 0, :, :].cpu().numpy() * 255).astype(np.uint8)
        viewShrinkBorder = (viewMask - viewShrinkMask)
        viewMaskTot = np.concatenate([viewMask, viewShrinkMask, viewShrinkBorder], axis=0)
        # pil.fromarray(viewMaskTot).show()

        return pil.fromarray(viewSurVar)


class BorderRegression(nn.Module):
    def __init__(self, toleWid = 3, senseWid = 7):
        super(BorderRegression, self).__init__()
        self.toleWid = toleWid
        self.senseWid = senseWid
        self.init_conv()
    def init_conv(self):
        weightsx = torch.Tensor([
                                [-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
                                [1., 2., 1.],
                                [0., 0., 0.],
                                [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)
        self.disp_convx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.disp_convy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.disp_convx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.disp_convy.weight = nn.Parameter(weightsy,requires_grad=False)


        self.seman_convx = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1, bias=False, groups=2)
        self.seman_convy = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1, bias=False, groups=2)

        self.seman_convx.weight = nn.Parameter(weightsx.repeat([2,1,1,1]),requires_grad=False)
        self.seman_convy.weight = nn.Parameter(weightsy.repeat([2,1,1,1]),requires_grad=False)


        toleWeight = torch.ones([3, self.toleWid]).unsqueeze(0).unsqueeze(0) / self.toleWid
        senseWeight = torch.ones([3, self.senseWid]).unsqueeze(0).unsqueeze(0) / self.senseWid

        self.toleKernel = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[3, self.toleWid], padding=[1, int((self.toleWid - 1) / 2)], bias=False)
        self.senseKernel = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[3, self.senseWid], padding=[1, int((self.senseWid - 1) / 2)], bias=False)

        self.toleKernel.weight = nn.Parameter(toleWeight, requires_grad=False)
        self.senseKernel.weight = nn.Parameter(senseWeight, requires_grad=False)


        self.visualizeTole = nn.MaxPool2d(kernel_size=[3, int((self.toleWid + 1) / 2)], stride=1, padding=[1, int((self.toleWid + 1) / 2 / 2)])
        self.visualizeSense = nn.MaxPool2d(kernel_size=[3, int((self.senseWid + 1) / 2)], stride=1, padding=[1, int((self.senseWid + 1) / 2 / 2)])
        self.expand = torch.nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

    def borderRegression(self, disp, combinedMask, suppresMask = None, ssimMask = None):
        dispGrad = torch.abs(self.disp_convx(disp)) + torch.abs(self.disp_convy(disp))
        maskGrad = torch.abs(self.seman_convx(combinedMask)) # + torch.abs(self.seman_convy(combinedMask))
        maskGrad = torch.sum(maskGrad, dim=1, keepdim=True)
        if suppresMask is not None:
            maskSuppress = torch.abs(self.disp_convx(suppresMask)) # + torch.abs(self.disp_convy(suppresMask))
            maskGrad = maskGrad - maskSuppress
        maskGrad = torch.clamp(maskGrad, min = 0, max = 1)
        ssimMask_expand = self.expand(ssimMask)
        maskGrad_expand = self.expand(maskGrad)
        ssimMask_expand = ssimMask_expand * maskGrad_expand

        toleLevel = self.toleKernel(dispGrad)
        senseLevel = self.senseKernel(dispGrad)
        # err = torch.tanh(torch.clamp(senseLevel - toleLevel, min=0)) + 1
        # diff = senseLevel - toleLevel
        # err = torch.tanh(torch.clamp(diff, min=0))
        # err = err * maskGrad
        # loss1 = torch.mean(err[err > 0.01])
        err = senseLevel * self.senseWid - toleLevel * self.toleWid
        err = err * maskGrad
        loss = torch.mean(err[err > 0.1])

        viewind = 0
        cm = plt.get_cmap('magma')
        viewssimMask_expand = ssimMask_expand[viewind, 0, :, :].detach().cpu().numpy()
        vmax = np.percentile(viewssimMask_expand, 95)
        viewssimMask_expand = (cm(viewssimMask_expand / vmax) * 255).astype(np.uint8)
        pil.fromarray(viewssimMask_expand).show()

        return loss

        # Second stage loss
        # outerArea = senseLevel * self.senseWid - toleLevel * self.toleWid
        # outerArea = outerArea * (maskGrad * (err < 0.01).float())
        # loss2 = 0
        # if torch.sum(outerArea > 0.01) > 1000:
        #     loss2 = loss2 + torch.mean(outerArea[outerArea > 0.01])
        # else:
        #     loss2 = 0
        # return loss1, loss2

    def visualize_computeBorder(self, disp, combinedMask, suppresMask = None, viewIndex = 0):
        # Sample at the boundary between foreground obj and background obj
        # combinedMask channel: [foregroundMask, backgroundMask]

        dispGrad = torch.abs(self.disp_convx(disp)) + torch.abs(self.disp_convy(disp))
        maskGrad = torch.abs(self.seman_convx(combinedMask)) # + torch.abs(self.seman_convy(combinedMask))
        maskGrad = torch.sum(maskGrad, dim=1, keepdim=True)
        if suppresMask is not None:
            maskSuppress = torch.abs(self.disp_convx(suppresMask)) # + torch.abs(self.disp_convy(suppresMask))
            maskGrad = maskGrad - maskSuppress
            maskGrad = torch.clamp(maskGrad, min=0)


        viewForeGroundMask = combinedMask[viewIndex, 0, :, :].detach().cpu().numpy()
        viewForeGroundMask = (viewForeGroundMask * 255).astype(np.uint8)
        # pil.fromarray(viewForeGroundMask).show()

        viewBackGroundMask = combinedMask[viewIndex, 1, :, :].detach().cpu().numpy()
        viewBackGroundMask = (viewBackGroundMask * 255).astype(np.uint8)
        # pil.fromarray(viewBackGroundMask).show()

        cm = plt.get_cmap('magma')
        viewDispGrad = dispGrad[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewDispGrad, 95)
        viewDispGrad = (cm(viewDispGrad / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewDispGrad).show()

        viewDisp = disp[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewDisp, 90)
        viewDisp = (cm(viewDisp / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewDisp).show()

        viewMaskGrad = maskGrad[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = 0.1
        viewMaskGrad = (cm(viewMaskGrad / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewMaskGrad).show()

        viewBackGroundMask_red = np.stack([viewBackGroundMask, np.zeros_like(viewBackGroundMask), np.zeros_like(viewBackGroundMask)], axis=2)
        viewMaskGradObj = (viewBackGroundMask_red * 0.7 + viewMaskGrad[:,:,0:3] * 0.3).astype(np.uint8)
        # pil.fromarray(viewMaskGradObj).resize([viewMaskGradObj.shape[1] * 2, viewMaskGradObj.shape[0] * 2]).show()

        cmk = plt.get_cmap('hot')
        maskGrad_tole = self.visualizeTole(maskGrad)
        maskGrad_tole = torch.clamp(maskGrad_tole, min=0, max=1)
        viewMaskGradTole = maskGrad_tole[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewMaskGradTole, 95)
        viewMaskGradTole = (cmk(viewMaskGradTole / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewMaskGradTole).show()

        toleOverlay = (viewDispGrad[:,:,0:3] * 0.7 + viewMaskGradTole[:, :-1, 0:3] * 0.3).astype(np.uint8)
        # pil.fromarray(toleOverlay).show()


        maskGrad_sense = self.visualizeSense(maskGrad)
        maskGrad_sense = torch.clamp(maskGrad_sense, min = 0, max = 1)
        viewMaskGradSense = maskGrad_sense[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewMaskGradSense, 95)
        viewMaskGradSense = (cmk(viewMaskGradSense / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewMaskGradSense).show()

        senseOverlay = (viewDispGrad[:,:,0:3] * 0.5 + viewMaskGradSense[:, :, 0:3] * 0.5).astype(np.uint8)
        # pil.fromarray(senseOverlay).show()


        dispInTole = maskGrad_tole[:,:,:,:-1] * dispGrad
        viewToleRe = dispInTole[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = 0.1
        viewToleRe = (cm(viewToleRe / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewToleRe).show()

        dispInSense = maskGrad_sense[:,:,:,:] * dispGrad
        viewSenseRe = dispInSense[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = 0.1
        viewSenseRe = (cm(viewSenseRe / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewSenseRe).show()



        toleLevel = self.toleKernel(dispGrad)
        senseLevel = self.senseKernel(dispGrad)
        # err = torch.tanh((1 - toleLevel / senseLevel)) + 1
        # err = err * maskGrad * disp
        # err = torch.tanh((toleLevel - senseLevel))
        # err = torch.tanh((senseLevel - toleLevel))
        maskGrad = torch.clamp(maskGrad, min=0, max = 1)
        # err = err * maskGrad
        # err = torch.tanh(torch.clamp(senseLevel - toleLevel, min=0))
        # err = err * maskGrad
        # loss = torch.mean(err[err > 0.01])
        err = senseLevel * self.senseWid - toleLevel * self.toleWid
        err = err * maskGrad

        viewToelLevel = (toleLevel * maskGrad)[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = 0.1
        viewToelLevel = (cm(viewToelLevel / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewToelLevel).show()

        viewSenselLevel = (senseLevel * maskGrad)[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = 0.1
        viewSenselLevel = (cm(viewSenselLevel / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewSenselLevel).show()


        viewErr = err[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = 0.1
        viewErr = viewErr / vmax
        viewErr = (cm(viewErr)* 255).astype(np.uint8)
        # pil.fromarray(viewErr).show()
        # check
        # plt.imshow(viewErr)
        # x = 402
        # y = 107
        # sr = 10
        # errVal = err[viewIndex, 0, y-sr : y+sr, x-sr : x+sr].max()
        # diff = pureDiff[viewIndex, 0, y-sr : y+sr, x-sr : x+sr].max()


        # outerArea = senseLevel * self.senseWid - toleLevel * self.toleWid
        # outerArea = outerArea * (maskGrad * (err < 0.01).float())
        # viewOuterArea = outerArea[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        # vmax = 0.1
        # viewOuterArea = viewOuterArea / vmax
        # viewOuterArea = (cm(viewOuterArea)* 255).astype(np.uint8)
        # pil.fromarray(viewOuterArea).show()


        errDisp = np.concatenate([viewToelLevel[:,:,0:3], viewSenselLevel[:, :, 0:3], toleOverlay[:, :, 0:3], viewErr[:, :, 0:3]], axis=0)
        # pil.fromarray(errDisp).show()
        return pil.fromarray(errDisp)


"""
class BorderSimilarity(nn.Module):
    def __init__(self):
        super(BorderSimilarity, self).__init__()
        self.init_conv()

    def init_conv(self):
        weightsx = torch.Tensor([
                                [-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
                                [1., 2., 1.],
                                [0., 0., 0.],
                                [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)
        self.disp_convx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.disp_convy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.disp_convx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.disp_convy.weight = nn.Parameter(weightsy,requires_grad=False)


        self.seman_convx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.seman_convy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.seman_convx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.seman_convy.weight = nn.Parameter(weightsy,requires_grad=False)

    def visualize_borderSimilarity(self, disp, foredgroundMask, suppresMask = None, viewIndex = 0):
        dispGrad = torch.cat([self.disp_convx(disp), self.disp_convy(disp)], dim=1)
        # dispGrad = torch.abs(self.disp_convx(disp)) + torch.abs(self.disp_convy(disp))
        maskGrad = torch.cat([self.seman_convx(foredgroundMask), self.seman_convy(foredgroundMask)], dim=1)
        # maskGrad = torch.abs(self.seman_convx(combinedMask)) + torch.abs(self.seman_convy(combinedMask))
        # maskGrad = torch.sum(maskGrad, dim=1, keepdim=True)
        if suppresMask is not None:
            maskSuppress = torch.abs(self.disp_convx(suppresMask)) + torch.abs(self.disp_convy(suppresMask))
            # maskSuppress = torch.masked_fill_(maskSuppress)
            maskSuppress = maskSuppress.masked_fill_(maskSuppress > 1e-1, 1)
            maskSuppress = maskSuppress.repeat([1,2,1,1])
            maskGrad = maskGrad * maskSuppress

        viewForeGroundMask = foredgroundMask[viewIndex, 0, :, :].detach().cpu().numpy()
        viewForeGroundMask = (viewForeGroundMask * 255).astype(np.uint8)
        pil.fromarray(viewForeGroundMask).show()

        # viewBackGroundMask = combinedMask[viewIndex, 1, :, :].detach().cpu().numpy()
        # viewBackGroundMask = (viewBackGroundMask * 255).astype(np.uint8)
        # pil.fromarray(viewBackGroundMask).show()

        cm = plt.get_cmap('magma')
        viewDispGrad = dispGrad[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewDispGrad, 95)
        viewDispGrad = (cm(viewDispGrad / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewDispGrad).show()

        viewDisp = disp[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewDisp, 90)
        viewDisp = (cm(viewDisp / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewDisp).show()

        viewMaskGrad = torch.sum(torch.abs(maskGrad), dim=1, keepdim=True)
        viewMaskGrad = viewMaskGrad[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = 0.1
        viewMaskGrad = (cm(viewMaskGrad / vmax)* 255).astype(np.uint8)
        pil.fromarray(viewMaskGrad).show()


        viewMaskSuppress = maskSuppress[viewIndex, 0, :, :].detach().cpu().numpy()
        vmax = 1
        viewMaskSuppress = (cm(viewMaskSuppress / vmax)* 255).astype(np.uint8)
        pil.fromarray(viewMaskSuppress).show()

        viewBackGroundMask_red = np.stack([viewBackGroundMask, np.zeros_like(viewBackGroundMask), np.zeros_like(viewBackGroundMask)], axis=2)
        viewMaskGradObj = (viewBackGroundMask_red * 0.7 + viewMaskGrad[:,:,0:3] * 0.3).astype(np.uint8)
        # pil.fromarray(viewMaskGradObj).resize([viewMaskGradObj.shape[1] * 2, viewMaskGradObj.shape[0] * 2]).show()

        cmk = plt.get_cmap('hot')
        maskGrad_tole = self.visualizeTole(maskGrad)
        maskGrad_tole = torch.clamp(maskGrad_tole, min=0, max=1)
        viewMaskGradTole = maskGrad_tole[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewMaskGradTole, 95)
        viewMaskGradTole = (cmk(viewMaskGradTole / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewMaskGradTole).show()

        toleOverlay = (viewDispGrad[:,:,0:3] * 0.7 + viewMaskGradTole[:, :-1, 0:3] * 0.3).astype(np.uint8)
        # pil.fromarray(toleOverlay).show()


        maskGrad_sense = self.visualizeSense(maskGrad)
        maskGrad_sense = torch.clamp(maskGrad_sense, min = 0, max = 1)
        viewMaskGradSense = maskGrad_sense[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewMaskGradSense, 95)
        viewMaskGradSense = (cmk(viewMaskGradSense / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewMaskGradSense).show()

        senseOverlay = (viewDispGrad[:,:,0:3] * 0.5 + viewMaskGradSense[:, :, 0:3] * 0.5).astype(np.uint8)
        # pil.fromarray(senseOverlay).show()


        dispInTole = maskGrad_tole[:,:,:,:-1] * dispGrad
        viewToleRe = dispInTole[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = 0.1
        viewToleRe = (cm(viewToleRe / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewToleRe).show()

        dispInSense = maskGrad_sense[:,:,:,:] * dispGrad
        viewSenseRe = dispInSense[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = 0.1
        viewSenseRe = (cm(viewSenseRe / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewSenseRe).show()



        toleLevel = self.toleKernel(dispGrad)
        senseLevel = self.senseKernel(dispGrad)
        # err = torch.tanh((1 - toleLevel / senseLevel)) + 1
        # err = err * maskGrad * disp
        # err = torch.tanh((toleLevel - senseLevel))
        # err = torch.tanh((senseLevel - toleLevel))
        maskGrad = torch.clamp(maskGrad, min=0, max = 1)
        # err = err * maskGrad
        # err = torch.tanh(torch.clamp(senseLevel - toleLevel, min=0))
        # err = err * maskGrad
        # loss = torch.mean(err[err > 0.01])
        err = senseLevel * self.senseWid - toleLevel * self.toleWid
        err = err * maskGrad

        viewToelLevel = (toleLevel * maskGrad)[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = 0.1
        viewToelLevel = (cm(viewToelLevel / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewToelLevel).show()

        viewSenselLevel = (senseLevel * maskGrad)[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = 0.1
        viewSenselLevel = (cm(viewSenselLevel / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewSenselLevel).show()


        viewErr = err[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = 0.1
        viewErr = viewErr / vmax
        viewErr = (cm(viewErr)* 255).astype(np.uint8)
        # pil.fromarray(viewErr).show()
        # check
        # plt.imshow(viewErr)
        # x = 402
        # y = 107
        # sr = 10
        # errVal = err[viewIndex, 0, y-sr : y+sr, x-sr : x+sr].max()
        # diff = pureDiff[viewIndex, 0, y-sr : y+sr, x-sr : x+sr].max()


        # outerArea = senseLevel * self.senseWid - toleLevel * self.toleWid
        # outerArea = outerArea * (maskGrad * (err < 0.01).float())
        # viewOuterArea = outerArea[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        # vmax = 0.1
        # viewOuterArea = viewOuterArea / vmax
        # viewOuterArea = (cm(viewOuterArea)* 255).astype(np.uint8)
        # pil.fromarray(viewOuterArea).show()


        errDisp = np.concatenate([viewToelLevel[:,:,0:3], viewSenselLevel[:, :, 0:3], toleOverlay[:, :, 0:3], viewErr[:, :, 0:3]], axis=0)
        # pil.fromarray(errDisp).show()
        return pil.fromarray(errDisp)

"""


class RandomSampleNeighbourPts(nn.Module):
    def __init__(self, batchNum = 10):
        super(RandomSampleNeighbourPts, self).__init__()
        self.wdSize = 11 # Generate points within a window of 5 by 5
        self.ptsNum = 50000 # Each image generate 50000 number of points
        self.smapleDense = 20 # For each position, sample 20 points
        self.batchNum = batchNum
        self.init_conv()
        self.channelInd = list()
        for i in range(self.batchNum):
            self.channelInd.append(torch.ones(self.ptsNum) * i)
        self.channelInd = torch.cat(self.channelInd, dim=0).long().cuda()
        self.zeroArea = torch.Tensor([0, 1, 2, 3, -1, -2, -3, -4]).long()

    def init_conv(self):
        weightsx = torch.Tensor([
                                [-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
                                [1., 2., 1.],
                                [0., 0., 0.],
                                [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)

        self.seman_convx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.seman_convy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)


        self.disp_convx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.disp_convy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.disp_convx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.disp_convy.weight = nn.Parameter(weightsy,requires_grad=False)

        self.seman_convx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.seman_convy.weight = nn.Parameter(weightsy,requires_grad=False)
        self.expand = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def randomSampleReg(self, disp, foredgroundMask):
        lossSim = 0
        lossContrast = 0

        maskGrad = torch.abs(self.seman_convx(foredgroundMask))
        maskGrad = self.expand(maskGrad) * (disp > 7e-3).float()
        maskGrad = torch.clamp(maskGrad, min = 3) - 3
        maskGrad[:,:,self.zeroArea,:] = 0
        maskGrad[:,:,:,self.zeroArea] = 0

        height = disp.shape[2]
        width = disp.shape[3]

        centerx = torch.LongTensor(self.ptsNum * self.batchNum).random_(self.wdSize, width - self.wdSize)
        centery = torch.LongTensor(self.ptsNum * self.batchNum).random_(self.wdSize, height - self.wdSize)

        onBorderSelection = maskGrad[self.channelInd, 0, centery, centerx] > 1e-1
        if torch.sum(onBorderSelection) < 100:
            return lossSim, lossContrast
        centery = centery[onBorderSelection].unsqueeze(1).repeat([1,self.smapleDense])
        centerx = centerx[onBorderSelection].unsqueeze(1).repeat([1,self.smapleDense])
        channelInd = self.channelInd[onBorderSelection].unsqueeze(1).repeat([1,self.smapleDense])

        validNum = centerx.shape[0]
        bx = torch.LongTensor(validNum * self.smapleDense).random_(-self.wdSize, self.wdSize + 1).view(validNum, self.smapleDense)
        by = torch.LongTensor(validNum * self.smapleDense).random_(-7, 8).view(validNum, self.smapleDense)

        sampledx = centerx + bx
        sampledy = centery + by

        objType = foredgroundMask[channelInd, 0, sampledy, sampledx]
        balancePts = (torch.sum(objType, dim=1) > 4) * (torch.sum(1-objType, dim=1) > 4)
        if torch.sum(balancePts) < 100:
            return lossSim, lossContrast

        sampledx = sampledx[balancePts]
        sampledy = sampledy[balancePts]
        channelInd = channelInd[balancePts]

        objType = foredgroundMask[channelInd, 0, sampledy, sampledx]
        objDisp = disp[channelInd, 0, sampledy, sampledx]
        objType_reverse = 1 - objType

        posNum = torch.sum(objType, dim=1, keepdim=True)
        negNum = torch.sum(1-objType, dim=1, keepdim=True)

        with torch.no_grad():
            zeromeanDisp = (torch.sum(objDisp * objType_reverse, dim = 1, keepdim=True) / negNum - objDisp) * objType_reverse
            zeromeanNormDisp = zeromeanDisp / (torch.norm(zeromeanDisp, dim =1, keepdim = True) + 1e-18)
            softWeights = torch.exp(zeromeanNormDisp) / torch.sum(torch.exp(zeromeanNormDisp) * objType_reverse, dim = 1, keepdim=True) * objType_reverse
            softMin = torch.sum(softWeights * objDisp, dim = 1, keepdim=True)

        lossContrast = torch.mean(torch.sum(torch.clamp(objDisp * objType_reverse - softMin, min=0) * objType_reverse, dim =1, keepdim = True) / negNum)
        lossSim = 0


        # posMean = (torch.sum(objDisp * objType, dim=1, keepdim=True) / posNum)
        # negMean = (torch.sum(objDisp * (1-objType), dim=1, keepdim=True) / negNum)
        # lossSimPos = torch.mean(torch.sqrt(torch.sum((objDisp - posMean)**2 * objType, dim=1, keepdim=True) / posNum + 1e-14))
        # lossSimNeg = torch.mean(torch.sqrt(torch.sum((objDisp - negMean)**2 * (1-objType), dim=1, keepdim=True) / negNum + 1e-14))
        # lossSim = (lossSimPos + lossSimNeg) / 2
        # lossContrast = torch.mean(negMean)
        # lossSim = lossSimPos
        # lossContrast = torch.mean(negMean - posMean) + 0.02
        # assert not torch.isnan(lossContrast) and not torch.isnan(lossSim), "nan occur"
        # if torch.isnan(lossContrast) or torch.isnan(lossSim):
        #     lossContrast = 0
        #     lossSim = 0
        return lossSim, lossContrast
    def visualize_randomSample(self, disp, foredgroundMask, suppresMask = None, viewIndex = 0):
        # maskGrad = torch.abs(self.seman_convx(foredgroundMask)) + torch.abs(self.seman_convy(foredgroundMask))

        # maskGrad = torch.abs(foredgroundMask[:,:,:-1,:] - foredgroundMask[:,:,1:, :])
        # maskGrad = F.pad(maskGrad, [0,0,1,0], 'constant', 0)
        maskGrad = torch.abs(self.seman_convx(foredgroundMask))
        # if suppresMask is not None:
        #     maskSuppress = torch.abs(self.disp_convx(suppresMask)) + torch.abs(self.disp_convy(suppresMask))
        #     maskSuppress = maskSuppress.masked_fill_(maskSuppress > 1e-1, 1)
        #     maskGrad = maskGrad * maskSuppress
        maskGrad = self.expand(maskGrad) * (disp > 7e-3).float()
        maskGrad = torch.clamp(maskGrad, min = 3) - 3
        maskGrad[:,:,self.zeroArea,:] = 0
        maskGrad[:,:,:,self.zeroArea] = 0

        # dispGrad = torch.abs(self.seman_convx(disp)) + torch.abs(self.seman_convy(disp))

        height = disp.shape[2]
        width = disp.shape[3]

        centerx = torch.LongTensor(self.ptsNum * self.batchNum).random_(self.wdSize, width - self.wdSize)
        centery = torch.LongTensor(self.ptsNum * self.batchNum).random_(self.wdSize, height - self.wdSize)

        onBorderSelection = maskGrad[self.channelInd, 0, centery, centerx] > 1e-1
        centery = centery[onBorderSelection].unsqueeze(1).repeat([1,self.smapleDense])
        centerx = centerx[onBorderSelection].unsqueeze(1).repeat([1,self.smapleDense])
        channelInd = self.channelInd[onBorderSelection].unsqueeze(1).repeat([1,self.smapleDense])

        validNum = centerx.shape[0]
        bx = torch.LongTensor(validNum * self.smapleDense).random_(-self.wdSize, self.wdSize + 1).view(validNum, self.smapleDense)
        by = torch.LongTensor(validNum * self.smapleDense).random_(-7, 8).view(validNum, self.smapleDense)

        sampledx = centerx + bx
        sampledy = centery + by

        # pairedx = pairedx[onBorderSelection]
        # pairedy = pairedy[onBorderSelection]

        objType = foredgroundMask[channelInd, 0, sampledy, sampledx]
        balancePts = (torch.sum(objType, dim=1) > 4) * (torch.sum(1-objType, dim=1) > 4)

        sampledx = sampledx[balancePts]
        sampledy = sampledy[balancePts]
        centerx = centerx[balancePts]
        centery = centery[balancePts]
        channelInd = channelInd[balancePts]

        objType = foredgroundMask[channelInd, 0, sampledy, sampledx]
        objDisp = disp[channelInd, 0, sampledy, sampledx]

        posNum = torch.sum(objType, dim=1, keepdim=True)
        negNum = torch.sum(1-objType, dim=1, keepdim=True)

        # Check
        # varPosRe = torch.sum((objDisp - (torch.sum(objDisp * objType, dim=1, keepdim=True) / posNum))**2 * objType, dim=1, keepdim=True) / posNum
        # ckInd = 100
        # pttt = objDisp[ckInd, objType[ckInd, :].type(torch.ByteTensor)]
        # varVal = torch.mean((pttt - torch.mean(pttt)) ** 2)
        # ckVal =varPosRe[ckInd]
        #
        # varNegRe = torch.sum((objDisp - (torch.sum(objDisp * (1-objType), dim=1, keepdim=True) / negNum))**2 * (1-objType), dim=1, keepdim=True) / negNum
        # ckInd = 100
        # pttt = objDisp[ckInd, 1-objType[ckInd, :].type(torch.ByteTensor)]
        # varVal = torch.mean((pttt - torch.mean(pttt)) ** 2)
        # ckVal =varNegRe[ckInd]

        posMean = (torch.sum(objDisp * objType, dim=1, keepdim=True) / posNum)
        negMean = (torch.sum(objDisp * (1-objType), dim=1, keepdim=True) / negNum)
        lossSimPos = torch.mean(torch.sqrt(torch.sum((objDisp - posMean)**2 * objType, dim=1, keepdim=True) / posNum))
        lossSimNeg = torch.mean(torch.sqrt(torch.sum((objDisp - negMean)**2 * (1-objType), dim=1, keepdim=True) / negNum))
        lossSim = (lossSimPos + lossSimNeg) / 2
        lossContrast = torch.mean(negMean - posMean)

        cm = plt.get_cmap('magma')
        viewMaskGrad = maskGrad[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewMaskGrad, 99.5)
        viewMaskGrad = (cm(viewMaskGrad / vmax) * 255).astype(np.uint8)
        # pil.fromarray(viewMaskGrad).show()

        # viewDispGrad = dispGrad[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        # vmax = np.percentile(viewDispGrad, 95)
        # viewDispGrad = (cm(viewDispGrad / vmax) * 255).astype(np.uint8)
        # pil.fromarray(viewDispGrad).show()

        viewDisp = disp[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewDisp, 90)
        viewDisp = (cm(viewDisp / vmax) * 255).astype(np.uint8)
        # pil.fromarray(viewDisp).show()

        viewDispGradOverlay = (viewMaskGrad * 0.3 + viewDisp * 0.7).astype(np.uint8)
        # pil.fromarray(viewDispGradOverlay).resize([viewDispGradOverlay.shape[1] * 2, viewDispGradOverlay.shape[0] * 2]).show()

        viewForeMask = foredgroundMask[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewForeMask, 99)
        viewForeMask = (cm(viewForeMask / vmax) * 255).astype(np.uint8)
        # pil.fromarray(viewForeMask).show()

        # View the point pairs within same objects cat
        sampleInter = 500
        colors = torch.rand((sampledx.shape[0], 1)).repeat(1, self.smapleDense)
        curFrameSel = channelInd[:,0] == viewIndex
        scX = sampledx[curFrameSel]
        scY = sampledy[curFrameSel]
        scC = colors[curFrameSel]
        # curFrameSel = channelInd[:, 0] == viewIndex
        # fx = sampledx[curFrameSel]
        # fy = sampledy[curFrameSel]
        # obcur = objType[curFrameSel].byte()
        # fx = fx[obcur]
        # fy = fy[obcur]
        plt.imshow(viewForeMask[:,:,0:3])
        plt.scatter(scX[::sampleInter,:].contiguous().view(-1).cpu().numpy(), scY[::sampleInter,:].contiguous().view(-1).cpu().numpy(), c = scC[::sampleInter,:].contiguous().view(-1).cpu().numpy(), s = 0.6)
        # plt.scatter(fx[::sampleInter].cpu().numpy(),
        #             fy[::sampleInter].cpu().numpy(),
        #             c='r', s=0.6)
        plt.close()

        # sampleInter = 10
        scCx = centerx[curFrameSel]
        scCy = centery[curFrameSel]
        plt.imshow(viewMaskGrad[:,:,0:3])
        # plt.scatter(scX[::sampleInter,:].contiguous().view(-1).cpu().numpy(), scY[::sampleInter,:].contiguous().view(-1).cpu().numpy(), c = scC[::sampleInter,:].contiguous().view(-1).cpu().numpy(), s = 0.6)
        plt.scatter(scCx[::sampleInter, 0].contiguous().view(-1).cpu().numpy(),
                    scCy[::sampleInter, 0].contiguous().view(-1).cpu().numpy(),
                    c='r', s=1.1)
        plt.close()
        # curChannelPosPts = (channelInd == viewIndex) * smilarComp
        # ptsSet1 = np.expand_dims(np.stack([centerx[curChannelPosPts][::2], centery[curChannelPosPts][::2]], axis=1), axis=1)
        # ptsSet2 = np.expand_dims(np.stack([pairedx[curChannelPosPts][::2], pairedy[curChannelPosPts][::2]], axis=1), axis=1)
        # ptsSet = np.concatenate([ptsSet1, ptsSet2], axis=1)
        # ln_coll = LineCollection(ptsSet, colors='r')
        # ax.add_collection(ln_coll)

class RandomSampleBorderSemanPts(nn.Module):
    def __init__(self, batchNum=10):
        super(RandomSampleBorderSemanPts, self).__init__()
        self.wdSize = 11  # Generate points within a window of 5 by 5
        self.ptsNum = 500  # Each image generate 50000 number of points
        self.smapleDense = 2000  # For each position, sample 20 points
        self.batchNum = batchNum
        self.init_conv()
        self.channelInd = list()
        for i in range(self.batchNum):
            self.channelInd.append(torch.ones(self.ptsNum) * i)
        self.channelInd = torch.cat(self.channelInd, dim=0).long().cuda()
        self.zeroArea = torch.Tensor([0,1,2,3,-1,-2,-3,-4]).long()
        self.kmeanNum = 10

    def init_conv(self):
        weightsx = torch.Tensor([
                                [-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
                                [1., 2., 1.],
                                [0., 0., 0.],
                                [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)

        self.seman_convx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.seman_convy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.disp_convx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.disp_convy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.disp_convx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.disp_convy.weight = nn.Parameter(weightsy,requires_grad=False)

        self.seman_convx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.seman_convy.weight = nn.Parameter(weightsy,requires_grad=False)
        self.expand = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.deblobConv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        weightBlob = torch.ones([3,3]).unsqueeze(0).unsqueeze(0) / 9
        self.deblobConv.weight = nn.Parameter(weightBlob, requires_grad=False)
    def visualizeBorderSample(self, disp, forepred, gtMask, viewIndex = 0):
        maskGrad = torch.abs(self.seman_convx(forepred))
        maskGrad = self.expand(maskGrad) * (disp > 7e-3).float()
        maskGrad = torch.clamp(maskGrad, min = 3) - 3
        maskGrad[:,:,self.zeroArea,:] = 0
        maskGrad[:,:,:,self.zeroArea] = 0


        height = disp.shape[2]
        width = disp.shape[3]

        centerx = torch.LongTensor(self.ptsNum * self.batchNum).random_(self.wdSize, width - self.wdSize)
        centery = torch.LongTensor(self.ptsNum * self.batchNum).random_(self.wdSize, height - self.wdSize)

        onBorderSelection = maskGrad[self.channelInd, 0, centery, centerx] > 1e-1
        centery = centery[onBorderSelection].unsqueeze(1).repeat([1,self.smapleDense])
        centerx = centerx[onBorderSelection].unsqueeze(1).repeat([1,self.smapleDense])
        channelInd = self.channelInd[onBorderSelection].unsqueeze(1).repeat([1,self.smapleDense])

        validNum = centerx.shape[0]
        bx = torch.LongTensor(validNum * self.smapleDense).random_(-self.wdSize, self.wdSize + 1).view(validNum, self.smapleDense)
        by = torch.LongTensor(validNum * self.smapleDense).random_(-7, 8).view(validNum, self.smapleDense)

        sampledx = centerx + bx
        sampledy = centery + by

        sampledDisp = disp[channelInd, 0, sampledy, sampledx]
        kmeanLarge, _ = torch.max(sampledDisp, dim=1, keepdim=True)
        kmeanSmall, _ = torch.min(sampledDisp, dim=1, keepdim=True)
        for i in range(self.kmeanNum):
            distLarge = torch.abs(sampledDisp - kmeanLarge)
            distSmall = torch.abs(sampledDisp - kmeanSmall)

            belongingLarge = (distLarge <= distSmall).float()
            belongingSmall = 1 - belongingLarge

            kmeanLarge = torch.sum(sampledDisp * belongingLarge, dim=1, keepdim=True) / torch.sum(belongingLarge, dim=1, keepdim=True)
            kmeanSmall = torch.sum(sampledDisp * belongingSmall, dim=1, keepdim=True) / torch.sum(belongingSmall, dim=1, keepdim=True)

        contrast = ((kmeanLarge - kmeanSmall) > 0.005) * (torch.sum(belongingLarge, dim=1) > 5)
        contrast = contrast[:,0]

        sampledx = sampledx[contrast]
        sampledy = sampledy[contrast]
        channelInd = channelInd[contrast]

        belongingLarge = belongingLarge[contrast]
        belongingSmall = belongingSmall[contrast]
        belongingLarge = belongingLarge.byte()
        belongingSmall = belongingSmall.byte()

        forex = sampledx[belongingLarge]
        forey = sampledy[belongingLarge]
        forec = channelInd[belongingLarge]

        backx = sampledx[belongingSmall]
        backy = sampledy[belongingSmall]
        backc = channelInd[belongingSmall]

        cm = plt.get_cmap('magma')
        viewMaskGrad = maskGrad[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewMaskGrad, 99.5)
        viewMaskGrad = (cm(viewMaskGrad / vmax) * 255).astype(np.uint8)
        pil.fromarray(viewMaskGrad).show()

        viewDisp = disp[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewDisp, 90)
        viewDisp = (cm(viewDisp / vmax) * 255).astype(np.uint8)
        pil.fromarray(viewDisp).show()

        viewForePred = forepred[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        viewForePred = (cm(viewForePred) * 255).astype(np.uint8)
        pil.fromarray(viewForePred).show()

        sampleInter = 10
        foreSel = forec == viewIndex
        scfX = forex[foreSel]
        scfY = forey[foreSel]
        backSel = backc == viewIndex
        scbX = backx[backSel]
        scbY = backy[backSel]
        plt.imshow(viewDisp[:,:,0:3])
        plt.scatter(scfX[::sampleInter].contiguous().view(-1).cpu().numpy(), scfY[::sampleInter].contiguous().view(-1).cpu().numpy(), c = 'r', s = 0.6)
        plt.scatter(scbX[::sampleInter].contiguous().view(-1).cpu().numpy(), scbY[::sampleInter].contiguous().view(-1).cpu().numpy(), c='g', s=0.6)

        plt.imshow(viewForePred[:,:,0:3])
        plt.scatter(scfX[::sampleInter].contiguous().view(-1).cpu().numpy(), scfY[::sampleInter].contiguous().view(-1).cpu().numpy(), c = 'r', s = 0.6)
        plt.scatter(scbX[::sampleInter].contiguous().view(-1).cpu().numpy(), scbY[::sampleInter].contiguous().view(-1).cpu().numpy(), c='g', s=0.6)

        correctRate = (torch.sum(gtMask[forec, 0, forey, forex] == 1) + torch.sum(gtMask[backc, 0, backy, backx] == 0)).float() / (forex.shape[0] + backx.shape[0])
        semanCorrectRate = torch.sum(forepred[channelInd.view(-1), 0, sampledy.view(-1), sampledx.view(-1)] == gtMask[channelInd.view(-1), 0, sampledy.view(-1), sampledx.view(-1)]).float() / (forex.shape[0] + backx.shape[0])
        # plt.close()

"""
class RandomSampleNeighbourPts(nn.Module):
    def __init__(self, batchNum = 10):
        super(RandomSampleNeighbourPts, self).__init__()
        self.wdSize = 11 # Generate points within a window of 5 by 5
        self.ptsNum = 50000 # Each image generate 50000 number of points
        self.batchNum = batchNum
        self.init_conv()
        self.channelInd = list()
        for i in range(self.batchNum):
            self.channelInd.append(torch.ones(self.ptsNum) * i)
        self.channelInd = torch.cat(self.channelInd, dim=0).long().cuda()


    def init_conv(self):
        weightsx = torch.Tensor([
                                [-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
                                [1., 2., 1.],
                                [0., 0., 0.],
                                [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)

        self.seman_convx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.seman_convy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)


        self.disp_convx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.disp_convy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.disp_convx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.disp_convy.weight = nn.Parameter(weightsy,requires_grad=False)

        self.seman_convx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.seman_convy.weight = nn.Parameter(weightsy,requires_grad=False)
        self.expand = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.zeroArea = torch.Tensor([0, 1, 2, 3, -1, -2, -3, -4]).long()


    def randomSampleReg(self, disp, foredgroundMask):
        # maskGrad = torch.abs(self.seman_convx(foredgroundMask)) + torch.abs(self.seman_convy(foredgroundMask))
        # maskGrad = self.expand(maskGrad)
        with torch.no_grad():
            maskGrad = torch.abs(self.seman_convx(foredgroundMask))
            maskGrad = self.expand(maskGrad) * (disp > 7e-3).float()
            maskGrad = torch.clamp(maskGrad, min = 3) - 3
            maskGrad[:,:,self.zeroArea,:] = 0
            maskGrad[:,:,:,self.zeroArea] = 0

        height = disp.shape[2]
        width = disp.shape[3]

        centerx = torch.LongTensor(self.ptsNum * self.batchNum).random_(self.wdSize, width - self.wdSize)
        centery = torch.LongTensor(self.ptsNum * self.batchNum).random_(self.wdSize, height - self.wdSize)

        bx = torch.LongTensor(self.ptsNum * self.batchNum).random_(-self.wdSize, self.wdSize + 1)
        by = torch.LongTensor(self.ptsNum * self.batchNum).random_(-self.wdSize, self.wdSize + 1)

        pairedx = centerx + bx
        pairedy = centery + by

        onBorderSelection = maskGrad[self.channelInd, 0, centery, centerx] > 1e-1
        centery = centery[onBorderSelection]
        centerx = centerx[onBorderSelection]
        pairedx = pairedx[onBorderSelection]
        pairedy = pairedy[onBorderSelection]


        channelInd = self.channelInd[onBorderSelection]
        anchorType = foredgroundMask[channelInd, 0, centery, centerx]
        pairType = foredgroundMask[channelInd, 0, pairedy, pairedx]
        anchorDisp = disp[channelInd, 0, centery, centerx]
        pairDisp = disp[channelInd, 0, pairedy, pairedx]

        smilarComp = anchorType == pairType
        contrastComp = 1 - smilarComp
        contrastCompAnchorPos = contrastComp * (anchorType == 1)
        contrastCompPairPos = contrastComp * (pairType == 1)


        if torch.sum(smilarComp) == 0:
            similarLoss = 0
        else:
            similarLoss = torch.mean(torch.abs(anchorDisp[smilarComp] - pairDisp[smilarComp]))
        if torch.sum(contrastCompPairPos) == 0 or torch.sum(contrastCompAnchorPos) == 0:
            contrastLoss = 0
        else:
            contrastLoss = torch.mean((anchorDisp[contrastCompPairPos] - pairDisp[contrastCompPairPos])) \
                            + torch.mean((pairDisp[contrastCompAnchorPos] - anchorDisp[contrastCompAnchorPos]))
            contrastLoss = contrastLoss / 2 + 0.02
        return similarLoss, contrastLoss
    def visualize_randomSample(self, disp, foredgroundMask, suppresMask = None, viewIndex = 0):
        # maskGrad = torch.abs(self.seman_convx(foredgroundMask)) + torch.abs(self.seman_convy(foredgroundMask))
        # maskGrad = self.expand(maskGrad)
        with torch.no_grad():
            maskGrad = torch.abs(self.seman_convx(foredgroundMask))
            maskGrad = self.expand(maskGrad) * (disp > 7e-3).float()
            maskGrad = torch.clamp(maskGrad, min = 3) - 3
            maskGrad[:,:,self.zeroArea,:] = 0
            maskGrad[:,:,:,self.zeroArea] = 0

        height = disp.shape[2]
        width = disp.shape[3]

        centerx = torch.LongTensor(self.ptsNum * self.batchNum).random_(self.wdSize, width - self.wdSize)
        centery = torch.LongTensor(self.ptsNum * self.batchNum).random_(self.wdSize, height - self.wdSize)

        bx = torch.LongTensor(self.ptsNum * self.batchNum).random_(-self.wdSize, self.wdSize + 1)
        by = torch.LongTensor(self.ptsNum * self.batchNum).random_(-7, 8)
        # by = torch.LongTensor(self.ptsNum * self.batchNum).random_(-self.wdSize, self.wdSize + 1)

        pairedx = centerx + bx
        pairedy = centery + by

        onBorderSelection = maskGrad[self.channelInd, 0, centery, centerx] > 1e-1
        centery = centery[onBorderSelection]
        centerx = centerx[onBorderSelection]
        pairedx = pairedx[onBorderSelection]
        pairedy = pairedy[onBorderSelection]


        channelInd = self.channelInd[onBorderSelection]
        anchorType = foredgroundMask[channelInd, 0, centery, centerx]
        pairType = foredgroundMask[channelInd, 0, pairedy, pairedx]
        anchorDisp = disp[channelInd, 0, centery, centerx]
        pairDisp = disp[channelInd, 0, pairedy, pairedx]

        smilarComp = anchorType == pairType
        contrastComp = 1 - smilarComp
        contrastCompAnchorPos = contrastComp * (anchorType == 1)
        contrastCompPairPos = contrastComp * (pairType == 1)

        similarLoss = torch.mean(torch.abs(anchorDisp[smilarComp] - pairDisp[smilarComp]))
        contrastLoss = torch.mean(torch.exp(anchorDisp[contrastCompPairPos] - pairDisp[contrastCompPairPos])) \
                        + torch.mean(torch.exp(pairDisp[contrastCompAnchorPos] - anchorDisp[contrastCompAnchorPos]))


        cm = plt.get_cmap('magma')
        viewMaskGrad = maskGrad[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewMaskGrad, 99)
        viewMaskGrad = (cm(viewMaskGrad / vmax) * 255).astype(np.uint8)
        pil.fromarray(viewMaskGrad).show()

        viewDisp = disp[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewDisp, 99)
        viewDisp = (cm(viewDisp / vmax) * 255).astype(np.uint8)
        pil.fromarray(viewDisp).show()

        viewForeMask = foredgroundMask[viewIndex, :, :, :].squeeze(0).detach().cpu().numpy()
        vmax = np.percentile(viewForeMask, 99)
        viewForeMask = (cm(viewForeMask / vmax) * 255).astype(np.uint8)
        pil.fromarray(viewForeMask).show()


        # View the point pairs within same objects cat
        plt.imshow(viewDisp[:,:,0:3])
        curChannelPosPts = (channelInd == viewIndex) * smilarComp
        plt.scatter(centerx[curChannelPosPts][::2], centery[curChannelPosPts][::2], c = 'r', s = 0.5)
        plt.scatter(pairedx[curChannelPosPts][::2], pairedy[curChannelPosPts][::2], c='g', s=0.5)
        plt.close()
"""

class DepthGuessesBySemantics(nn.Module):
    def __init__(self, width, height, batchNum=10):
        super(DepthGuessesBySemantics, self).__init__()
        # self.wdSize = 11  # Generate points within a window of 5 by 5
        # self.wdSize = 17  # Generate points within a window of 5 by 5
        self.wdSize = 23  # Generate points within a window of 5 by 5
        # self.ptsNum = 500  # Each image generate 50000 number of points
        self.ptsNum = 1000
        self.smapleDense = 200  # For each position, sample 20 points
        self.batchNum = batchNum
        self.width = width
        self.height = height
        self.init_conv()
        self.channelInd = list()
        for i in range(self.batchNum):
            self.channelInd.append(torch.ones(self.ptsNum * 100) * i)
        self.channelInd = torch.cat(self.channelInd, dim=0).long().cuda()

        self.channelInd_wall = list()
        for i in range(self.batchNum):
            self.channelInd_wall.append(torch.ones(self.ptsNum * 100) * i)
        self.channelInd_wall = torch.cat(self.channelInd_wall, dim=0).long().cuda()

        self.zeroArea = torch.Tensor([0,1,2,3,-1,-2,-3,-4]).long()

        yy, xx = torch.meshgrid([torch.arange(0, self.height), torch.arange(0, self.width)])
        xx = xx.float().unsqueeze(0).repeat(self.batchNum, 1, 1)
        yy = yy.float().unsqueeze(0).repeat(self.batchNum, 1, 1)
        self.xx = xx.contiguous().view(-1).cuda()
        self.yy = yy.contiguous().view(-1).cuda()
        self.ones = torch.ones(self.xx.shape[0], device=torch.device("cuda"))
        self.tdevice = torch.device("cuda")
        self.maxDepth = 30
    def init_conv(self):
        weightsx = torch.Tensor([
                                [-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
                                [1., 2., 1.],
                                [0., 0., 0.],
                                [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)

        self.seman_convx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.seman_convy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.disp_convx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.disp_convy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.disp_convx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.disp_convy.weight = nn.Parameter(weightsy,requires_grad=False)

        self.seman_convx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.seman_convy.weight = nn.Parameter(weightsy,requires_grad=False)
        # self.expand = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.expand = torch.nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.expand_road = torch.nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
    def regBySeman(self, realDepth, dispAct, foredgroundMask, wallTypeMask, groundTypeMask, intrinsic, extrinsic):
        with torch.no_grad():
            maskGrad = torch.abs(self.seman_convx(foredgroundMask)) + torch.abs(self.seman_convy(foredgroundMask))
            maskGrad = self.expand_road(maskGrad) * (dispAct > 7e-3).float()
            maskGrad = torch.clamp(maskGrad, min=3) - 3
            maskGrad[:, :, self.zeroArea, :] = 0
            maskGrad[:, :, :, self.zeroArea] = 0

            centerx = torch.LongTensor(self.ptsNum * self.batchNum * 100).random_(self.wdSize, self.width - self.wdSize).cuda()
            centery = torch.LongTensor(self.ptsNum * self.batchNum * 100).random_(self.wdSize, self.height - self.wdSize).cuda()
            # centerx = torch.LongTensor(self.ptsNum * self.batchNum * 100).random_(self.wdSize, self.width - self.wdSize)
            # centery = torch.LongTensor(self.ptsNum * self.batchNum * 100).random_(self.wdSize, self.height - self.wdSize)
            onBorderSelection = maskGrad[self.channelInd, 0, centery, centerx] > 1e-1
            centerx = centerx[onBorderSelection]
            centery = centery[onBorderSelection]
            channelInd = self.channelInd[onBorderSelection]

            roadBackGroundPts = groundTypeMask[channelInd, 0, centery, centerx].byte()
            onBorderSelection[onBorderSelection] = roadBackGroundPts
            channelInd = channelInd[roadBackGroundPts]
            centerx = centerx[roadBackGroundPts]
            centery = centery[roadBackGroundPts]

            # Road
            realDepth_flat = realDepth.view(-1)
            roadPts3DCam = torch.stack([self.xx * realDepth_flat, self.yy * realDepth_flat, realDepth.view(-1), self.ones], dim=1).view(self.batchNum, -1, 4).permute(0,2,1)
            roadPts3DParam = torch.matmul(intrinsic, extrinsic)
            roadPts3DParam_inv = torch.inverse(roadPts3DParam)
            roadPts3DCam = torch.matmul(roadPts3DParam_inv, roadPts3DCam)
            roadPts3DCam = roadPts3DCam.view(self.batchNum, 4, self.height, self.width)

            planeParamEst = torch.ones(self.batchNum, 1, 4, device=self.tdevice)
            for i in range(self.batchNum):
                roadPts = roadPts3DCam[i, :, groundTypeMask[i, 0, :, :].byte()][0:3, :]
                for j in range(4):
                    meanz = torch.mean(roadPts[2,:])
                    selected = torch.abs(roadPts[2,:] - meanz) < 0.5
                    roadPts = roadPts[:,selected]
                planeParam = torch.Tensor([0,0,1,-meanz]).cuda()
                planeParamEst[i, 0, :] = planeParam

        depthOld = torch.clamp(realDepth[channelInd, 0, centery, centerx], min = 0.3)
        M = torch.matmul(planeParamEst, roadPts3DParam_inv)
        depthNew = - (M[channelInd, 0, 3] / (M[channelInd, 0, 0] * centerx.float() + M[channelInd, 0, 1] * centery.float() + M[channelInd,0,2]))
        depthNew = torch.clamp(depthNew, min = 0.3, max=self.maxDepth)
        lossRoad = torch.mean(torch.clamp(depthNew.detach() - depthOld, min=0)) * 1e-3
        if torch.isnan(lossRoad):
            lossRoad = 0

        # Wall Part
        with torch.no_grad():
            maskGrad = torch.abs(self.seman_convx(foredgroundMask)) + torch.abs(self.seman_convy(foredgroundMask))
            # maskGrad = self.expand(maskGrad) * (dispAct > 7e-3).float()
            maskGrad = self.expand(maskGrad) * (dispAct > 1e-2).float()
            maskGrad = torch.clamp(maskGrad, min=3) - 3
            maskGrad[:, :, self.zeroArea, :] = 0
            maskGrad[:, :, :, self.zeroArea] = 0
            maskGrad = self.expand(maskGrad)
            centerx = torch.LongTensor(self.ptsNum * self.batchNum * 100).random_(self.wdSize, self.width - self.wdSize)
            centery = torch.LongTensor(self.ptsNum * self.batchNum * 100).random_(self.wdSize, self.height - self.wdSize)


            onBorderSelection = (maskGrad[self.channelInd_wall, 0, centery, centerx] > 1e-1) * (
                    wallTypeMask[self.channelInd_wall, 0, centery, centerx] == 1)
            centery = centery[onBorderSelection]
            centerx = centerx[onBorderSelection]
            channelInd = self.channelInd_wall[onBorderSelection]

            validNum = centerx.shape[0]
            bx = torch.LongTensor(validNum).random_(-self.wdSize, self.wdSize + 1)
            by = torch.LongTensor(validNum).random_(-7, 8)

            sampledx_pair = centerx + bx
            sampledy_pair = centery + by

            onBackSelection = wallTypeMask[channelInd, 0, sampledy_pair, sampledx_pair] == 1
            centery = centery[onBackSelection]
            centerx = centerx[onBackSelection]
            channelInd = channelInd[onBackSelection]
            sampledx_pair = sampledx_pair[onBackSelection]
            sampledy_pair = sampledy_pair[onBackSelection]
        sourceDisp = dispAct[channelInd, 0, centery, centerx]
        targetDisp = dispAct[channelInd, 0, sampledy_pair, sampledx_pair]
        # errSel = (sourceDisp / targetDisp > 1.15).float()
        errSel = ((sourceDisp / targetDisp > 1.15) * (sourceDisp > 1e-2)).float()
        lossWall = torch.sum(torch.clamp(sourceDisp - targetDisp.detach(), min=0) * errSel) / torch.sum(errSel) * 0.1
        # lossWall = torch.mean(torch.clamp(sourceDisp - targetDisp.detach(), min=0)) * 0.1
        if torch.isnan(lossWall):
            lossWall = 0

        return lossRoad, lossWall


    def visualizeDepthGuess(self, realDepth, dispAct, foredgroundMask, wallTypeMask, groundTypeMask, intrinsic, extrinsic, semantic = None, cts_meta = None, viewInd = 0):
        """
        with torch.no_grad():
            height = 1024
            width = 2048
            xx, yy = np.meshgrid(np.arange(width), np.arange(height))
            xx = xx.flatten()
            yy = yy.flatten()
            objType = 19
            colorMap = np.zeros((objType + 1, xx.shape[0], 3), dtype=np.uint8)
            for i in range(objType):
                if i == objType:
                    k = 255
                else:
                    k = i
                colorMap[i, :, :] = np.repeat(np.expand_dims(np.array(trainId2label[k].color), 0), xx.shape[0], 0)
            colorMap = colorMap.astype(np.float)
            colorMap = colorMap / 255
            semantic_small = F.interpolate(semantic.unsqueeze(1).float(), size = list(realDepth.shape[2:]), mode='nearest')

            gt_semanticMap = semantic[viewInd, :, :].cpu().numpy().flatten()
            gt_semanticMap[gt_semanticMap == 255] = 19
            colors = colorMap[gt_semanticMap, np.arange(xx.shape[0]), :]
            gtdepth = cts_meta['depthMap'][viewInd, :, :].cpu().numpy()
            mask = gtdepth > 0
            mask = mask.flatten()
            depthFlat = gtdepth.flatten()
            oneColumn = np.ones(gtdepth.shape[0] * gtdepth.shape[1])
            pixelLoc = np.stack(
                [xx[mask] * depthFlat[mask], yy[mask] * depthFlat[mask], depthFlat[mask], oneColumn[mask]],
                axis=1)
            intrinsic_gt = cts_meta['intrinsic'][viewInd, :, :].cpu().numpy()
            extrinsic_gt = cts_meta['extrinsic'][viewInd, :, :].cpu().numpy()
            cam_coord_gt = (np.linalg.inv(intrinsic_gt) @ pixelLoc.T).T
            veh_coord = (np.linalg.inv(extrinsic_gt) @ cam_coord_gt.T).T
            colors = colors[mask, :].astype(np.float)

            sampleDense = 100
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=6., azim=170)
            ax.dist = 4
            ax.scatter(veh_coord[::sampleDense, 0], veh_coord[::sampleDense, 1], veh_coord[::sampleDense, 2], s=0.1, c = colors[::sampleDense, :])
            ax.set_zlim(-10, 10)
            plt.ylim([-10, 10])
            plt.xlim([10, 16])
            set_axes_equal(ax)

            colorMap = np.zeros((objType + 1, xx.shape[0], 3), dtype=np.uint8)
            for i in range(objType):
                if i == objType:
                    k = 255
                else:
                    k = i
                colorMap[i, :, :] = np.repeat(np.expand_dims(np.array(trainId2label[k].color), 0), xx.shape[0], 0)
            colorMap = colorMap.astype(np.float)
            colorMap = colorMap / 255
            semantic_small = F.interpolate(semantic.unsqueeze(1).float(), size = list(realDepth.shape[2:]), mode='nearest')
            initrinsic_test = intrinsic[viewInd, :, :].cpu().numpy()
            extrinsic_test = extrinsic[viewInd, :, :].cpu().numpy()
            seman_test = semantic_small[viewInd, :, :].cpu().numpy().flatten()
            seman_test[seman_test == 255] = 19
            seman_test = seman_test.astype(np.int)
            height = realDepth.shape[2]
            width = realDepth.shape[3]
            xx, yy = np.meshgrid(np.arange(width), np.arange(height))
            xx = xx.flatten()
            yy = yy.flatten()
            predDepth = realDepth[viewInd, 0,:,:].cpu().numpy().flatten()
            oneColumn = np.ones(height * width)
            pixelLoc = np.stack(
                [xx * predDepth, yy * predDepth, predDepth, oneColumn],
                axis=1)
            cam_coord_gt2 = (np.linalg.inv(initrinsic_test) @ pixelLoc.T).T
            veh_coord2 = (np.linalg.inv(extrinsic_test) @ cam_coord_gt2.T).T
            colors2 = colorMap[seman_test, np.arange(xx.shape[0]), :].astype(np.float)

            sampleDense = 10
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=6., azim=170)
            ax.dist = 4
            ax.scatter(veh_coord2[::sampleDense, 0], veh_coord2[::sampleDense, 1], veh_coord2[::sampleDense, 2], s=0.1, c = colors2[::sampleDense, :])
            # ax.scatter(veh_coord[::sampleDense, 0], veh_coord[::sampleDense, 1], veh_coord[::sampleDense, 2], s=0.1,
            #            c='r')
            ax.set_zlim(-10, 10)
            plt.ylim([-10, 10])
            plt.xlim([10, 16])
            set_axes_equal(ax)

            sampleDense = 10
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=6., azim=170)
            ax.dist = 4
            ax.scatter(veh_coord2[::sampleDense, 0], veh_coord2[::sampleDense, 1], veh_coord2[::sampleDense, 2], s=0.1, c = 'r')
            ax.scatter(veh_coord[::sampleDense, 0], veh_coord[::sampleDense, 1], veh_coord[::sampleDense, 2], s=0.1,
                       c='g')
            ax.set_zlim(-10, 10)
            plt.ylim([-10, 10])
            plt.xlim([10, 16])
            set_axes_equal(ax)


            maskGrad = torch.abs(self.seman_convx(foredgroundMask))
            maskGrad = self.expand(maskGrad) * (dispAct > 7e-3).float()
            maskGrad = torch.clamp(maskGrad, min=3) - 3
            maskGrad[:, :, self.zeroArea, :] = 0
            maskGrad[:, :, :, self.zeroArea] = 0

            height = dispAct.shape[2]
            width = dispAct.shape[3]

            centerx = torch.LongTensor(self.ptsNum * self.batchNum).random_(self.wdSize, width - self.wdSize)
            centery = torch.LongTensor(self.ptsNum * self.batchNum).random_(self.wdSize, height - self.wdSize)

            onBorderSelection = maskGrad[self.channelInd, 0, centery, centerx] > 1e-1
            centery = centery[onBorderSelection].unsqueeze(1).repeat([1, self.smapleDense])
            centerx = centerx[onBorderSelection].unsqueeze(1).repeat([1, self.smapleDense])
            channelInd = self.channelInd[onBorderSelection].unsqueeze(1).repeat([1, self.smapleDense])

            validNum = centerx.shape[0]
            bx = torch.LongTensor(validNum * self.smapleDense).random_(-self.wdSize, self.wdSize + 1).view(validNum,
                                                                                                           self.smapleDense)
            by = torch.LongTensor(validNum * self.smapleDense).random_(-7, 8).view(validNum, self.smapleDense)

            sampledx = centerx + bx
            sampledy = centery + by

            roadBackGroundPts = groundTypeMask[channelInd, 0, sampledy, sampledx]
            roadBackSelection = torch.sum(roadBackGroundPts, 1) > 20
            roadBackGroundPts = roadBackGroundPts[roadBackSelection, :]
            roadX = sampledx[roadBackSelection, :]
            roadY = sampledy[roadBackSelection, :]
            roadChannle = channelInd[roadBackSelection, :]
            roadDepth = realDepth[roadChannle, 0, roadY, roadX]
            roadX = roadX.float().cuda()
            roadY = roadY.float().cuda()
            roadPts3DCam = torch.stack([roadX * roadDepth, roadY * roadDepth, roadDepth, torch.ones_like(roadDepth)], dim=2).view(-1,4,1)
            roadPts3DParam = torch.inverse(torch.matmul(intrinsic[roadChannle[:,0], :, :], extrinsic[roadChannle[:,0], :, :])).unsqueeze(1).repeat(1, self.smapleDense, 1, 1).view(-1,4,4)
            roadPts3D = torch.matmul(roadPts3DParam, roadPts3DCam).view(-1, self.smapleDense, 4, 1)
            k_ground = (torch.sum(roadPts3D[:,:,2,0] * roadBackGroundPts, dim=1) / torch.sum(roadBackGroundPts, dim=1))
            roadPlane = torch.stack([torch.zeros(roadPts3D.shape[0], device=torch.device('cuda')), torch.zeros(roadPts3D.shape[0], device=torch.device('cuda')), torch.ones(roadPts3D.shape[0], device=torch.device('cuda')), -k_ground], dim=1)
            roadPlane = roadPlane.unsqueeze(dim=1).repeat(1,self.smapleDense,1,1).view(-1, 1, 4)
            roadInterm = torch.matmul(roadPlane, roadPts3DParam)
            roadXFlat = roadX.view(-1)
            roadYFlat = roadY.view(-1)
            intepedDepth = -(roadInterm[:,0,3] / (roadInterm[:,0,0] * roadXFlat + roadInterm[:,0,1] * roadYFlat + roadInterm[:,0,2])).view(-1, self.smapleDense)
            # Please Filter out Abnormal results



            framSelector = ((roadChannle == viewInd).float() * roadBackGroundPts.float()).byte()
            roadPts3D_view = roadPts3D[framSelector, :, :].cpu().numpy()
            view2dX = roadX[framSelector].cpu().numpy().astype(np.float)
            view2dY = roadY[framSelector].cpu().numpy().astype(np.float)
            semanView = semantic[viewInd, :, :].cpu().numpy()
            semanFig = visualize_semantic(semanView).resize([dispAct.shape[3], dispAct.shape[2]], resample=Image.NEAREST)

            viewInterpedRoadDepth = intepedDepth
            viewInterpedRoad = torch.stack([roadX * viewInterpedRoadDepth, roadY * viewInterpedRoadDepth, viewInterpedRoadDepth, torch.ones_like(viewInterpedRoadDepth)], dim=2).view(-1,4,1)
            viewInterpedRoad = torch.matmul(roadPts3DParam, viewInterpedRoad).view(-1, self.smapleDense, 4, 1)
            viewInterpedRoad = viewInterpedRoad[framSelector, :, :].cpu().numpy()
            doubleCheck = torch.matmul(roadPlane[framSelector.view(-1), :], torch.Tensor(viewInterpedRoad).cuda())



            sampleDense = 10
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=6., azim=170)
            ax.dist = 4
            # ax.scatter(veh_coord2[::sampleDense, 0], veh_coord2[::sampleDense, 1], veh_coord2[::sampleDense, 2],
            #            s=0.1, c=colors2[::sampleDense, :])
            ax.scatter(roadPts3D_view[:, 0, 0], roadPts3D_view[:, 1, 0], roadPts3D_view[:, 2, 0],
                       s=0.1, c='g')
            ax.scatter(viewInterpedRoad[:, 0, 0], viewInterpedRoad[:, 1, 0], viewInterpedRoad[:, 2, 0],
                       s=0.1, c='r')
            ax.set_zlim(-10, 10)
            plt.ylim([-10, 10])
            plt.xlim([10, 16])
            set_axes_equal(ax)

            plt.imshow(semanFig)
            plt.scatter(view2dX, view2dY, c = 'c', s = 0.2)



            wallBackPts = wallTypeMask[channelInd, 0, sampledy, sampledx]
            walllBackSelection = torch.sum(wallBackPts, 1) > 20
            wallBackPts = wallBackPts[walllBackSelection, :]
            wallX = sampledx[walllBackSelection, :]
            wallY = sampledy[walllBackSelection, :]
            wallChannle = channelInd[walllBackSelection, :]
            wallDepth = realDepth[wallChannle, 0, wallY, wallX]
            wallX = wallX.float().cuda()
            wallY = wallY.float().cuda()
            wallPts3DCam = torch.stack([wallX * wallDepth, wallY * wallDepth, wallDepth, torch.ones_like(wallDepth)], dim=2).view(-1,4,1)
            wallPts3DParam = torch.inverse(torch.matmul(intrinsic[wallChannle[:,0], :, :], extrinsic[wallChannle[:,0], :, :])).unsqueeze(1).repeat(1, self.smapleDense, 1, 1).view(-1,4,4)
            wallPts3D = torch.matmul(wallPts3DParam, wallPts3DCam).view(-1, self.smapleDense, 4, 1)
            a = torch.sum((wallPts3D[:,:,0,0] * wallBackPts) ** 2, dim=1) # x.T @ x
            b = torch.sum((wallPts3D[:,:,0,0] * wallBackPts), dim=1) # x.T @ 1
            c = b # 1 @ x.T
            d = torch.sum(wallBackPts, dim=1) # 1.T @ 1
            m1 = -torch.sum((wallPts3D[:,:,0,0] * wallPts3D[:,:,1,0] * wallBackPts), dim=1) # -y.T @ x
            m2 = -torch.sum((wallPts3D[:,:,1,0] * wallBackPts), dim=1) # -y.T @ 1
            invScale = 1 / (a *d - b * c)
            normal = (torch.abs(invScale) < 1e3).float() * (torch.abs(invScale) > 1e-5).float()
            invMatrix = invScale.view(-1,1,1).repeat(1,2,2) * torch.cat([torch.stack([d, -b], dim=1).view(-1, 1, 2), torch.stack([-c, a], dim = 1).view(-1, 1, 2)], dim = 1)
            equRight = torch.stack([m1, m2], dim=1).view(-1,1,2)
            wallPlaneComp = torch.matmul(equRight, invMatrix)
            wallPlane = torch.stack([wallPlaneComp[:,0,0], torch.ones(wallBackPts.shape[0], device=torch.device("cuda")), torch.zeros(wallBackPts.shape[0], device=torch.device("cuda")), wallPlaneComp[:,0,1]], dim=1)

            # doubleCheck = torch.matmul(wallPlane, wallPts3D.view(-1,4,1))
            # Check
            # p1 = wallPlaneComp
            # p2 = torch.cat([torch.stack([a, b], dim=1).view(-1, 1, 2), torch.stack([c, d], dim = 1).view(-1, 1, 2)], dim = 1)
            # p3 = torch.stack([m1, m2], dim=1).view(-1, 1, 2)
            # newP = torch.matmul(p3, torch.inverse(p2))
            # cck = torch.abs(torch.matmul(newP, p2) - p3)[normal < 0.5]
            # wallPlaneComp = newP
            # wallPlane = torch.stack([wallPlaneComp[:,0,0], torch.ones(wallBackPts.shape[0], device=torch.device("cuda")), torch.zeros(wallBackPts.shape[0], device=torch.device("cuda")), wallPlaneComp[:,0,1]], dim=1)
            # wallPts3D_newView = wallPts3D.view(-1,4,1)
            # dist = torch.matmul(wallPlane, wallPts3D_newView).view(-1, self.smapleDense,1,1)
            # dist_l = dist[framSelector]




            # k_wall = wallPts3D
            # k_wall = (torch.sum(wallPts3D[:,:,2,0] * wallBackPts, dim=1) / torch.sum(wallBackPts, dim=1))
            # wallPlane = torch.stack([torch.zeros(wallPts3D.shape[0], device=torch.device('cuda')), torch.zeros(wallPts3D.shape[0], device=torch.device('cuda')), torch.ones(wallPts3D.shape[0], device=torch.device('cuda')), -k_wall], dim=1)
            wallPlane = wallPlane.unsqueeze(dim=1).repeat(1,self.smapleDense,1,1).view(-1, 1, 4)
            wallInterm = torch.matmul(wallPlane, wallPts3DParam)
            wallXFlat = wallX.view(-1)
            wallYFlat = wallY.view(-1)
            interpedDepth = -(wallInterm[:,0,3] / (wallInterm[:,0,0] * wallXFlat + wallInterm[:,0,1] * wallYFlat + wallInterm[:,0,2])).view(-1, self.smapleDense)
            # Please Filter out Abnormal results



            framSelector = ((wallChannle == viewInd).float() * wallBackPts.float() * normal.unsqueeze(1).repeat(1,self.smapleDense)).byte()
            wallPts3D_view = wallPts3D[framSelector, :, :].cpu().numpy()
            colorsWall = torch.Tensor(np.expand_dims(np.random.rand(wallX.shape[0], 3), axis=1).repeat(self.smapleDense, axis=1)).cuda()
            colorsWall = colorsWall[framSelector].cpu().numpy()
            view2dX = wallX[framSelector].cpu().numpy().astype(np.float)
            view2dY = wallY[framSelector].cpu().numpy().astype(np.float)
            semanView = semantic[viewInd, :, :].cpu().numpy()
            semanFig = visualize_semantic(semanView).resize([dispAct.shape[3], dispAct.shape[2]], resample=Image.NEAREST)

            viewInterpedWallDepth = interpedDepth
            viewInterpedWallDepth = torch.stack([wallX * viewInterpedWallDepth, wallY * viewInterpedWallDepth, viewInterpedWallDepth, torch.ones_like(viewInterpedWallDepth)], dim=2).view(-1,4,1)
            viewInterpedWallDepth = torch.matmul(wallPts3DParam, viewInterpedWallDepth).view(-1, self.smapleDense, 4, 1)


            tryInd = 9
            tryPts = wallPts3D[tryInd, :, :, :]
            tryPts = tryPts[wallBackPts[tryInd, :].byte(), :, :]

            planeParam = wallPlane.view(-1, self.smapleDense, 1, 4)[tryInd, :][wallBackPts[tryInd, :].byte(), :][0,:,:]
            xx = tryPts[:,0]
            yy = -(xx * planeParam[0,0] + planeParam[0,3])
            zz = tryPts[:,2]

            view3dPts = viewInterpedWallDepth[tryInd, :, :, :]
            view3dPts = view3dPts[wallBackPts[tryInd, :].byte(), :, :]
            view3dPts = view3dPts.cpu().numpy()

            camPos = torch.inverse(extrinsic[viewInd, :, :]) @ torch.Tensor([0,0,0,1]).unsqueeze(1).cuda()
            camPos = camPos.cpu().numpy()
            tryPts_draw = tryPts.cpu().numpy()
            xx_draw = xx.cpu().numpy()
            yy_draw = yy.cpu().numpy()
            zz_draw = zz.cpu().numpy()


            lindir1 = (view3dPts - np.repeat(np.expand_dims(camPos, axis=0), view3dPts.shape[0], axis=0))[:, 0:3, :]
            lindir2 = tryPts[:, 0:3, :].cpu().numpy() - np.repeat(np.expand_dims(camPos, axis=0), view3dPts.shape[0], axis=0)[:, 0:3, :]

            ptsckProjected1 = torch.matmul(extrinsic[viewInd,:,:], torch.Tensor(view3dPts).cuda())
            ptsckProjected1[:,0] = ptsckProjected1[:,0] / ptsckProjected1[:,2]
            ptsckProjected1[:, 1] = ptsckProjected1[:, 1] / ptsckProjected1[:, 2]

            ptsckProjected2 = torch.matmul(extrinsic[viewInd,:,:], tryPts)
            ptsckProjected2[:,0] = ptsckProjected2[:,0] / ptsckProjected2[:,2]
            ptsckProjected2[:, 1] = ptsckProjected2[:, 1] / ptsckProjected2[:, 2]
            ckProjection = torch.abs(ptsckProjected1[:,0] - ptsckProjected2[:,0]) + torch.abs(ptsckProjected1[:,1] - ptsckProjected2[:,1])


            linDiff = np.abs(lindir1 / lindir2)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(tryPts_draw[:, 0, 0], tryPts_draw[:, 1, 0], tryPts_draw[:, 2, 0],
                       s=0.3, c='r')
            ax.scatter(xx_draw, yy_draw, zz_draw,
                       s=0.3, c='g')
            # ax.scatter(view3dPts[:,0,0], view3dPts[:,1,0], view3dPts[:,2,0],
            #            s=0.3, c='b')
            ax.scatter(camPos[0,0], camPos[1,0], camPos[2,0],
                       s=4, c='c')
            set_axes_equal(ax)


            viewInterpedWallDepth = viewInterpedWallDepth[framSelector, :, :].cpu().numpy()
            doubleCheck = torch.matmul(wallPlane[framSelector.view(-1), :], torch.Tensor(viewInterpedWallDepth).cuda())
            ckp1 = torch.matmul(intrinsic[viewInd, :, :], torch.matmul(extrinsic[viewInd, :, :], torch.Tensor(viewInterpedWallDepth).cuda()))
            ckp1[:,0] = ckp1[:,0] / ckp1[:,2]
            ckp1[:, 1] = ckp1[:, 1] / ckp1[:, 2]

            sampleDense = 10
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=6., azim=170)
            ax.dist = 4
            ax.scatter(veh_coord2[::sampleDense, 0], veh_coord2[::sampleDense, 1], veh_coord2[::sampleDense, 2],
                       s=0.1, c=colors2[::sampleDense, :])
            ax.scatter(wallPts3D_view[:, 0, 0], wallPts3D_view[:, 1, 0], wallPts3D_view[:, 2, 0],
                       s=0.6, c=colorsWall)
            ax.scatter(viewInterpedWallDepth[:, 0, 0], viewInterpedWallDepth[:, 1, 0], viewInterpedWallDepth[:, 2, 0],
                       s=0.1, c='r')
            ax.set_zlim(-10, 10)
            plt.ylim([-10, 10])
            plt.xlim([10, 16])
            set_axes_equal(ax)

            plt.figure()
            plt.imshow(semanFig)
            plt.scatter(view2dX, view2dY, c = colorsWall, s = 0.2)

            cm = plt.get_cmap('magma')
            viewDisp = dispAct[viewInd, :, :, :].squeeze(0).detach().cpu().numpy()
            vmax = np.percentile(viewDisp, 90)
            viewDisp = (cm(viewDisp / vmax) * 255).astype(np.uint8)
            pil.fromarray(viewDisp).show()

            plt.imshow(viewDisp)
            plt.scatter(view2dX, view2dY, c = colorsWall, s = 0.2)
        """
        with torch.no_grad():
            if cts_meta is not None:
                height = 1024
                width = 2048
                xx, yy = np.meshgrid(np.arange(width), np.arange(height))
                xx = xx.flatten()
                yy = yy.flatten()
                objType = 19
                colorMap = np.zeros((objType + 1, xx.shape[0], 3), dtype=np.uint8)
                for i in range(objType):
                    if i == objType:
                        k = 255
                    else:
                        k = i
                    colorMap[i, :, :] = np.repeat(np.expand_dims(np.array(trainId2label[k].color), 0), xx.shape[0], 0)
                colorMap = colorMap.astype(np.float)
                colorMap = colorMap / 255
                semantic_small = F.interpolate(semantic.unsqueeze(1).float(), size = list(realDepth.shape[2:]), mode='nearest')

                gt_semanticMap = semantic[viewInd, :, :].cpu().numpy().flatten()
                gt_semanticMap[gt_semanticMap == 255] = 19
                colors = colorMap[gt_semanticMap, np.arange(xx.shape[0]), :]
                gtdepth = cts_meta['depthMap'][viewInd, :, :].cpu().numpy()
                mask = gtdepth > 0
                mask = mask.flatten()
                depthFlat = gtdepth.flatten()
                oneColumn = np.ones(gtdepth.shape[0] * gtdepth.shape[1])
                pixelLoc = np.stack(
                    [xx[mask] * depthFlat[mask], yy[mask] * depthFlat[mask], depthFlat[mask], oneColumn[mask]],
                    axis=1)
                intrinsic_gt = cts_meta['intrinsic'][viewInd, :, :].cpu().numpy()
                extrinsic_gt = cts_meta['extrinsic'][viewInd, :, :].cpu().numpy()
                cam_coord_gt = (np.linalg.inv(intrinsic_gt) @ pixelLoc.T).T
                veh_coord = (np.linalg.inv(extrinsic_gt) @ cam_coord_gt.T).T
                colors = colors[mask, :].astype(np.float)

                sampleDense = 100
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.view_init(elev=6., azim=170)
                ax.dist = 4
                ax.scatter(veh_coord[::sampleDense, 0], veh_coord[::sampleDense, 1], veh_coord[::sampleDense, 2], s=0.1, c = colors[::sampleDense, :])
                ax.set_zlim(-10, 10)
                plt.ylim([-10, 10])
                plt.xlim([10, 16])
                set_axes_equal(ax)

                colorMap = np.zeros((objType + 1, xx.shape[0], 3), dtype=np.uint8)
                for i in range(objType):
                    if i == objType:
                        k = 255
                    else:
                        k = i
                    colorMap[i, :, :] = np.repeat(np.expand_dims(np.array(trainId2label[k].color), 0), xx.shape[0], 0)
                colorMap = colorMap.astype(np.float)
                colorMap = colorMap / 255
                semantic_small = F.interpolate(semantic.unsqueeze(1).float(), size = list(realDepth.shape[2:]), mode='nearest')
                initrinsic_test = intrinsic[viewInd, :, :].cpu().numpy()
                extrinsic_test = extrinsic[viewInd, :, :].cpu().numpy()
                seman_test = semantic_small[viewInd, :, :].cpu().numpy().flatten()
                seman_test[seman_test == 255] = 19
                seman_test = seman_test.astype(np.int)
                height = realDepth.shape[2]
                width = realDepth.shape[3]
                xx, yy = np.meshgrid(np.arange(width), np.arange(height))
                xx = xx.flatten()
                yy = yy.flatten()
                predDepth = realDepth[viewInd, 0,:,:].cpu().numpy().flatten()
                oneColumn = np.ones(height * width)
                pixelLoc = np.stack(
                    [xx * predDepth, yy * predDepth, predDepth, oneColumn],
                    axis=1)
                cam_coord_gt2 = (np.linalg.inv(initrinsic_test) @ pixelLoc.T).T
                veh_coord2 = (np.linalg.inv(extrinsic_test) @ cam_coord_gt2.T).T
                colors2 = colorMap[seman_test, np.arange(xx.shape[0]), :].astype(np.float)

                sampleDense = 10
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.view_init(elev=6., azim=170)
                ax.dist = 4
                ax.scatter(veh_coord2[::sampleDense, 0], veh_coord2[::sampleDense, 1], veh_coord2[::sampleDense, 2], s=0.1, c = colors2[::sampleDense, :])
                # ax.scatter(veh_coord[::sampleDense, 0], veh_coord[::sampleDense, 1], veh_coord[::sampleDense, 2], s=0.1,
                #            c='r')
                ax.set_zlim(-10, 10)
                plt.ylim([-10, 10])
                plt.xlim([10, 16])
                set_axes_equal(ax)

                sampleDense = 10
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.view_init(elev=6., azim=170)
                ax.dist = 4
                ax.scatter(veh_coord2[::sampleDense, 0], veh_coord2[::sampleDense, 1], veh_coord2[::sampleDense, 2], s=0.1, c = 'r')
                ax.scatter(veh_coord[::sampleDense, 0], veh_coord[::sampleDense, 1], veh_coord[::sampleDense, 2], s=0.1,
                           c='g')
                ax.set_zlim(-10, 10)
                plt.ylim([-10, 10])
                plt.xlim([10, 16])
                set_axes_equal(ax)


            semanView = semantic[viewInd, :, :].cpu().numpy()
            semanFig = visualize_semantic(semanView).resize([dispAct.shape[3], dispAct.shape[2]], resample=Image.NEAREST)



            # tdevice = torch.device("cuda")

            maskGrad = torch.abs(self.seman_convx(foredgroundMask)) + torch.abs(self.seman_convy(foredgroundMask))
            maskGrad = self.expand_road(maskGrad) * (dispAct > 7e-3).float()
            maskGrad = torch.clamp(maskGrad, min=3) - 3
            maskGrad[:, :, self.zeroArea, :] = 0
            maskGrad[:, :, :, self.zeroArea] = 0

            cm = plt.get_cmap('magma')
            viewMaskGrad = maskGrad[viewInd, :, :, :].squeeze(0).detach().cpu().numpy()
            vmax = np.percentile(viewMaskGrad, 99.5)
            viewMaskGrad = (cm(viewMaskGrad / vmax) * 255).astype(np.uint8)
            pil.fromarray(viewMaskGrad).show()

            # height = dispAct.shape[2]
            # width = dispAct.shape[3]
            # centerx = torch.LongTensor(self.ptsNum * self.batchNum).random_(self.wdSize, width - self.wdSize)
            # centery = torch.LongTensor(self.ptsNum * self.batchNum).random_(self.wdSize, height - self.wdSize)
            # onBorderSelection = maskGrad[self.channelInd, 0, centery, centerx] > 1e-1
            # centery = centery[onBorderSelection].unsqueeze(1).repeat([1, self.smapleDense])
            # centerx = centerx[onBorderSelection].unsqueeze(1).repeat([1, self.smapleDense])
            # channelInd = self.channelInd[onBorderSelection].unsqueeze(1).repeat([1, self.smapleDense])
            # validNum = centerx.shape[0]
            # bx = torch.LongTensor(validNum * self.smapleDense).random_(-self.wdSize, self.wdSize + 1).view(validNum, self.smapleDense)
            # by = torch.LongTensor(validNum * self.smapleDense).random_(-7, 8).view(validNum, self.smapleDense)
            # sampledx = centerx + bx
            # sampledy = centery + by

            centerx = torch.LongTensor(self.ptsNum * self.batchNum * 100).random_(self.wdSize, self.width - self.wdSize).cuda()
            centery = torch.LongTensor(self.ptsNum * self.batchNum * 100).random_(self.wdSize, self.height - self.wdSize).cuda()
            onBorderSelection = maskGrad[self.channelInd, 0, centery, centerx] > 1e-1
            centerx = centerx[onBorderSelection]
            centery = centery[onBorderSelection]
            channelInd = self.channelInd[onBorderSelection]

            roadBackGroundPts = groundTypeMask[channelInd, 0, centery, centerx].byte()
            onBorderSelection[onBorderSelection] = roadBackGroundPts
            channelInd = channelInd[roadBackGroundPts]
            centerx = centerx[roadBackGroundPts]
            centery = centery[roadBackGroundPts]

            # Road
            realDepth_flat = realDepth.view(-1)
            roadPts3DCam = torch.stack([self.xx * realDepth_flat, self.yy * realDepth_flat, realDepth.view(-1), self.ones], dim=1).view(self.batchNum, -1, 4).permute(0,2,1)
            roadPts3DParam = torch.matmul(intrinsic, extrinsic)
            roadPts3DParam_inv = torch.inverse(roadPts3DParam)
            roadPts3DCam = torch.matmul(roadPts3DParam_inv, roadPts3DCam)
            roadPts3DCam = roadPts3DCam.view(self.batchNum, 4, self.height, self.width)

            # roadPts3DCam_view = roadPts3DCam[viewInd, :, :].permute(1,0).cpu().numpy()
            # sampleDense = 10
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # ax.view_init(elev=6., azim=170)
            # ax.dist = 4
            # ax.scatter(roadPts3DCam_view[::sampleDense, 0], roadPts3DCam_view[::sampleDense, 1], roadPts3DCam_view[::sampleDense, 2], s=0.1, c = 'r')
            # ax.scatter(veh_coord[::sampleDense, 0], veh_coord[::sampleDense, 1], veh_coord[::sampleDense, 2], s=0.1, c='g')
            # ax.set_zlim(-10, 10)
            # plt.ylim([-10, 10])
            # plt.xlim([10, 16])
            # set_axes_equal(ax)

            # roadPts3DCam = roadPts3DCam.view(self.batchNum, 4, self.height, self.width)
            # roadPts3DCam_view = roadPts3DCam[viewInd, :, groundTypeMask[viewInd, 0, :, :].byte()].permute(1,0).cpu().numpy()
            # sampleDense = 10
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # ax.view_init(elev=6., azim=170)
            # ax.dist = 4
            # ax.scatter(roadPts3DCam_view[::sampleDense, 0], roadPts3DCam_view[::sampleDense, 1], roadPts3DCam_view[::sampleDense, 2], s=0.1, c = 'r')
            # ax.scatter(veh_coord[::sampleDense, 0], veh_coord[::sampleDense, 1], veh_coord[::sampleDense, 2], s=0.1, c='g')
            # ax.set_zlim(-10, 10)
            # plt.ylim([-10, 10])
            # plt.xlim([10, 16])
            # set_axes_equal(ax)

            planeParamEst = torch.ones(self.batchNum, 1, 4, device=self.tdevice)
            for i in range(self.batchNum):
                # roadPts = roadPts3DCam[i, :, groundTypeMask[viewInd, 0, :, :].byte()][0:3, :]
                # for j in range(4):
                #     meanShift = torch.mean(roadPts, dim=1, keepdim=True)
                #     roadPts_shifted = roadPts - meanShift
                #     roadPts_shiftedM = torch.matmul(roadPts_shifted, roadPts_shifted.t())
                #     e, v = torch.eig(roadPts_shiftedM, eigenvectors = True)
                #     planeParam_shifted = torch.cat([v[:,2], torch.Tensor([0]).cuda()])
                #     transM = torch.eye(4, device=self.tdevice)
                #     transM[0:3,3] = -meanShift[:, 0]
                #     planeParam = planeParam_shifted @ transM
                #     dist = torch.abs(planeParam_shifted @ torch.cat([roadPts_shifted, torch.ones(1, roadPts_shifted.shape[1], device=self.tdevice)], dim=0))
                #     sel = dist < 0.7
                #     roadPts = roadPts[:, sel]
                roadPts = roadPts3DCam[i, :, groundTypeMask[viewInd, 0, :, :].byte()][0:3, :]
                for j in range(4):
                    meanz = torch.mean(roadPts[2,:])
                    selected = torch.abs(roadPts[2,:] - meanz) < 0.5
                    roadPts = roadPts[:,selected]
                planeParam = torch.Tensor([0,0,1,-meanz]).cuda()
                planeParamEst[i, 0, :] = planeParam

            oldPts = roadPts3DCam[channelInd, :, centery, centerx]
            depthOld = realDepth[channelInd, 0, centery, centerx]
            selector1 = depthOld > 0.3
            depthOld = torch.clamp(depthOld, min=0.3)
            M = torch.matmul(planeParamEst, roadPts3DParam_inv)
            depthNew = - (M[channelInd, 0, 3] / (
                        M[channelInd, 0, 0] * centerx.float() + M[channelInd, 0, 1] * centery.float() + M[
                    channelInd, 0, 2]))
            newPts = torch.stack([centerx.float() * depthNew, centery.float() * depthNew, depthNew, torch.ones_like(depthNew)], dim=1).unsqueeze(2)
            newPts = torch.matmul(roadPts3DParam_inv[channelInd, :, :], newPts)
            selector2 = (depthNew > 0.3) * (depthNew < self.maxDepth)
            depthNew = torch.clamp(depthNew, min=0.3, max=self.maxDepth)
            selector3 = (depthNew - depthOld) > 0
            lossRoad = torch.mean(torch.clamp(depthNew.detach() - depthOld, min=0)) * 1e-3
            selector = selector1 * selector2 * selector3

            # depthOld = torch.clamp(realDepth[channelInd, 0, centery, centerx], min = 0.3, max=100)
            # M = torch.matmul(planeParamEst, roadPts3DParam_inv)
            # depthNew = - (M[channelInd, 0, 3] / (M[channelInd, 0, 0] * centerx.float() + M[channelInd, 0, 1] * centery.float() + M[channelInd,0,2]))
            # depthNew = torch.clamp(depthNew, min = 0.3, max= 30)

            # lossRoad = torch.mean(torch.clamp(depthNew.detach() - depthOld, min=0)) * 1e-3
            # lossGround = torch.mean(torch.clamp(depthNew.detach() - depthOld, min=0))




            # oldPts = roadPts3DCam[channelInd, :, centery, centerx]
            # newPts = oldPts.clone()
            # newPts[:,2] = -planeParamEst[channelInd, 0, 3]

            # depthNew = torch.clamp(torch.matmul(roadPts3DParam[channelInd, :, :], newPts.unsqueeze(2))[:,2,0], min = 0, max = 100)
            # depthOld = torch.clamp(realDepth[channelInd, 0, centery, centerx], max = 100)
            # diff = torch.abs(depthOld - depthNew)
            # a = diff.cpu().numpy()
            # loss = torch.mean(torch.clamp(depthOld - depthNew, min=0))
            viewDense = 10
            roadPtsView = roadPts3DCam[viewInd, :, :, :].permute(1,2,0)[groundTypeMask[viewInd, 0, :, :].byte(), :].cpu().numpy()
            oldPts_view = oldPts[(channelInd == viewInd) * selector, :].cpu().numpy()
            newPts_view = newPts[(channelInd == viewInd) * selector, :].cpu().numpy()
            centerx_view = centerx[(channelInd == viewInd) * selector].cpu().numpy()
            centery_view = centery[(channelInd == viewInd) * selector].cpu().numpy()
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=6., azim=170)
            ax.dist = 4
            # ax.scatter(veh_coord2[::sampleDense, 0], veh_coord2[::sampleDense, 1], veh_coord2[::sampleDense, 2], s=0.1, c=colors2[::sampleDense, :])
            ax.scatter(oldPts_view[:, 0], oldPts_view[:, 1], oldPts_view[:, 2], s=0.3, c='r')
            ax.scatter(roadPtsView[::viewDense, 0], roadPtsView[::viewDense, 1], roadPtsView[::viewDense, 2], s=0.3, c='g')
            ax.scatter(newPts_view[:, 0], newPts_view[:, 1], newPts_view[:, 2], s=0.3, c='c')
            ax.set_zlim(-10, 10)
            plt.ylim([-10, 10])
            plt.xlim([10, 16])
            set_axes_equal(ax)

            plt.figure()
            plt.imshow(semanFig)
            plt.scatter(centerx_view, centery_view, s=0.3, c='c')

            cm = plt.get_cmap('magma')
            viewDisp = dispAct[viewInd, :, :, :].squeeze(0).detach().cpu().numpy()
            vmax = np.percentile(viewDisp, 90)
            viewDisp = (cm(viewDisp / vmax) * 255).astype(np.uint8)
            plt.figure()
            plt.imshow(viewDisp)
            plt.scatter(centerx_view, centery_view, s=0.3, c='c')
            # pil.fromarray(viewDisp).show()


            # Wall
            height = dispAct.shape[2]
            width = dispAct.shape[3]
            maskGrad = torch.abs(self.seman_convx(foredgroundMask)) + torch.abs(self.seman_convy(foredgroundMask))
            maskGrad = self.expand(maskGrad) * (dispAct > 1e-2).float()
            maskGrad = torch.clamp(maskGrad, min=3) - 3
            maskGrad[:, :, self.zeroArea, :] = 0
            maskGrad[:, :, :, self.zeroArea] = 0
            maskGrad = self.expand(maskGrad)

            centerx = torch.LongTensor(self.ptsNum * self.batchNum * 100).random_(self.wdSize, width - self.wdSize)
            centery = torch.LongTensor(self.ptsNum * self.batchNum * 100).random_(self.wdSize, height - self.wdSize)
            # maskGrad = self.expand(maskGrad)

            onBorderSelection = (maskGrad[self.channelInd_wall, 0, centery, centerx] > 1e-1) * (
                        wallTypeMask[self.channelInd_wall, 0, centery, centerx] == 1)
            centery = centery[onBorderSelection]
            centerx = centerx[onBorderSelection]
            channelInd = self.channelInd_wall[onBorderSelection]

            validNum = centerx.shape[0]
            bx = torch.LongTensor(validNum).random_(-self.wdSize, self.wdSize + 1)
            by = torch.LongTensor(validNum).random_(-7, 8)

            sampledx_pair = centerx + bx
            sampledy_pair = centery + by

            onBackSelection = wallTypeMask[channelInd, 0, sampledy_pair, sampledx_pair] == 1
            onBorderSelection[onBorderSelection] = onBackSelection
            centery = centery[onBackSelection]
            centerx = centerx[onBackSelection]
            channelInd = channelInd[onBackSelection]
            sampledx_pair = sampledx_pair[onBackSelection]
            sampledy_pair = sampledy_pair[onBackSelection]
            sourceDisp = dispAct[channelInd, 0, centery, centerx]
            targetDisp = dispAct[channelInd, 0, sampledy_pair, sampledx_pair]
            errSel = ((sourceDisp / targetDisp > 1.15) * (sourceDisp > 1e-2)).float()
            lossWall = torch.sum(torch.clamp(sourceDisp - targetDisp.detach(), min=0) * errSel) / torch.sum(errSel)

            viewSel = channelInd == viewInd
            viewcx = centerx[viewSel].cpu().numpy()
            viewcy = centery[viewSel].cpu().numpy()
            fig = plt.figure()
            plt.imshow(semanFig)
            plt.scatter(viewcx, viewcy, c = 'c', s = 0.5)



            cm = plt.get_cmap('magma')
            viewDisp = dispAct[viewInd, :, :, :].squeeze(0).detach().cpu().numpy()
            vmax = np.percentile(viewDisp, 90)
            viewDisp = (cm(viewDisp / vmax) * 255).astype(np.uint8)
            fig = plt.figure()
            plt.imshow(viewDisp)
            plt.scatter(viewcx, viewcy, c = 'c', s = 0.5)


            fig, ax = plt.subplots()
            plt.imshow(viewDisp)
            curChannelPosPts = (channelInd == viewInd)
            curChannelPosPts = curChannelPosPts.float() * errSel
            curChannelPosPts[::2] = 0
            curChannelPosPts = curChannelPosPts.byte()
            ptsSet1 = np.expand_dims(np.stack([centerx[curChannelPosPts], centery[curChannelPosPts]], axis=1), axis=1)
            ptsSet2 = np.expand_dims(np.stack([sampledx_pair[curChannelPosPts], sampledy_pair[curChannelPosPts]], axis=1), axis=1)
            ptsSet = np.concatenate([ptsSet1, ptsSet2], axis=1)
            ln_coll = LineCollection(ptsSet, colors='c', linewidths=0.6)
            ax.add_collection(ln_coll)
            ax.set_title('Line collection with masked arrays')
            plt.show()


            """
            # Wall part starts
            lossDepthWall = 0
            lossRegXWall = 0
            lossRegYWall = 0

            wallBackGroundPts = wallTypeMask[channelInd, 0, sampledy, sampledx]
            wallBackSelection = torch.sum(wallBackGroundPts, 1) > 20
            # if torch.sum(wallBackSelection) < 20:
            #     return
            wallBackGroundPts = wallBackGroundPts[wallBackSelection, :]
            wallX = sampledx[wallBackSelection, :]
            wallY = sampledy[wallBackSelection, :]
            wallChannle = channelInd[wallBackSelection, :]
            wallDepth = realDepth[wallChannle, 0, wallY, wallX]
            wallX = wallX.float().cuda()
            wallY = wallY.float().cuda()
            wallPts3DCam = torch.stack([wallX * wallDepth, wallY * wallDepth, wallDepth, torch.ones_like(wallDepth)], dim=2).view(-1,4,1)
            wallPts3DParam = torch.matmul(intrinsic, extrinsic)
            wallPts3DParam_inv = torch.inverse(wallPts3DParam)
            wallPts3DParam = wallPts3DParam[wallChannle, :, :].view(-1,4,4)
            wallPts3DParam_inv = wallPts3DParam_inv[wallChannle, :, :].view(-1,4,4)
            wallPts3D = torch.matmul(wallPts3DParam_inv, wallPts3DCam).view(-1, self.smapleDense, 4, 1)



            a = torch.sum((wallPts3D[:,:,1,0] * wallBackGroundPts) ** 2, dim=1) # y.T @ y
            b = torch.sum((wallPts3D[:,:,1,0] * wallBackGroundPts), dim=1) # y.T @ 1
            c = b # 1 @ y.T
            d = torch.sum(wallBackGroundPts, dim=1) # 1.T @ 1
            m1 = torch.sum((wallPts3D[:,:,1,0] * wallPts3D[:,:,0,0] * wallBackGroundPts), dim=1) # -x.T @ y
            m2 = torch.sum((wallPts3D[:,:,0,0] * wallBackGroundPts), dim=1) # -x.T @ 1
            invScale = 1 / (a *d - b * c)
            normal = (torch.abs(invScale) < 1e4).float()
            invMatrix = invScale.view(-1,1,1).repeat(1,2,2) * torch.cat([torch.stack([d, -b], dim=1).view(-1, 1, 2), torch.stack([-c, a], dim = 1).view(-1, 1, 2)], dim = 1)
            equRight = torch.stack([m1, m2], dim=1).view(-1,1,2)
            wallPlaneComp = torch.matmul(equRight, invMatrix)
            # equLeft = torch.cat([torch.stack([a, b], dim=1).view(-1, 1, 2), torch.stack([c, d], dim = 1).view(-1, 1, 2)], dim = 1)
            # equRight = torch.stack([m1, m2], dim=1).view(-1, 1, 2)
            # wallPlaneComp = torch.matmul(equRight, torch.inverse(equLeft))
            wallPlane = torch.stack([-torch.ones(wallBackGroundPts.shape[0], device=tdevice), wallPlaneComp[:,0,0], torch.zeros(wallBackGroundPts.shape[0], device=tdevice), wallPlaneComp[:,0,1]], dim=1).unsqueeze(1).unsqueeze(1).repeat(1,self.smapleDense,1,1).view(-1,1,4)
            wallPlane = wallPlane / torch.norm(wallPlane[:,:,0:3], dim=2, keepdim=True).repeat(1,1,4)


            # wallPlane_normalized = wallPlane / torch.norm(wallPlane, dim=2).unsqueeze(2).repeat(1, 1, 4)
            ptsOnPlane = torch.stack([(wallPlane[:,0,1] + wallPlane[:,0,3]) / (-wallPlane[:,0,0]), torch.ones(wallPlane.shape[0], device=tdevice), torch.zeros(wallPlane.shape[0], device=tdevice), torch.ones(wallPlane.shape[0], device=tdevice)], dim=1).unsqueeze(dim = 2)
            wallPts3D_flat = wallPts3D.view(-1, 4, 1)
            ptsFromOrgToIntrest = (wallPts3D_flat - ptsOnPlane)[:, 0:3, 0]
            dirLength = torch.sum(wallPlane[:, 0, 0:3] * ptsFromOrgToIntrest, dim = 1)
            # dirLength = ptsFromOrgToIntrest[:, 2]

            wallPtsProjectedToPlane = ptsFromOrgToIntrest - (dirLength.unsqueeze(1).repeat(1,3) * wallPlane[:,0,0:3])
            wallPtsProjectedToPlane = ptsOnPlane[:,0:3,0] + wallPtsProjectedToPlane
            # roadPtsProjectedToPlane = roadPtsProjectedToPlane.view(-1, self.smapleDense, 3, 1)
            wallPtsProjectedToPlane = torch.cat([wallPtsProjectedToPlane, torch.ones([wallPtsProjectedToPlane.shape[0],1], device=tdevice)], dim=1)
            wallPtsProjectedToPlane = wallPtsProjectedToPlane.view(-1,4,1)


            wallProjectedNew = torch.matmul(wallPts3DParam, wallPtsProjectedToPlane)
            wallXNew = wallProjectedNew[:,0,0] / wallProjectedNew[:,2,0]
            wallYNew = wallProjectedNew[:, 1, 0] / wallProjectedNew[:, 2,0]
            wallDepthNew = wallProjectedNew[:,2,0]

            wallBackGroundPts = wallBackGroundPts.byte()
            wallDepthOld = wallDepth[wallBackGroundPts]
            wallXOld = wallX[wallBackGroundPts]
            wallYOld = wallY[wallBackGroundPts]

            wallBackGroundPts = wallBackGroundPts.view(-1)
            wallDepthNew = wallDepthNew[wallBackGroundPts]
            wallXNew = wallXNew[wallBackGroundPts]
            wallYNew = wallYNew[wallBackGroundPts]

            # lossDepthWall = torch.mean((wallDepthOld - wallDepthNew)**2 * normal)
            # lossRegXWall = torch.mean((wallXOld - wallXNew)**2 * normal)
            # lossRegYWall = torch.mean((wallYOld - wallYNew)**2 * normal)




            # Check
            ptsck = wallPtsProjectedToPlane
            ptsck = torch.abs(torch.matmul(wallPlane, ptsck))
            ptsck = ptsck[:,0,0][wallBackGroundPts]


            framSelector = ((wallChannle == viewInd).float() * wallBackGroundPts.view(-1, self.smapleDense).float()).byte()
            wallPts3D_view = wallPts3D[framSelector, :, :].cpu().numpy()
            view2dX = wallX[framSelector].cpu().numpy().astype(np.float)
            view2dY = wallY[framSelector].cpu().numpy().astype(np.float)
            semanView = semantic[viewInd, :, :].cpu().numpy()
            semanFig = visualize_semantic(semanView).resize([dispAct.shape[3], dispAct.shape[2]], resample=Image.NEAREST)

            wallColors = torch.rand([framSelector.shape[0], 3], device=tdevice).unsqueeze(1).repeat(1,200,1).view(-1,3)[framSelector.view(-1), :].cpu().numpy()
            viewInterpedWall = wallPtsProjectedToPlane
            viewInterpedWall = viewInterpedWall[framSelector.view(-1), :, :].cpu().numpy()



            subViewInd = 3
            pts = wallPts3D[subViewInd, :, 0:3, 0]
            pts = pts[framSelector[subViewInd, :], :].cpu().numpy()
            ptsInterped = wallPtsProjectedToPlane.view(-1,self.smapleDense, 4, 1)
            ptsInterped = ptsInterped[subViewInd, framSelector[subViewInd, :], :, 0].cpu().numpy()
            pxView = wallX[subViewInd, framSelector[subViewInd, :]].cpu().numpy()
            pyView = wallY[subViewInd, framSelector[subViewInd, :]].cpu().numpy()
            # fig, axs = plt.subplots(1, 1)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=6., azim=170)
            ax.dist = 4
            ax.scatter(veh_coord2[::sampleDense, 0], veh_coord2[::sampleDense, 1], veh_coord2[::sampleDense, 2],
                       s=0.1, c=colors2[::sampleDense, :])
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, c='c')
            ax.scatter(ptsInterped[:, 0], ptsInterped[:, 1], ptsInterped[:, 2], s=3, c='g')
            ax.set_zlim(-10, 10)
            plt.ylim([-10, 10])
            plt.xlim([10, 16])
            set_axes_equal(ax)
            plt.xlabel('x')
            plt.savefig('1.png', dpi = 200)

            cm = plt.get_cmap('magma')
            viewDisp = dispAct[viewInd, :, :, :].squeeze(0).detach().cpu().numpy()
            vmax = np.percentile(viewDisp, 90)
            viewDisp = (cm(viewDisp / vmax) * 255).astype(np.uint8)
            # pil.fromarray(viewDisp).show()

            fig = plt.figure()
            plt.imshow(semanFig)
            plt.scatter(pxView, pyView, c = 'c', s = 0.2)
            plt.savefig('2.png', dpi=200)

            fig = plt.figure()
            plt.imshow(viewDisp)
            plt.scatter(pxView, pyView, c = 'c', s = 0.2)
            plt.savefig('3.png', dpi=200)

            fig, ax = plt.subplots()
            ax.scatter(pts[:,0], pts[:,1], c = 'r')
            ax.scatter(ptsInterped[:, 0], ptsInterped[:, 1], c = 'g')
            plt.savefig('4.png', dpi=200)

            # sampleDense = 10
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=6., azim=170)
            ax.dist = 4
            ax.scatter(veh_coord2[::sampleDense, 0], veh_coord2[::sampleDense, 1], veh_coord2[::sampleDense, 2],
                       s=0.1, c=colors2[::sampleDense, :])
            ax.scatter(wallPts3D_view[  :, 0, 0], wallPts3D_view[:, 1, 0], wallPts3D_view[:, 2, 0],
                       s=0.3, c=wallColors)


            ax.set_zlim(-10, 10)
            plt.ylim([-10, 10])
            plt.xlim([10, 16])
            set_axes_equal(ax)
            plt.xlabel('x')

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=6., azim=170)
            ax.dist = 4
            # ax.scatter(veh_coord2[::sampleDense, 0], veh_coord2[::sampleDense, 1], veh_coord2[::sampleDense, 2],
            #            s=0.1, c=colors2[::sampleDense, :])
            ax.scatter(viewInterpedWall[:, 0, 0], viewInterpedWall[:, 1, 0], viewInterpedWall[:, 2, 0],
                       s=0.7, c='r')
            ax.scatter(wallPts3D_view[:, 0, 0], wallPts3D_view[:, 1, 0], wallPts3D_view[:, 2, 0],
                       s=0.3, c='c')
            ax.set_zlim(-10, 10)
            plt.ylim([-10, 10])
            plt.xlim([10, 16])
            set_axes_equal(ax)


            fig = plt.figure()
            plt.imshow(semanFig)
            plt.scatter(view2dX, view2dY, c = 'c', s = 0.2)
            # plt.savefig("1.png", dpi = 200)
            """