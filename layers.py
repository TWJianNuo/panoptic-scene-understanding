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
        height = inputs[('color', 0, 0)].shape[2]
        width = inputs[('color', 0, 0)].shape[3]
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
    def forward(self, inputs, outputs):
        # height = inputs['seman_gt'].shape[2]
        # width = inputs['seman_gt'].shape[3]
        label = inputs['seman_gt']
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
        img = surfacecolor[viewindex, :, :, :].permute(1,2,0).cpu().numpy()
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
        self.conv.bias = nn.Parameter(torch.arange(self.maxDisp).type(torch.FloatTensor), requires_grad=False)
        self.conv.weight = nn.Parameter(convweights, requires_grad=False)

        # convweights_opp = torch.flip(convweights, dims=[1])
        # self.conv_opp = torch.nn.Conv2d(in_channels=1, out_channels=self.maxDisp, kernel_size=(3,self.maxDisp + 2), stride=1, padding=self.pad, bias=False)
        # self.conv_opp.bias = nn.Parameter(torch.arange(self.maxDisp).type(torch.FloatTensor), requires_grad=False)
        # self.conv_opp.weight = nn.Parameter(convweights_opp, requires_grad=False)

        # self.weightck = (torch.sum(torch.abs(self.conv.weight)) + torch.sum(torch.abs(self.conv.bias)))
        # self.gausconv = get_gaussian_kernel(channels = 1, padding = 1)
        # self.gausconv.cuda()
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
            return mask
    def computeMask(self, dispmap, direction):
        with torch.no_grad():
            width = dispmap.shape[3]
            if direction == 'l':
                output = self.conv(dispmap)
                output = torch.min(output, dim=1, keepdim=True)[0]
                output = output[:,:,self.pad-1:-(self.pad-1):,-width:]
                mask = torch.tanh(-output * self.boostfac)
                mask = mask.masked_fill(mask < 0.9, 0)
                # mask = mask.masked_fill(mask < 0, 0)
            elif direction == 'r':
                dispmap_opp = torch.flip(dispmap, dims=[3])
                output_opp = self.conv(dispmap_opp)
                output_opp = torch.min(output_opp, dim=1, keepdim=True)[0]
                output_opp = output_opp[:, :, self.pad - 1:-(self.pad - 1):, -width:]
                mask = torch.tanh(-output_opp * self.boostfac)
                mask = mask.masked_fill(mask < 0.9, 0)
                # mask = mask.masked_fill(mask < 0, 0)
                mask = torch.flip(mask, dims=[3])
            return mask
    def visualize(self, dispmap, viewind = 0):
        cm = plt.get_cmap('magma')

        # pad = int(self.maxDisp + 2 -1) / 2
        # dispmap = self.gausconv(dispmap)
        # height = dispmap.shape[2]
        width = dispmap.shape[3]
        output = self.conv(dispmap)
        output = torch.min(output, dim=1, keepdim=True)[0]
        output = output[:,:,self.pad-1:-(self.pad-1):,-width:]
        # output = output[:,:,pad:-pad, pad:-pad]
        mask = torch.tanh(-output * self.boostfac)
        # mask = torch.clamp(mask, min=0)
        mask = mask.masked_fill(mask < 0.9, 0)

        dispmap_opp = torch.flip(dispmap, dims=[3])
        output_opp = self.conv(dispmap_opp)
        output_opp = torch.min(output_opp, dim=1, keepdim=True)[0]
        output_opp = output_opp[:,:,self.pad-1:-(self.pad-1):,-width:]
        # output = output[:,:,pad:-pad, pad:-pad]
        mask_opp = torch.tanh(-output_opp * self.boostfac)
        # mask_opp = torch.clamp(mask_opp, min=0)
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

        dispmap = dispmap * (1 - mask)
        viewdisp = dispmap[viewind, 0, :, :].detach().cpu().numpy()
        vmax = np.percentile(viewdisp, 90)
        viewdisp = (cm(viewdisp / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewdisp).show()

        # viewdisp_opp = dispmap_opp[viewind, 0, :, :].detach().cpu().numpy()
        # vmax = np.percentile(viewdisp_opp, 90)
        # viewdisp_opp = (cm(viewdisp_opp / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewdisp_opp).show()
        return pil.fromarray(viewmask), pil.fromarray(viewdisp)


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
        self.ptsNum = 5000 # Each image generate 50000 number of points
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

        posNum = torch.sum(objType, dim=1, keepdim=True)
        negNum = torch.sum(1-objType, dim=1, keepdim=True)

        posMean = (torch.sum(objDisp * objType, dim=1, keepdim=True) / posNum)
        negMean = (torch.sum(objDisp * (1-objType), dim=1, keepdim=True) / negNum)
        lossSimPos = torch.mean(torch.sqrt(torch.sum((objDisp - posMean)**2 * objType, dim=1, keepdim=True) / posNum + 1e-14))
        lossSimNeg = torch.mean(torch.sqrt(torch.sum((objDisp - negMean)**2 * (1-objType), dim=1, keepdim=True) / negNum + 1e-14))
        lossSim = (lossSimPos + lossSimNeg) / 2
        lossContrast = torch.mean(negMean - posMean) + 0.02
        # assert not torch.isnan(lossContrast) and not torch.isnan(lossSim), "nan occur"
        # if torch.isnan(lossContrast) or torch.isnan(lossSim):
        #     lossContrast = 0
        #     lossSim = 0
        return lossSim, lossContrast
    def visualize_randomSample(self, disp, foredgroundMask, suppresMask = None, viewIndex = 0):
        # maskGrad = torch.abs(self.seman_convx(foredgroundMask)) + torch.abs(self.seman_convy(foredgroundMask))
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
        sampleInter = 10
        colors = torch.rand((sampledx.shape[0], 1)).repeat(1, self.smapleDense)
        curFrameSel = channelInd[:,0] == viewIndex
        scX = sampledx[curFrameSel]
        scY = sampledy[curFrameSel]
        scC = colors[curFrameSel]
        plt.imshow(viewDisp[:,:,0:3])
        ax = plt.gca()
        plt.scatter(scX[::sampleInter,:].contiguous().view(-1).cpu().numpy(), scY[::sampleInter,:].contiguous().view(-1).cpu().numpy(), c = scC[::sampleInter,:].contiguous().view(-1).cpu().numpy(), s = 0.6)
        plt.close()

        # sampleInter = 10
        scCx = centerx[curFrameSel]
        scCy = centery[curFrameSel]
        plt.imshow(viewMaskGrad[:,:,0:3])
        plt.scatter(scX[::sampleInter,:].contiguous().view(-1).cpu().numpy(), scY[::sampleInter,:].contiguous().view(-1).cpu().numpy(), c = scC[::sampleInter,:].contiguous().view(-1).cpu().numpy(), s = 0.6)
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
