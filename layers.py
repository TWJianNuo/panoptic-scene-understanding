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

        self.convx.cuda()
        self.convy.cuda()


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
    def __init__(self, scale = 0, windowsize = 3, inchannel = 3):
        super(varLoss, self).__init__()
        assert windowsize % 2 != 0, "pls input odd kernel size"
        self.scale = scale
        self.windowsize = windowsize
        self.inchannel = inchannel
        self.initkernel()
    def initkernel(self):
        # kernel is for mean value calculation
        weights = torch.ones((self.inchannel, 1, self.windowsize, self.windowsize))
        weights = weights / (self.windowsize * self.windowsize)
        self.conv = nn.Conv2d(in_channels=self.inchannel, out_channels=self.inchannel, kernel_size=self.windowsize, padding=0, bias=False, groups=self.inchannel)
        self.conv.weight = nn.Parameter(weights, requires_grad=False)
        self.conv.cuda()
    def forward(self, input):
        pad = int((self.windowsize - 1) / 2)
        scaled = input[:,:, pad : -pad, pad : -pad] - self.conv(input)
        loss = torch.mean(scaled * scaled) * self.scale
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

class ObjRegularization(nn.Module):
    # do object type wise regularization
    # suppose we have channel wise object type
    def __init__(self):
        super(ObjRegularization, self).__init__()
        regularizer = dict()
        for id in trainId2label:
            entry = trainId2label[id]
            if entry.trainId >= 0 and entry.trainId != 255:
                if entry.name == 'traffic sign':
                    regularizer[entry.name] = varLoss(scale=1, windowsize = 7, inchannel = 3)

    def regularize(self, tensor):
        a = 1


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
    def forward(self, dispmap, side):
        # dispmap = self.gausconv(dispmap)

        # assert torch.abs(self.weightck - (torch.sum(torch.abs(self.conv.weight)) + torch.sum(torch.abs(self.conv.bias)))) < 1e-2, "weights changed"
        with torch.no_grad():
            if side == 1:
                width = dispmap.shape[3]
                output = self.conv(dispmap)
                output = torch.min(output, dim=1, keepdim=True)[0]
                output = output[:,:,self.pad-1:-(self.pad-1):,-width:]
                mask = torch.tanh(-output * self.boostfac)
                # mask = torch.clamp(mask, min=0)
                mask = mask.masked_fill(mask < 0.9, 0)
            elif side == -1:
                width = dispmap.shape[3]
                dispmap_opp = torch.flip(dispmap, dims=[3])
                output_opp = self.conv(dispmap_opp)
                output_opp = torch.min(output_opp, dim=1, keepdim=True)[0]
                output_opp = output_opp[:, :, self.pad - 1:-(self.pad - 1):, -width:]
                mask = torch.tanh(-output_opp * self.boostfac)
                # mask = torch.clamp(mask, min=0)
                mask = mask.masked_fill(mask < 0.9, 0)
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
        return pil.fromarray(viewmask), pil.fromarray(viewmask_opp)


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

