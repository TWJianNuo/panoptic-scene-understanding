# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import torch

from kitti_utils import generate_depth_map
from .SingleDataset import SingleDataset
from torchvision import transforms
import json
import copy


class CITYSCAPEDataset(SingleDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(CITYSCAPEDataset, self).__init__(*args, **kwargs)
        # self.kitti_K = np.array([[0.58, 0, 0.5, 0],
        #                         [0, 1.92, 0.5, 0],
        #                         [0, 0, 1, 0],
        #                         [0, 0, 0, 1]], dtype=np.float32)
        self.downR = 32 # downsample ratio between input img and feature map
        self.kitti_uf = np.array((0.58, 1.92)) # [kitti unit focal x per pixel, kitti unit focal y per pixel]
        self.full_res_shape = (2048, 1024) # decide to use 512 by 256
        self.side_map = {"l": "leftImg8bit", "r": "rightImg8bit"}
        self.change_resize()
        self.mask = None
        self.load_mask()
        # self.ctsImg_sz_rec = dict()

    def load_mask(self):
        imgPath = 'assets/cityscapemask.png'
        self.mask = dict()
        maskImg = self.loader(imgPath)
        for i in range(self.num_scales):
            mask = np.array(maskImg.resize((int(self.width / (2**i)), int(self.height / (2**i))), pil.NEAREST))[:,:,0]
            mask = (mask < 1).astype(np.uint8)
            self.mask[('mask', i)] = torch.from_numpy(mask)
    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        imgPath = self.get_image_path(folder, frame_index, side)
        color = self.loader(imgPath)
        # cts_focal, baseline = self.get_cityscape_cam_param(folder)

        # targ_f = self.kitti_uf * np.array((self.width, self.height))
        # re_size = np.round(self.full_res_shape / cts_focal * targ_f).astype(np.intc)
        # re_size = (np.round(re_size / self.downR) * self.downR).astype(np.intc)
        # self.ctsImg_sz_rec[imgPath] = re_size
        # re_size = self.ctsImg_sz_rec[self.t_folder(folder)]
        # color = color.resize(re_size, self.interp)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    def change_resize(self):
        cts_focal = np.array((2268.36, 2225.5405988775956)) # suppose all cityscape img possess same pixel focal length
        targ_f = self.kitti_uf * np.array((self.width, self.height))
        re_size = np.round(self.full_res_shape / cts_focal * targ_f).astype(np.int)
        re_size = (np.round(re_size / self.downR) * self.downR).astype(np.int)
        re_size = np.array((512, 256),dtype=np.int)
        self.org_width = copy.copy(self.width)
        self.width = re_size[0].item()
        self.height = re_size[1].item()

        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
        self.seman_resize = transforms.Resize((self.height, self.width),
                                               interpolation=pil.NEAREST)

    # def get_img_size(self, folder):
    #     return self.ctsImg_sz_rec[self.t_folder(folder)]

    def get_K(self, folder):
        # Return K as well as record all necessary things
        cts_focal, baseline = self.get_cityscape_cam_param(folder)
        unit_focal_pixel = cts_focal / self.full_res_shape
        K = np.eye(4, dtype=np.float32)
        K[0, 2] = 0.5
        K[1, 2] = 0.5
        K[0, 0] = unit_focal_pixel[0]
        K[1, 1] = unit_focal_pixel[1]

        # targ_f = self.kitti_uf * np.array((self.width, self.height))
        # re_size = np.round(self.full_res_shape / cts_focal * targ_f).astype(np.intc)
        # re_size = (np.round(re_size / self.downR) * self.downR).astype(np.intc)

        # ck = self.t_folder(folder)
        # if ck not in self.ctsImg_sz_rec.keys():
        #     self.ctsImg_sz_rec[ck] = re_size
        return K
    def get_rescaleFac(self, folder):
        cts_focal, cts_baseline = self.get_cityscape_cam_param(folder)
        kitti_baseline = 0.54
        kitti_unit_focal = 0.58
        cts_unit_focal = cts_focal[0] / self.full_res_shape[0]
        # rescale_fac = (cts_baseline * cts_unit_focal * self.width) / (kitti_baseline * kitti_unit_focal * self.org_width)
        rescale_fac = cts_baseline / kitti_baseline
        return rescale_fac

    def get_seman(self, folder, do_flip):
        seman_path, ins_path = self.get_ins_seman_path(folder)
        color = self.loader(seman_path)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        seman_label = np.array(color)[:,:,0]
        # ins_label = np.array(self.loader(ins_path))
        # pil.fromarray(((seman_label == 1)*255).astype(np.uint8)).show()
        return seman_label

    def check_seman(self):
        return True

class CITYSCAPERawDataset(CITYSCAPEDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(CITYSCAPERawDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        image_path = os.path.join(self.data_path, self.side_map[side], folder + self.side_map[side] + self.img_ext)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        return None

    def get_cityscape_cam_param(self, folder):
        # camName = os.path.join("camera_trainvaltest", "camera")
        k_path = os.path.join(self.data_path, "camera", folder + "camera.json")
        with open(k_path) as json_file:
            data = json.load(json_file)
            baseline = data['extrinsic']['baseline']
            cts_focal = np.array((data['intrinsic']['fx'], data['intrinsic']['fy'])) # [fx, fy]
        return cts_focal, baseline

    def get_ins_seman_path(self, folder):
        if folder.split("/")[0] == 'train' or folder.split("/")[0] == 'val':
            ins_path = os.path.join(self.data_path, "gtFine_processed", folder + "gtFine_instanceTrainIds" + self.img_ext)
            seman_path = os.path.join(self.data_path, "gtFine_processed", folder + "gtFine_labelTrainIds" + self.img_ext)
        elif folder.split("/")[0] == 'train_extra':
            ins_path = os.path.join(self.data_path, "gtCoarse_processed", folder + "gtCoarse_instanceTrainIds" + self.img_ext)
            seman_path = os.path.join(self.data_path, "gtCoarse_processed", folder + "gtCoarse_labelTrainIds" + self.img_ext)
        return seman_path, ins_path

    # def t_folder(self, folder):
    #     return folder.split('/')[2][:-1:]