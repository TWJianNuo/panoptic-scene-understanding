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

from kitti_utils import generate_depth_map
from .SingleDataset import SingleDataset
from kitti_utils import read_calib_file, load_velodyne_points
from cityscapesscripts.helpers.labels import labels
from torchvision import transforms
from utils import visualize_semantic
class KITTIDataset(SingleDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)
        # K(0, 0) = 7.215377e+02 / 1242
        # K(1, 1) = 7.215377e+02 / 375
        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.mask = None # Cityscape requires mask
        self.seman_resize = transforms.Resize((self.height, self.width),
                                               interpolation=pil.NEAREST)

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_K(self, folder):
        K = np.array([[0.58, 0, 0.5, 0],
                      [0, 1.92, 0.5, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        return K

    def get_rescaleFac(self, folder):
        rescale_fac = 1
        return rescale_fac

    def check_cityscape_meta(self):
        return False

    def get_camK(self, folder, frame_index):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        cam2cam = read_calib_file(os.path.join(calib_path, 'calib_cam_to_cam.txt'))
        velo2cam = read_calib_file(os.path.join(calib_path, 'calib_velo_to_cam.txt'))
        velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

        # get image shape
        im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

        sfx = self.height / im_shape[0]
        sfy = self.width / im_shape[1]
        scaleM = np.eye(4)
        scaleM[0,0] = sfx
        scaleM[1,1] = sfy
        # compute projection matrix velodyne->image plane
        R_cam2rect = np.eye(4)
        R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
        P_rect = np.eye(4)
        P_rect[0:3,:] = cam2cam['P_rect_0' + str(2)].reshape(3, 4)
        P_rect = scaleM @ P_rect
        P_rect = P_rect[0:3, :]
        P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

        camK = np.eye(4)
        camK[0:3,0:4] = P_velo2im
        invcamK = np.linalg.inv(camK)

        realIn = np.eye(4)
        realIn[0:3,0:4] = P_rect

        realEx = R_cam2rect @ velo2cam

        if not self.is_train:
            velo = load_velodyne_points(velo_filename)
            velo = velo[velo[:, 0] >= 0, :]
            np.random.shuffle(velo)
            velo = velo[0 : 10000, :]
        else:
            velo = None
        # Check
        # ptsprojected = (realIn @ realEx @ velo.T).T
        # ptsprojected[:,0] = ptsprojected[:,0] / ptsprojected[:,2]
        # ptsprojected[:, 1] = ptsprojected[:, 1] / ptsprojected[:, 2]
        # minx = ptsprojected[:, 0].min()
        # maxx = ptsprojected[:, 0].max()
        # miny = ptsprojected[:, 1].min()
        # maxy = ptsprojected[:, 1].max()
        #
        # ptsprojected = (P_velo2im @ velo.T).T
        # ptsprojected[:,0] = ptsprojected[:,0] / ptsprojected[:,2]
        # ptsprojected[:, 1] = ptsprojected[:, 1] / ptsprojected[:, 2]
        # minx = ptsprojected[:, 0].min()
        # maxx = ptsprojected[:, 0].max()
        # miny = ptsprojected[:, 1].min()
        # maxy = ptsprojected[:, 1].max()
        # P_rect @ R_cam2rect @ velo2cam - (realIn @ realEx)[0:3,:]
        return camK, invcamK, realIn, realEx, velo

class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, data_path, filenames, height, width, frame_idxs, num_scales, tag, is_train=False, img_ext='.png', load_depth = False, load_meta = False, is_sep_train_seman = False):
        super(KITTIRAWDataset, self).__init__(data_path, filenames, height, width, frame_idxs, num_scales, tag, is_train, img_ext='.png', is_sep_train_seman = is_sep_train_seman)
        self.load_meta = load_meta
        if load_meta:
            self.semanticDataset = '/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_semantics'
            with open('/media/shengjie/other/sceneUnderstanding/monodepth2/splits/train_mapping.txt') as f:
                self.mapping = f.readlines()
            self.mapping = [x.strip() for x in self.mapping]
            self.seman_resize = transforms.Resize((self.height, self.width),
                                                  interpolation=pil.NEAREST)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_seman(self, folder, frame_index, side, do_flip):
        if self.load_meta:
            semanFolder = '/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_semantics/training/semantic'
            mappedLkKey = folder.split('/')
            mappedLkKey = mappedLkKey[0] + ' ' + mappedLkKey[1] + ' ' + str(frame_index).zfill(10)
            isFind = False
            foundId = -1
            for idx, chars in enumerate(self.mapping):
                if chars == mappedLkKey:
                    isFind = True
                    foundId = idx
                    break
            if isFind:
                semantics = self.loader(os.path.join(semanFolder, str(foundId).zfill(6) + '_10.png')).resize([self.full_res_shape[0], self.full_res_shape[1]], resample=pil.NEAREST)
                if do_flip:
                    semantics = semantics.transpose(pil.FLIP_LEFT_RIGHT)
                semantics = np.array(semantics)[:, :, 0]
                trainId_semantics = np.zeros_like(semantics)
                # processed_pix = 0
                for i in np.unique(semantics):
                    selector = semantics == i
                    trainId = labels[i].trainId
                    trainId_semantics[selector] = trainId
                    # processed_pix = processed_pix + np.sum(selector)
                return trainId_semantics, None
            else:
                return None, None
        else:
            folder = '2011_10_03/2011_10_03_drive_0034_sync'
            side = 'l'
            frame_index = 29
            rgb_path = self.get_image_path(folder, frame_index, side)
            if 'image_03' in rgb_path:
                semantic_label_path = rgb_path.replace('image_03/data', 'semantic_prediction/image_03')
            elif 'image_02' in rgb_path:
                semantic_label_path = rgb_path.replace('image_02/data', 'semantic_prediction/image_02')
            semantic_label = pil.open(semantic_label_path)
            if do_flip:
                semantic_label = semantic_label.transpose(pil.FLIP_LEFT_RIGHT)
            semantic_label_copy = np.array(semantic_label.copy())
            for k in np.unique(semantic_label):
                semantic_label_copy[semantic_label_copy == k] = labels[k].trainId
            # visualize_semantic(semantic_label_copy)
            return semantic_label_copy, None

    # def check_seman(self):
    #     return True

class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
