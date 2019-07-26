# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
from additional_util import visualize_img

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class SingleDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 tag,
                 is_train=False,
                 img_ext='.png',
                 is_sep_train_seman = False
                 ):
        super(SingleDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.tag = tag
        self.is_sep_train_seman = is_sep_train_seman

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1



        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()
        self.load_seman = self.check_seman()
        self.normalizer = transforms.Normalize([125.3, 123.0, 113.9], [63.0, 62.1, 66.7])
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))



    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        # if self.is_train:
        #     if do_flip:
        #         print("do flip")
        #     else:
        #         print("not do flip")
        # do_flip = True
        # do_color_aug =  False
        # do_flip = False

        line = self.filenames[index].split()
        folder = line[0]

        org_K = self.get_K(folder).copy()

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None
        # if side != 'l':
        #     raise("side is wrong")

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            # K = self.get_K(folder, frame_index, side).copy()
            # K = self.K.copy()
            K = org_K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if self.load_seman:
            seman_gt, _ = self.get_seman(folder, do_flip, frame_index)
            if seman_gt is not None:
                inputs["seman_gt_eval"] = seman_gt
                seman_gt = np.array(self.seman_resize(Image.fromarray(seman_gt)))
                inputs["seman_gt"] = np.expand_dims(seman_gt, 0)
                inputs["seman_gt"] = torch.from_numpy(inputs["seman_gt"].astype(np.int))

                if self.is_sep_train_seman:
                    # Do extra augmentation to make semantic training better
                    _, semanTrain_label = self.get_seman(folder, False, frame_index)
                    semanTrain_rgb = self.get_color(folder, frame_index, "l", False)

                    # Data augmentation
                    # Color jittering
                    color_aug = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
                    semanTrain_rgb = color_aug(semanTrain_rgb)

                    # Random Flip
                    if random.random() > 0.5:
                        semanTrain_rgb = semanTrain_rgb.transpose(Image.FLIP_LEFT_RIGHT)
                        semanTrain_label = semanTrain_label.transpose(Image.FLIP_LEFT_RIGHT)

                    # Random scale
                    random_scaling = random.random() * 1.5 + 0.5
                    scaledWidth = np.int(semanTrain_rgb.size[0] * random_scaling)
                    scaledHeight = np.int(semanTrain_rgb.size[1] * random_scaling)
                    semanTrain_rgb = semanTrain_rgb.resize([scaledWidth, scaledHeight], resample = Image.BILINEAR)
                    semanTrain_label = semanTrain_label.resize([scaledWidth, scaledHeight], resample = Image.NEAREST)

                    # Random Crop
                    cropSize = 512
                    scaledWidth = semanTrain_rgb.size[0]
                    scaledHeight = semanTrain_rgb.size[1]
                    st_width = np.int(np.ceil(random.random() * (scaledWidth - cropSize)))
                    st_height = np.int(np.ceil(random.random() * (scaledHeight - cropSize)))
                    # st_width = np.int(np.ceil(1.0 * (scaledWidth - cropSize)))
                    # st_height = np.int(np.ceil(1.0 * (scaledHeight - cropSize)))
                    semanTrain_rgb = semanTrain_rgb.crop([st_width, st_height, st_width + cropSize, st_height + cropSize])
                    semanTrain_label = semanTrain_label.crop([st_width, st_height, st_width + cropSize, st_height + cropSize])

                    # Attention here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    ###################### Danger!!!!!!!!!!!!!!!!!!!!!!!!!!######################
                    semanTrain_rgb = self.to_tensor(semanTrain_rgb)
                    semanTrain_rgb = self.normalizer(semanTrain_rgb)

                    inputs["semanTrain_rgb"] = semanTrain_rgb
                    inputs["semanTrain_label"] = torch.from_numpy(np.array(semanTrain_label)[:,:,0].astype(np.int)).unsqueeze(0)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.check_cityscape_meta():
            inputs["cts_meta"] = self.get_cityscape_meta(folder)

        # baseline = self.get_baseLine(folder)
        rescale_fac = self.get_rescaleFac(folder)
        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            # stereo_T[0, 3] = side_sign * baseline_sign * baseline / 5.4
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1 * rescale_fac

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        if self.mask is not None:
            if side == 'l':
                spec_mask = self.mask['left']
            else:
                spec_mask = self.mask['right']

            for entry in spec_mask:
                if do_flip:
                    inputs[entry] = torch.flip(spec_mask[entry], dims=[1])
                else:
                    inputs[entry] = spec_mask[entry]

        # additional info
        inputs["height"] = self.height # final image height
        inputs["width"] = self.width # final image width
        inputs["tag"] = self.tag # final image tags
        camK, invcamK, realIn, realEx, velo = self.get_camK(folder, frame_index)

        inputs["camK"] = camK # Intrinsic by extrinsic
        inputs["invcamK"] = invcamK # inverse of Intrinsic by extrinsic
        inputs["realIn"] = realIn # Intrinsic
        inputs["realEx"] = realEx # Extrinsic, possibly edited to form in accordance with kitti
        if velo is not None:
            inputs["velo"] = velo
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_K(self, folder):
        raise NotImplementedError

    def get_rescaleFac(self, folder):
        raise NotImplementedError

    def get_seman(self, folder, do_flip, frame_index):
        raise NotImplementedError

    def check_seman(self):
        raise NotImplementedError

    def check_cityscape_meta(self):
        raise NotImplementedError

    def get_cityscape_meta(self, folder):
        raise NotImplementedError

    def get_camK(self, folder, frame_index):
        raise NotImplementedError