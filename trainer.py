# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
# from IPython import embed
from utils import my_Sampler
import cityscapesscripts.helpers.labels
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import *


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        if self.opt.switchMode == 'on':
            self.switchMode = True
        else:
            self.switchMode = False
        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.semanticCoeff = 1
        self.sfx = nn.Softmax()
        # self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales, isSwitch=self.switchMode)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # self.models["semantic"] = networks.CombinedDecoder(
        #     self.models["encoder"].num_ch_enc, self.opt.scales)
        # self.models["semantic"].to(self.device)
        # self.parameters_to_train += list(self.models["semantic"].parameters())

        # if self.use_pose_net:
        #     if self.opt.pose_model_type == "separate_resnet":
        #         self.models["pose_encoder"] = networks.ResnetEncoder(
        #             self.opt.num_layers,
        #             self.opt.weights_init == "pretrained",
        #             num_input_images=self.num_pose_frames)
        #
        #         self.models["pose_encoder"].to(self.device)
        #         self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        #
        #         self.models["pose"] = networks.PoseDecoder(
        #             self.models["pose_encoder"].num_ch_enc,
        #             num_input_features=1,
        #             num_frames_to_predict_for=2)
        #
        #     elif self.opt.pose_model_type == "shared":
        #         self.models["pose"] = networks.PoseDecoder(
        #             self.models["encoder"].num_ch_enc, self.num_pose_frames)
        #
        #     elif self.opt.pose_model_type == "posecnn":
        #         self.models["pose"] = networks.PoseCNN(
        #             self.num_input_frames if self.opt.pose_model_input == "all" else 2)
        #
        #     self.models["pose"].to(self.device)
        #     self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        self.set_dataset()
        # data
        # datasets_dict = {
        #     "kitti": datasets.KITTIRAWDataset,
        #     "kitti_odom": datasets.KITTIOdomDataset,
        #     "cityscape": datasets.CITYSCAPERawDataset
        #                  }
        # self.dataset = datasets_dict[self.opt.dataset]
        #
        # fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        #
        # train_filenames = readlines(fpath.format("train"))
        # val_filenames = readlines(fpath.format("val"))
        # img_ext = '.png' if self.opt.png else '.jpg'
        #
        # num_train_samples = len(train_filenames)
        # self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        #
        # train_dataset = self.dataset(
        #     self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
        #     self.opt.frame_ids, 4, tag=self.opt.dataset, is_train=True, img_ext=img_ext, require_seman=self.opt.require_semantic)
        # self.train_loader = DataLoader(
        #     train_dataset, self.opt.batch_size, True,
        #     num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        # val_dataset = self.dataset(
        #     self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
        #     self.opt.frame_ids, 4, tag=self.opt.dataset, is_train=False, img_ext=img_ext)
        # self.val_loader = DataLoader(
        #     val_dataset, self.opt.batch_size, True,
        #     num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        # self.val_iter = iter(self.val_loader)

        # Change height and weight accordingly
        # self.opt.height = train_dataset.height
        # self.opt.width = train_dataset.width

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.set_layers()
        # self.backproject_depth = {}
        # self.project_3d = {}
        # for scale in self.opt.scales:
        #     h = self.opt.height // (2 ** scale)
        #     w = self.opt.width // (2 ** scale)
        #
        #     self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
        #     self.backproject_depth[scale].to(self.device)
        #
        #     self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
        #     self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("Switch mode on") if self.switchMode else print("Switch mode off")
        print("There are {:d} training items and {:d} validation items\n".format(
            self.train_num, self.val_num))

        self.save_opts()

    def set_layers(self):
        """properly handle layer initialization under multiple dataset situation
        """
        self.semanticLoss = Compute_SemanticLoss(min_scale = self.opt.semantic_minscale[0])
        self.backproject_depth = {}
        self.project_3d = {}
        tags = list()
        for t in self.format:
            tags.append(t[0])
        for p, tag in enumerate(tags):
            height = self.format[p][1]
            width = self.format[p][2]
            for n, scale in enumerate(self.opt.scales):
                h = height // (2 ** scale)
                w = width // (2 ** scale)

                self.backproject_depth[(tag, scale)] = BackprojectDepth(self.opt.batch_size, h, w)
                self.backproject_depth[(tag, scale)].to(self.device)

                self.project_3d[(tag, scale)] = Project3D(self.opt.batch_size, h, w)
                self.project_3d[(tag, scale)].to(self.device)

    def set_dataset(self):
        """properly handle multiple dataset situation
        """
        datasets_dict = {
            "kitti": datasets.KITTIRAWDataset,
            "kitti_odom": datasets.KITTIOdomDataset,
            "cityscape": datasets.CITYSCAPERawDataset,
            "joint": datasets.JointDataset
                         }
        dataset_set = self.opt.dataset.split('+')
        split_set = self.opt.split.split('+')
        datapath_set = self.opt.data_path.split('+')
        assert len(dataset_set) == len(split_set), "dataset and split should have same number"
        stacked_train_datasets = list()
        stacked_val_datasets = list()
        train_sample_num = np.zeros(len(dataset_set), dtype=np.int)
        val_sample_num = np.zeros(len(dataset_set), dtype=np.int)
        for i, d in enumerate(dataset_set):
            initFunc = datasets_dict[d]
            fpath = os.path.join(os.path.dirname(__file__), "splits", split_set[i], "{}_files.txt")
            train_filenames = readlines(fpath.format("train"))
            val_filenames = readlines(fpath.format("val"))
            img_ext = '.png' if self.opt.png else '.jpg'

            train_dataset = initFunc(
                datapath_set[i], train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, tag=dataset_set[i], is_train=True, img_ext=img_ext)
            train_sample_num[i] = train_dataset.__len__()
            stacked_train_datasets.append(train_dataset)

            val_dataset = initFunc(
                datapath_set[i], val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, tag=dataset_set[i], is_train=False, img_ext=img_ext)
            val_sample_num[i] = val_dataset.__len__()
            stacked_val_datasets.append(val_dataset)

        initFunc = datasets_dict['joint']
        self.joint_dataset_train = initFunc(stacked_train_datasets)
        joint_dataset_val = initFunc(stacked_val_datasets)

        self.trainSample = my_Sampler(train_sample_num, self.opt.batch_size) # train sampler is used for multi-stage training
        valSample = my_Sampler(val_sample_num, self.opt.batch_size)

        self.train_loader = DataLoader(
            self.joint_dataset_train, self.opt.batch_size, shuffle=False, sampler=self.trainSample,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            joint_dataset_val, self.opt.batch_size, shuffle=False, sampler=valSample,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        num_train_samples = self.joint_dataset_train.__len__()
        self.train_num = self.joint_dataset_train.__len__()
        self.val_num = joint_dataset_val.__len__()
        self.format = self.joint_dataset_train.format
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            # self.set_dataset()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()
        print("Training")
        self.set_train()
        # adjust by changing the sampler
        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 1
            late_phase = self.step % 1 == 0

            if early_phase or late_phase:

                if "loss_semantic" in losses:
                    loss_seman = losses["loss_semantic"].cpu().data
                    self.compute_semantic_losses(inputs, outputs, losses)
                else:
                    loss_seman = -1
                if "loss_depth" in losses:
                    loss_depth = losses["loss_depth"].cpu().data
                else:
                    loss_depth = -1

                self.log_time(batch_idx, duration, loss_seman, loss_depth)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                if self.step % 50 == 0:
                    self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if not(key == 'height' or key == 'width' or key == 'tag'):
                inputs[key] = ipt.to(self.device)

        # if self.opt.pose_model_type == "shared":
        #     all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
        #     all_features = self.models["encoder"](all_color_aug)
        #     all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
        #
        #     features = {}
        #     for i, k in enumerate(self.opt.frame_ids):
        #         features[k] = [f[i] for f in all_features]
        #
        #     outputs = self.models["depth"](features[0])
        # else:
        #     features = self.models["encoder"](inputs["color_aug", 0, 0])
        #     outputs = self.models["depth"](features)


        features = self.models["encoder"](inputs["color_aug", 0, 0])
        # just for check
        """
        i = 1
        img = pil.fromarray((inputs[("color_aug", 0, 0)].permute(0,2,3,1)[i,:,:,:].cpu().numpy() * 255).astype(np.uint8))
        img.show()
        label = inputs['seman_gt'].permute(0,2,3,1)[i,:,:,0].cpu().numpy()
        visualize_semantic(label).show()
        """



        # Switch between semantic and depth estimation
        if 'seman_gt' in inputs:
            outputs = self.models["depth"](features, computeSemantic = True, computeDepth = False)
        else:
            outputs = self.models["depth"](features, computeSemantic = False, computeDepth = True)
            if self.opt.predictive_mask:
                outputs["predictive_mask"] = self.models["predictive_mask"](features)
            self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    """
    def predict_poses(self, inputs, features):
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs
    """

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)
            if 'seman_gt_eval' in inputs:
                self.compute_semantic_losses(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()


    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        tag = inputs['tag'][0]
        height = inputs["height"][0]
        width = inputs["width"][0]
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                # disp = F.interpolate(
                #     disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                disp = F.interpolate(disp, [height, width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]


                # if self.opt.pose_model_type == "posecnn":
                #
                #     axisangle = outputs[("axisangle", 0, frame_id)]
                #     translation = outputs[("translation", 0, frame_id)]
                #
                #     inv_depth = 1 / depth
                #     mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                #
                #     T = transformation_from_parameters(
                #         axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[(tag, source_scale)](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[(tag, source_scale)](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")
                # visualize_outpu(inputs, outputs, '/media/shengjie/other/sceneUnderstanding/monodepth2/internalRe/recon_rg_img/kitti', np.random.randint(0, 100000, 1)[0])
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        if ('disp', 0) in outputs:
            for scale in self.opt.scales:
                loss = 0
                reprojection_losses = []

                if self.opt.v1_multiscale:
                    source_scale = scale
                else:
                    source_scale = 0

                disp = outputs[("disp", scale)]
                color = inputs[("color", 0, scale)]
                target = inputs[("color", 0, source_scale)]

                for frame_id in self.opt.frame_ids[1:]:
                    pred = outputs[("color", frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))

                reprojection_losses = torch.cat(reprojection_losses, 1)

                if not self.opt.disable_automasking:
                    identity_reprojection_losses = []
                    for frame_id in self.opt.frame_ids[1:]:
                        pred = inputs[("color", frame_id, source_scale)]
                        identity_reprojection_losses.append(
                            self.compute_reprojection_loss(pred, target))

                    identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                    if self.opt.avg_reprojection:
                        identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                    else:
                        # save both images, and do min all at once below
                        identity_reprojection_loss = identity_reprojection_losses

                elif self.opt.predictive_mask:
                    # use the predicted mask
                    mask = outputs["predictive_mask"]["disp", scale]
                    if not self.opt.v1_multiscale:
                        # mask = F.interpolate(
                        #     mask, [self.opt.height, self.opt.width],
                        #     mode="bilinear", align_corners=False)
                        mask = F.interpolate(
                            mask, [inputs["height"], inputs["width"]],
                            mode="bilinear", align_corners=False)
                    reprojection_losses *= mask

                    # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                    weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                    loss += weighting_loss.mean()

                if self.opt.avg_reprojection:
                    reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                else:
                    reprojection_loss = reprojection_losses

                if not self.opt.disable_automasking:
                    # add random numbers to break ties
                    identity_reprojection_loss += torch.randn(
                        identity_reprojection_loss.shape).cuda() * 0.00001

                    combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
                else:
                    combined = reprojection_loss

                if combined.shape[1] == 1:
                    to_optimise = combined
                else:
                    to_optimise, idxs = torch.min(combined, dim=1)

                if not self.opt.disable_automasking:
                    outputs["identity_selection/{}".format(scale)] = (idxs > 1).float()

                loss += to_optimise.mean()

                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)
                smooth_loss = get_smooth_loss(norm_disp, color)

                loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                total_loss += loss
                losses["loss_depth/{}".format(scale)] = loss
            total_loss = total_loss / self.num_scales
            losses["loss_depth"] = total_loss
        if ('seman', 0) in outputs:
            loss_seman, loss_semantoshow = self.semanticLoss(inputs, outputs) # semantic loss is scaled already
            for entry in loss_semantoshow:
                losses[entry] = loss_semantoshow[entry]
            total_loss = total_loss + self.semanticCoeff * loss_seman
            # total_loss = torch.mean(torch.exp(-outputs[('seman', 0)][:,0,:,:]))
            losses["loss_semantic"] = loss_seman
            # losses["loss_semantic"] = total_loss
        # assert total_loss == 0, "toatal loss is zero"
        losses["loss"] = total_loss
        return losses



    def compute_semantic_losses(self, inputs, outputs, losses):
        """Compute semantic metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        gt = inputs['seman_gt_eval'].cpu().numpy().astype(np.uint8)
        pred = self.sfx(outputs[('seman', 0)]).detach()
        pred = torch.argmax(pred, dim=1).type(torch.float).unsqueeze(1)
        pred = F.interpolate(pred, [gt.shape[1], gt.shape[2]], mode='nearest')
        pred = pred.squeeze(1).cpu().numpy().astype(np.uint8)
        # visualize_semantic(gt[0,:,:]).show()
        # visualize_semantic(pred[0,:,:]).show()

        confMatrix = generateMatrix(args)
        # instStats = generateInstanceStats(args)
        # perImageStats = {}
        # nbPixels = 0

        groundTruthNp = gt
        predictionNp = pred
        # imgWidth = groundTruthNp.shape[1]
        # imgHeight = groundTruthNp.shape[0]
        nbPixels = groundTruthNp.shape[0] * groundTruthNp.shape[1] * groundTruthNp.shape[2]

        # encoding_value = max(groundTruthNp.max(), predictionNp.max()).astype(np.int32) + 1
        encoding_value = 256 # precomputed
        encoded = (groundTruthNp.astype(np.int32) * encoding_value) + predictionNp

        values, cnt = np.unique(encoded, return_counts=True)
        count255 = 0
        for value, c in zip(values, cnt):
            pred_id = value % encoding_value
            gt_id = int((value - pred_id) / encoding_value)
            if pred_id == 255 or gt_id == 255:
                count255 = count255 + c
                continue
            if not gt_id in args.evalLabels:
                printError("Unknown label with id {:}".format(gt_id))
            confMatrix[gt_id][pred_id] += c

        if confMatrix.sum() +  count255!= nbPixels:
            printError(
                'Number of analyzed pixels and entries in confusion matrix disagree: contMatrix {}, pixels {}'.format(
                    confMatrix.sum(), nbPixels))

        classScoreList = {}
        for label in args.evalLabels:
            labelName = trainId2label[label].name
            classScoreList[labelName] = getIouScoreForLabel(label, confMatrix, args)
        vals = np.array(list(classScoreList.values()))
        losses['mIOU'] = np.mean(vals[np.logical_not(np.isnan(vals))])

        # for i in range(pred.shape[0]):
        #     groundTruthNp = gt[i, :, :]
        #     predictionNp = pred[i, :, :]
        #     imgWidth = groundTruthNp.shape[1]
        #     imgHeight = groundTruthNp.shape[0]
        #     nbPixels = imgWidth * imgHeight
        #
        #     encoding_value = max(groundTruthNp.max(), predictionNp.max()).astype(np.int32) + 1
        #     encoded = (groundTruthNp.astype(np.int32) * encoding_value) + predictionNp
        #
        #     values, cnt = np.unique(encoded, return_counts=True)
        #
        #     for value, c in zip(values, cnt):
        #         pred_id = value % encoding_value
        #         gt_id = int((value - pred_id) / encoding_value)
        #         if pred_id == 255 or gt_id == 255:
        #             continue
        #         if not gt_id in args.evalLabels:
        #             printError("Unknown label with id {:}".format(gt_id))
        #         confMatrix[gt_id][pred_id] += c
        #
        #     if confMatrix.sum() != nbPixels:
        #         printError(
        #             'Number of analyzed pixels and entries in confusion matrix disagree: contMatrix {}, pixels {}'.format(
        #                 confMatrix.sum(), nbPixels))

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss_semantic, loss_depth):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss_semantic: {:.5f} | loss_depth: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss_semantic, loss_depth,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        # for j in range(min(4, self.opt.batch_size)):
        #     for s in self.opt.scales:
        #         for frame_id in self.opt.frame_ids:
        #             writer.add_image(
        #                 "color_{}_{}/{}".format(frame_id, s, j),
        #                 inputs[("color", frame_id, s)][j].data, self.step)
        #             if s == 0 and frame_id != 0:
        #                 writer.add_image(
        #                     "color_pred_{}_{}/{}".format(frame_id, s, j),
        #                     outputs[("color", frame_id, s)][j].data, self.step)
        #
        #         writer.add_image(
        #             "disp_{}/{}".format(s, j),
        #             normalize_image(outputs[("disp", s)][j]), self.step)
        #
        #         if self.opt.predictive_mask:
        #             for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
        #                 writer.add_image(
        #                     "predictive_mask_{}_{}/{}".format(frame_id, s, j),
        #                     outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
        #                     self.step)
        #
        #         elif not self.opt.disable_automasking:
        #             writer.add_image(
        #                 "automask_{}/{}".format(s, j),
        #                 outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
                # to_save['item_recList'] = self.joint_dataset_train.item_recList
            # cpk_dict = self.generate_cpk(model.state_dict())
            # to_save['cpk_dict'] = cpk_dict # To check load correctness
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        print("save to %s" % save_folder)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            # if n == 'encoder':
            #     self.item_recList = pretrained_dict['item_recList']
            # saved_cpk_dict = pretrained_dict['cpk_dict']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
            # updated_cpk_dict = self.generate_cpk(self.models[n].state_dict())
            # assert torch.abs(torch.mean(torch.Tensor(list(updated_cpk_dict.values()))) - torch.mean(torch.Tensor(list(saved_cpk_dict.values())))) < 1e-3, print("%s check failed" % n)


        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def multi_stage_training(self, setting_file_path = None):
        schedule = list()
        if setting_file_path is None:
            # set nothing
            schedule.append(np.array([1,1,self.opt.num_epochs])) # ratio is 1 by 1
        else:
            f = open(setting_file_path, 'r')
            x = f.readlines()
            f.close()
            schedule = schedule + x
        return schedule

    # def generate_cpk(self, model_dict):
    #     cpk_dict = dict()
    #     for entry in model_dict:
    #         cpk_dict[entry] = torch.mean(torch.abs(model_dict[entry].type(torch.double)).unsqueeze(0))
    #     return cpk_dict

