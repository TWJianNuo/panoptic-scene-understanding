from __future__ import absolute_import, division, print_function

# import os
# import sys
# import glob
# import argparse
# import numpy as np
# import PIL.Image as pil
#
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from mpl_toolkits.mplot3d import Axes3D
#
# import torch
#
# from layers import disp_to_depth
# from utils import download_model_if_doesnt_exist
# from utils import set_axes_equal
# from utils import reconstruct3dPts
# from utils import project3dPts
# from utils import diffICP
# from kitti_utils import readCamParam
# from kitti_utils import readSeman
#
# from PIL import Image
# import kitti_semantics_util.labels
# from globalInfo import acGInfo
# from scipy.interpolate import griddata
# from scipy.interpolate import RectBivariateSpline
# from scipy.misc import imresize
from utils import *
from kitti_utils import *
from layers import *

import sys
import datasets
import networks
# from IPython import embed
from utils import my_Sampler
from torch.utils.data import DataLoader
import warnings
import cv2
warnings.filterwarnings("ignore")
def test_simple(model_name, paths, val_iter_list, batch_it_num, backproject_depth_l, project_3d_l, sv_path_l):
    """Function to predict for a single image or folder of images
    """
    device = torch.device("cuda")
    model_path = model_name
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(paths, model_path, "encoder.pth")
    depth_decoder_path = os.path.join(paths, model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    # feed_height = loaded_dict_enc['height']
    # feed_width = loaded_dict_enc['width']
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

    print("-> Predicting on test images")

    # PREDICTING ON EACH IMAGE IN TURN
    disp_resized_np_list = list()
    source_scale = 0
    with torch.no_grad():
        for count, val_iter in enumerate(val_iter_list):
            backproject_depth = backproject_depth_l[count]
            project_3d = project_3d_l[count]
            svcount = 0
            for k in range(batch_it_num[count]):
                try:
                    inputs = val_iter.next()
                except StopIteration:
                    print("Finish iterating all available data")
                    break
                T = inputs["stereo_T"].cuda()
                input_rgb = inputs[('color', 0, 0)].cuda()
                sample_rgb = inputs[('color', 's', 0)].cuda()
                features = encoder(input_rgb)
                outputs = depth_decoder(features)
                disp = outputs[("disp", 0)]
                _, depth = disp_to_depth(disp, 0.1, 100)

                cam_points = backproject_depth(
                    depth, inputs[("inv_K", source_scale)].cuda())
                pix_coords = project_3d(
                    cam_points, inputs[("K", source_scale)].cuda(), T)
                reconstructed_rgb = F.grid_sample(
                    sample_rgb,
                    pix_coords,
                    padding_mode="border")
                reconstructed_rgb = reconstructed_rgb.permute(0,2,3,1).cpu()
                for picind in range(reconstructed_rgb.shape[0]):
                    c_sv_path = os.path.join(sv_path_l[count], str(svcount) + ".png")
                    img1 = inputs[('color', 's', 0)].permute(0,2,3,1)[picind,:,:,:].numpy()
                    img2 = reconstructed_rgb[picind,:,:,:].numpy()
                    img3 = inputs[('color', 0, 0)].permute(0,2,3,1)[picind,:,:,:].numpy()
                    combined_img = np.concatenate((img1, img2, img3), axis=0)
                    Image.fromarray((combined_img * 255).astype(np.uint8)).save(c_sv_path)
                    svcount = svcount + 1
                print("finish %dth dataset %dth batch" % (count, k))

def recover_the_other_side():
    datasets_dict = {
        "kitti": datasets.KITTIRAWDataset,
        "kitti_odom": datasets.KITTIOdomDataset,
        "cityscape": datasets.CITYSCAPERawDataset,
        "joint": datasets.JointDataset
    }
    model_path = '/media/shengjie/other/sceneUnderstanding/monodepth2/models'
    model_name = 'stereo_640x192'
    dataset_set = ("cityscape",
                   "kitti")
    split_set = ("cityscape",
                 "eigen_full")
    datapath_set = ("/media/shengjie/other/cityscapesData",
                    "/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data")
    sv_path = "/media/shengjie/other/sceneUnderstanding/monodepth2/internalRe/recon_rg_img"
    sv_path_l = list()
    batch_size = 12
    img_sizes = np.array((192, 640))
    batch_it_num = np.array((10, 10))
    val_iter_list = list()

    backproject_depth_l = list()
    project_3d_l = list()
    for i, d in enumerate(dataset_set):
        height = img_sizes[0]
        width = img_sizes[1]
        initFunc = datasets_dict[d]
        fpath = os.path.join(os.path.dirname(__file__), "splits", split_set[i], "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png'
        val_dataset = initFunc(
            datapath_set[i], val_filenames, height, width,
            [0, 's'], 4, tag=dataset_set[i], is_train=False, img_ext=img_ext)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True,
            num_workers=batch_size, pin_memory=True, drop_last=True)
        backproject_depth_l.append(BackprojectDepth(batch_size, val_dataset.height, val_dataset.width).cuda())
        project_3d_l.append(Project3D(batch_size, val_dataset.height, val_dataset.width).cuda())
        val_iter = iter(val_loader)
        val_iter_list.append(val_iter)
        sv_path_l.append(os.path.join(sv_path, dataset_set[i]))
    test_simple(model_name, model_path, val_iter_list, batch_it_num, backproject_depth_l, project_3d_l, sv_path_l)

if __name__ == "__main__":
    # Edit python path
    recover_the_other_side()