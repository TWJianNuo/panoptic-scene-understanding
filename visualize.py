from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
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
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4
def wrap_visual_rgb(tensor, ind):
    slice = (tensor[ind, :, :, :].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    # pil.fromarray(slice).show()
    return pil.fromarray(slice)

def wrap_visual_semantic(tensor, ind):
    slice = tensor[ind, :, :, :]
    slice = F.softmax(slice, dim=0)
    slice = torch.argmax(slice, dim=0).cpu().numpy()
    # visualize_semantic(slice).show()
    return visualize_semantic(slice)
def wrap_visual_disp(tensor, ind):
    # slice = tensor[]
    # plt.imsave(name_dest_im, disp_resized_np, cmap='magma', vmax=vmax)
    slice = tensor[ind, 0, :, :].cpu().numpy()
    vmax = np.percentile(slice, 95)
    slice = slice / vmax
    cm = plt.get_cmap('magma')
    slice = (cm(slice) * 255).astype(np.uint8)
    # pil.fromarray(slice).show()
    return pil.fromarray(slice)
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
        dataset = datasets.KITTISemanticDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False, tag=opt.dataset)
    else:
        raise ValueError("No predefined dataset")
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

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
    sfx = torch.nn.Softmax(dim=1)
    mergeDisp = Merge_MultDisp(opt.scales, batchSize = opt.batch_size)
    svRoot = '/media/shengjie/other/sceneUnderstanding/monodepth2/internalRe/figure_visual'

    index = 0
    isvisualize = True
    isHist = False
    if isHist:
        rec = np.zeros((19,100))
    if opt.isMulChannel:
        app = 'mulDispOn'
    else:
        app = 'mulDispOff'
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
            ## if visualize the image
            if isvisualize:
                mergeDisp(inputs, outputs, eval=True)
                fig_seman = wrap_visual_semantic(outputs[('seman', 0)], ind=index)
                fig_rgb = wrap_visual_rgb(inputs[('color', 0, 0)], ind=index)
                fig_disp = wrap_visual_disp(outputs[('disp', 0)], ind=index)
                combined = [np.array(fig_disp)[:,:,0:3], np.array(fig_seman), np.array(fig_rgb)]
                combined = np.concatenate(combined, axis=1)
                fig = pil.fromarray(combined)
                fig.save(os.path.join(svRoot, app, str(idx) + '.png'))

                # for k in range(10):
                #     fig_disp = wrap_visual_disp(outputs[('disp', 0)], ind=k)
                #     fig_rgb = wrap_visual_rgb(inputs[('color', 0, 0)], ind=k)
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
