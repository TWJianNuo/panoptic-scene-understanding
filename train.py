# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from trainer_ASPP import Trainer_ASPP
from options import MonodepthOptions
import warnings
warnings.filterwarnings("ignore")


options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    if opts.backBone[0] == "unet":
        trainer = Trainer(opts)
    elif opts.backBone[0] == "ASPP":
        trainer = Trainer_ASPP(opts)
    trainer.train()
