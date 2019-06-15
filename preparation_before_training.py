# Things needed to be done before training
# Define option args
import argparse
from additional_util import PreapareCityscape
import os

class PreparationOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="prepare before training options")

        # PATHS
        self.parser.add_argument("--cts_path",
                                 type=str,
                                 default=None,
                                 help="path to the cityscape dataset")
        self.parser.add_argument("--fold_appen",
                                 type=str,
                                 default='_processed',
                                 help="change cityscape dataset semantic label to predefined value")
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options


if __name__ == "__main__":
    options = PreparationOptions()
    opts = options.parse()
    assert (opts.cts_path is not None), "Please enter necessary parameter"
    prepcts = PreapareCityscape(opts.cts_path, opts.fold_appen)