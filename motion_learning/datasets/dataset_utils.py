from __future__ import division
from __future__ import print_function
from pathlib import Path
import torch


class StaticRandomCrop(object):
    """
    Helper function for random spatial crop
    """
    def __init__(self, size, image_shape):
        h, w = image_shape
        self.th, self.tw = size
        self.h1 = torch.randint(0, h - self.th + 1, (1,)).item()
        self.w1 = torch.randint(0, w - self.tw + 1, (1,)).item()

    def __call__(self, img):
        return img[self.h1:(self.h1 + self.th), self.w1:(self.w1 + self.tw), :]

def trainval_split(data_dir):
    """
    get train/valid filenames
    """

    train_file_names = []
    val_file_names = []
    full_train=[]

    for idx in range(1, 9):
        # sort according to filename
        filenames = (Path(data_dir) / ('instrument_dataset_' + str(idx)) / 'images').glob('*')
        filenames = list(sorted(filenames))
        # file_tmp = []
        # for i in range(len(filenames)):
        #     if i % 10 == 0:
        #         file_tmp.append(filenames[i])
        # set folds[fold] as validation set
        if idx in [8]:
            val_file_names += filenames
        else:
            train_file_names += filenames
            full_train += filenames

    return train_file_names, val_file_names