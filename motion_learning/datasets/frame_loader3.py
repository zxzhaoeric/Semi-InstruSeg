from __future__ import division
from __future__ import print_function
import argparse
import os
import natsort
import numpy as np
import cv2
from pathlib import Path
import torch
from torch.utils import data
import random

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
        if idx in [1]:
            val_file_names += filenames
        else:
            train_file_names += filenames
        full_train += filenames

    return train_file_names, val_file_names

class FrameLoader(data.Dataset):
    def __init__(self, args, filename, is_training = False, transform=None, back_propagation = False):

        self.is_training = is_training
        self.transform = transform
        self.chsize = 3

        # carry over command line arguments
        assert args.sequence_length > 1, 'sequence length must be > 1'
        self.sequence_length = args.sequence_length

        assert args.sample_rate > 0, 'sample rate must be > 0'
        self.sample_rate = args.sample_rate

        self.crop_size = args.crop_size
        self.start_index = args.start_index
        self.stride = args.stride

        if self.is_training:
            self.start_index = 0

        # collect, colors, motion vectors, and depth
        self.filename = filename
        self.back_propagation = back_propagation


    def __len__(self):
        return len(self.filename)-1

    def __getitem__(self, index):
        # adjust index
        if self.is_training:
            if (index  + 1) % 225 ==0:
                index = index - 1
            input_files = [str(self.filename[index + offset]) for offset in range(2)]

        else:
            input_files = [str(self.filename[index + offset]) for offset in range(2)]

        # reverse image order with p=0.5
        if self.is_training and torch.randint(0, 2, (1,)).item():
            input_files = input_files[::-1]

        # images = [imageio.imread(imfile)[..., :self.chsize] for imfile in input_files]
        images = [cv2.imread(str(imfile))[..., :self.chsize] for imfile in input_files]
        input_shape = images[0].shape[:2]


        if self.is_training:
            imgs = []
            for img in images:
                im = cv2.resize(img, (640, 512))
                imgs.append(im)
            cropper = StaticRandomCrop(self.crop_size, (512, 640))
            images = map(cropper, imgs)
        else:
            images = [cv2.resize(img, (448, 448)) for img in images]



        input_images = [torch.from_numpy(im.transpose(2, 0, 1)).float() for im in images]

        output_dict = {
            'image': input_images, 'ishape': input_shape, 'input_files': input_files
        }

        return output_dict




if __name__ == '__main__':
    root = '../../../data/cropped_train'
    parser = argparse.ArgumentParser(description='A PyTorch Implementation of SDCNet2D')
    parser.add_argument('--sequence_length', default=2, type=int, metavar="SEQUENCE_LENGTH",
                        help='number of interpolated frames (default : 7)')
    parser.add_argument("--sample_rate", type=int, default=1, help="step size in looping through datasets")
    parser.add_argument('--crop_size', type=int, nargs='+', default=[448, 448], metavar="CROP_SIZE",
                        help="Spatial dimension to crop training samples for training (default : [448, 448])")
    parser.add_argument("--start_index", type=int, default=0, metavar="START_INDEX",
                        help="Index to start loading input data (default : 0)")
    parser.add_argument('--stride', default=64, type=int,
                        help='The factor for which padded validation image sizes should be evenly divisible. (default: 64)')
    train_filenames, valid_filenames =trainval_split(root)
    args = parser.parse_args()
    train_dataset = FrameLoader(args, filename=train_filenames, is_training=True)
    print(train_dataset[0]['image'])