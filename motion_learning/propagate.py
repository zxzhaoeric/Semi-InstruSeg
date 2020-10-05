from __future__ import division
from __future__ import print_function
import os
import torch.nn as nn
import sys
import argparse
import cv2
import numpy as np
import shutil
import torch
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils import data
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from models.sdc_net2d import SDCNet2DRecon
sys.path.append("../")
import ds_utils.robseg_2017 as utils

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='BATCH_SIZE',
                    help='mini-batch per gpu size (default : 4)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to trained video reconstruction checkpoint')
parser.add_argument('--flownet2_checkpoint', default='', type=str, metavar='PATH',
                    help='path to flownet-2 best checkpoint')
parser.add_argument('--source_dir', default='', type=str,
                    help='directory for data (default: Cityscapes root directory)')
parser.add_argument('--target_dir', default='', type=str, help='directory to save augmented data')
parser.add_argument('--sequence_length', default=2, type=int, metavar="SEQUENCE_LENGTH",
                    help='number of interpolated frames (default : 2)')
parser.add_argument("--rgb_max", type=float, default=255.)
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--propagate', type=int, default=1, help='propagate how many steps')
parser.add_argument('--vis', action='store_true', default=False, help='augment color encoded segmentation map')
parser.add_argument('--problem_type', type=str, default='parts', metavar='parts', choices=['parts', 'parts', 'parts'], help='segmentation problem types.')
parser.add_argument('--stride', default=64, type=int,
                    help='The factor for which padded validation image sizes should be evenly divisible. (default: 64)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loader workers (default: 10)')
parser.add_argument('--device_ids', type=int, default=[0,1,2,3], nargs='+', help='GPU devices ids.')


def trainval_split(data_dir, val_id):
    """
    get train/valid filenames
    """

    train_file_names = []
    val_file_names = []
    full_train=[]
    eval_filenames = []


    for idx in range(1, 9):
        # sort according to filename
        filenames = (Path(data_dir) / ('instrument_dataset_' + str(idx)) / 'images').glob('*')
        filenames = list(sorted(filenames))
        # file_tmp = []
        # for i in range(len(filenames)):
        #     if i % 10 == 0:
        #         file_tmp.append(filenames[i])
        # set folds[fold] as validation set
        if idx in [int(val_id)]:
            val_file_names += filenames
        else:
            train_file_names += filenames
            full_train += filenames
    for i in range(len(val_file_names)):
        if (i - args.propagate) % (2 * args.propagate + 1) == 0:
            eval_filenames.append(val_file_names[i])

    return train_file_names, val_file_names, eval_filenames

class FrameLoader(data.Dataset):
    def __init__(self, args, val_file_names, eval_filenames, propagate, is_training = False, transform=None, reverse = False, recurse=0):

        self.is_training = is_training
        self.transform = transform
        self.chsize = 3

        # carry over command line arguments
        assert args.sequence_length > 1, 'sequence length must be > 1'
        self.sequence_length = 2

        self.stride = args.stride

        # collect, colors, motion vectors, and depth
        self.val_file_names = val_file_names
        self.eval_filenames = eval_filenames
        self.reverse = reverse
        self.propagate = propagate
        self.recurse = recurse


    def __len__(self):
        return len(self.eval_filenames) #25

    def __getitem__(self, index):
        # adjust index
        id = index * (2 * args.propagate + 1) + args.propagate
        if self.propagate > 2:
            if not self.reverse:
                input_files = [str(self.val_file_names[id+self.recurse]), str(self.val_file_names[id+1+self.recurse]), str(self.val_file_names[id+2+self.recurse])]
                mask_path = str(self.val_file_names[id+1+self.recurse]).replace('images', utils.mask_folder['parts'])
            else:
                input_files = [str(self.val_file_names[id-self.recurse]), str(self.val_file_names[id-1-self.recurse]), str(self.val_file_names[id-2-self.recurse])]
                mask_path = str(self.val_file_names[id-1-self.recurse]).replace('images', utils.mask_folder['parts'])
        elif self.propagate < 2:
            if not self.reverse:
                input_files = [str(self.val_file_names[id-1]), str(self.val_file_names[id]), str(self.val_file_names[id+1])]
            else:
                input_files = [str(self.val_file_names[id + 1]), str(self.val_file_names[id]), str(self.val_file_names[id - 1])]
            mask_path = str(self.val_file_names[id]).replace('images', utils.mask_folder['parts'])
        else:
            if not self.reverse:
                input_files = [str(self.val_file_names[id]), str(self.val_file_names[id + 1]), str(self.val_file_names[id + 2])]
                mask_path = str(self.val_file_names[id + 1]).replace('images', utils.mask_folder['parts'])
            else:
                input_files = [str(self.val_file_names[id]), str(self.val_file_names[id - 1]), str(self.val_file_names[id-2])]
                mask_path = str(self.val_file_names[id - 1]).replace('images', utils.mask_folder['parts'])
        images = [cv2.imread(str(imfile))[..., :self.chsize] for imfile in input_files]
        input_shape = images[0].shape[:2]

        height, width = input_shape
        if (height % self.stride) != 0:
            padded_height = (height // self.stride + 1) * self.stride
            images = [np.pad(im, ((0, padded_height - height), (0, 0), (0, 0)), 'reflect') for im in images]

        if (width % self.stride) != 0:
            padded_width = (width // self.stride + 1) * self.stride
            images = [np.pad(im, ((0, 0), (0, padded_width - width), (0, 0)), 'reflect') for im in images]


        input_images = [torch.from_numpy(im.transpose(2, 0, 1)).float() for im in images]

        output_dict = {
            'image': input_images, 'ishape': input_shape, 'input_files': input_files, 'mask_path': mask_path
        }

        return output_dict

def get_model():
    model = SDCNet2DRecon(args)
    model = nn.DataParallel(model, device_ids=args.device_ids).cuda()
    checkpoint = torch.load(args.pretrained)
    args.start_epoch = 0 if 'epoch' not in checkpoint else checkpoint['epoch']
    state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    print("Loaded checkpoint '{}' (at epoch {})".format(args.pretrained, args.start_epoch))
    return model



def one_step_augmentation(model, mode, reverse):
    if not os.path.exists(args.target_dir):
        print("Perform --propagate 1 first. The augmentation is in an auto-regressive manner. ")
    val_id = args.target_dir.split('_')[-1]
    _, val_filenames, eval_filenames = trainval_split(args.source_dir, val_id)
    vkwargs = {'batch_size': args.batch_size,
               'num_workers': args.workers,
               'pin_memory': False, 'drop_last': True}
    loader = FrameLoader(args, val_file_names = val_filenames, eval_filenames=eval_filenames, propagate= 1, reverse=reverse)
    val_loader = torch.utils.data.DataLoader(loader, shuffle=False, **vkwargs)

    gHeight, gWidth = val_loader.dataset[0]['ishape']
    for i,info in enumerate(tqdm(val_loader,total=len(val_loader)-2)):
        input_filenames = info['input_files'][-1]
        im2 = info['image'][1]
        im2 = (im2.data.cpu().numpy().squeeze().transpose(1, 2, 0)).astype(np.uint8)
        im2_path = info['input_files'][1]
        inputs = {k: [b.cuda() for b in info[k]] for k in info if k == 'image'}
        mask_path = info['mask_path']
        if mode == "rgb_image":
            _, pred3_rgb, _ = model(inputs)
            for p in range(args.batch_size):
                pred3_rgb = (pred3_rgb[p].data.cpu().numpy().squeeze().transpose(1, 2, 0)).astype(np.uint8)
                pred3_rgb = pred3_rgb[:gHeight, :gWidth, :]
                image_filename = input_filenames[p].split('/')[-1]

                if not os.path.exists(os.path.join(args.target_dir, 'images')):
                    os.makedirs(os.path.join(args.target_dir, 'images'))
                path = os.path.join(args.target_dir, 'images', image_filename)
                cv2.imwrite(path, pred3_rgb)
                im2_path = os.path.join(args.target_dir, 'images', im2_path[p].split('/')[-1])
                cv2.imwrite(im2_path, im2)

        if mode == "color_segmap":
            gt2_rgb = cv2.imread(mask_path[0])  # (1024, 1280, 3)
            mask = gt2_rgb
            gt2_rgb = gt2_rgb.transpose((2, 0, 1))
            gt2_rgb = np.expand_dims(gt2_rgb, axis=0)
            gt2_rgb = torch.from_numpy(gt2_rgb.astype(np.float32))
            gt2_rgb = Variable(gt2_rgb).contiguous().cuda()
            _, pred3_rgb, _,_ = model(inputs, label_image=gt2_rgb)
            for p in range(args.batch_size):
                pred3_rgb = (pred3_rgb[p].data.cpu().numpy().squeeze().transpose(1, 2, 0)).astype(np.uint8)
                pred3_rgb = pred3_rgb[:gHeight, :gWidth, :]
                image_filename = input_filenames[p].split('/')[-1]

                if not os.path.exists(os.path.join(args.target_dir, 'parts_masks')):
                    os.makedirs(os.path.join(args.target_dir, 'parts_masks'))
                path = os.path.join(args.target_dir, 'parts_masks', image_filename)
                cv2.imwrite(path, pred3_rgb)
                im2_path = os.path.join(args.target_dir, 'parts_masks', im2_path[p].split('/')[-1])
                cv2.imwrite(im2_path, mask)

def multi_step_augmentation(model, mode, reverse, recurse=0):
    if not os.path.exists(args.target_dir):
        print("Perform --propagate 1 first. The augmentation is in an auto-regressive manner. ")
    val_id = args.target_dir.split('_')[-1]
    _, val_filenames, eval_filenames = trainval_split(args.source_dir, val_id)
    vkwargs = {'batch_size': args.batch_size,
               'num_workers': args.workers,
               'pin_memory': False, 'drop_last': True}
    loader = FrameLoader(args, val_file_names = val_filenames, eval_filenames=eval_filenames, propagate= args.propagate, reverse=reverse, recurse=recurse)
    val_loader = torch.utils.data.DataLoader(loader, shuffle=False, **vkwargs)
    gHeight, gWidth = val_loader.dataset[0]['ishape']
    for i,info in enumerate(tqdm(val_loader,total=len(val_loader))):
        input_filenames = info['input_files'][-1]
        inputs = {k: [b.cuda() for b in info[k]] for k in info if k == 'image'}
        mask_path = info['mask_path']
        if mode == "rgb_image":
            im2_path = info['input_files'][1]
            im2_path = os.path.join(args.target_dir, 'images', im2_path[0].split('/')[-1])
            middle_im2 = cv2.imread(im2_path)
            middle_im2 = middle_im2.transpose((2, 0, 1))
            middle_im2 = np.expand_dims(middle_im2, axis=0)
            middle_im2 = torch.from_numpy(middle_im2.astype(np.float32))
            middle_im2 = Variable(middle_im2).contiguous().cuda()
            _, pred3_rgb, _ = model(inputs, label_image=middle_im2)
            for p in range(args.batch_size):
                pred3_rgb = (pred3_rgb[p].data.cpu().numpy().squeeze().transpose(1, 2, 0)).astype(np.uint8)
                pred3_rgb = pred3_rgb[:gHeight, :gWidth, :]
                image_filename = input_filenames[p].split('/')[-1]

                if not os.path.exists(os.path.join(args.target_dir, 'images')):
                    os.makedirs(os.path.join(args.target_dir, 'images'))
                path = os.path.join(args.target_dir, 'images', image_filename)
                cv2.imwrite(path, pred3_rgb)
                # im2_path = os.path.join(args.target_dir, 'images', im2_path[p].split('/')[-1])
                # cv2.imwrite(im2_path, im2)

        if mode == "color_segmap":
            gt2_rgb = cv2.imread(mask_path[0].replace('cropped_train', 'aug0.33_back'))  # (1024, 1280, 3)
            mask = gt2_rgb
            gt2_rgb = gt2_rgb.transpose((2, 0, 1))
            gt2_rgb = np.expand_dims(gt2_rgb, axis=0)
            gt2_rgb = torch.from_numpy(gt2_rgb.astype(np.float32))
            gt2_rgb = Variable(gt2_rgb).contiguous().cuda()
            _, pred3_rgb, _,_ = model(inputs, label_image=gt2_rgb)
            for p in range(args.batch_size):
                pred3_rgb = (pred3_rgb[p].data.cpu().numpy().squeeze().transpose(1, 2, 0)).astype(np.uint8)
                pred3_rgb = pred3_rgb[:gHeight, :gWidth, :]
                image_filename = input_filenames[p].split('/')[-1]

                if not os.path.exists(os.path.join(args.target_dir, 'parts_masks')):
                    os.makedirs(os.path.join(args.target_dir, 'parts_masks'))
                path = os.path.join(args.target_dir, 'parts_masks', image_filename)
                cv2.imwrite(path, pred3_rgb)
                # im2_path = os.path.join(args.target_dir, 'parts_masks', im2_path[p].split('/')[-1])
                # cv2.imwrite(im2_path, mask)




if __name__ == '__main__':
    global args
    args = parser.parse_args()
    args.sequence_length = 2
    # Load pre-trained video reconstruction model
    net = get_model()
    net.eval()

    # Config paths
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    problem_type = args.problem_type

    # Generate augmented dataset
    # create +-1 data
    print('One step augmentaion', 'rgb_image', 'reverse=False')
    one_step_augmentation(net, mode='rgb_image', reverse=False)
    print('One step augmentaion', 'rgb_image', 'reverse=True')
    one_step_augmentation(net,  mode='rgb_image', reverse=True)
    print('One step augmentaion', 'color_segmap', 'reverse=False')
    one_step_augmentation(net, mode='color_segmap', reverse=False)
    print('One step augmentaion', 'color_segmap', 'reverse=True')
    one_step_augmentation(net, mode='color_segmap', reverse=True)
    print('multi step augmentation')
    for i in range(0, args.propagate-1):
        print(i, 'th iteration')
        multi_step_augmentation(net, mode='rgb_image', reverse=False, recurse=i)
        multi_step_augmentation(net, mode='rgb_image', reverse=True, recurse=i)
        multi_step_augmentation(net, mode='color_segmap', reverse=False, recurse=i)
        multi_step_augmentation(net, mode='color_segmap', reverse=True, recurse=i)
