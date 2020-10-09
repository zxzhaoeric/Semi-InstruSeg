import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from albumentations.pytorch.functional import img_to_tensor
from albumentations import Resize
import random
from scipy import ndimage
import ds_utils.robseg_2017 as utils


class RobotSegDataset(Dataset):
    """docstring for RobotSegDataset"""
    def __init__(self, filenames, transform, mode, model,
        problem_type, semi=False, **kwargs):
        super(RobotSegDataset, self).__init__()
        self.filenames = filenames
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type
        self.factor = utils.problem_factor[problem_type]
        self.mask_folder = utils.mask_folder[problem_type]
        self.model = model



    def __len__(self):
        # num of imgs
        return len(self.filenames)

    def __getitem__(self, idx):
        '''
        pytorch dataloader get_item_from_index
        input:
        idx: corresponding with __len__
        output:
        input_dict: a dictionary stores all return value
        '''

        # input dict for return
        input_dict = {}

        # abs_idx: absolute index in original order
        filename, abs_idx = self.get_filename_from_idx(idx)

        image = self.load_image(filename)


        # gts
        mask = self.load_mask(filename, self.mask_folder)

        # augment
        data = {'image': image, 'mask': mask}

        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        # input image
        input_dict['input'] = img_to_tensor(image)
        if self.mode == 'eval':
            # in evaluation mode should input the filename of input image
            input_dict['input_filename'] = str(filename)
        else:
            # input gt
            input_dict['target'] = torch.from_numpy(mask).long()

        return input_dict

    def mark_labels(self, method):
        '''
        mark labeles for dataset
        the first and last label should be 1
        e.g. semi_percentage = 0.3
        interval: labeled = 1,0,0,1,0,0,1......1
        random: randomly mark semi_percentage of data as labeled
        '''
        num_files = len(self.filenames)
        ratio = self.semi_percentage
        assert num_files > 2
        if method == 'interval':
            labeled = np.zeros((num_files,))
            sep = int(1 // ratio)
            labeled[::sep] = 1
            labeled[-1] = 1
            # print(sep)
            # print(labeled)
        elif method == 'random':
            labeled = np.random.choice([0,1], size=num_files, p=[1-ratio, ratio])
        else:
            raise NotImplementedError

        return labeled



    def get_filename_from_idx(self, idx):
        '''
        get filename from pytorch shuffled index

        input:
        pytorch dataset __getitem__ idx

        output:
        filename: correct filename in specific order
        abs_idx: absolute index in original dataset order

        '''
        filename = self.filenames[idx]
        abs_idx = idx
        if 'TAPNet' in self.model and self.mode == 'train':
            if self.schedule == "shuffle":
                filename = self.shuffled_filenames[idx]
                abs_idx = self.shuffled_idx[idx]
            elif self.schedule == "ordered":
                filename = self.ordered_filenames[idx]
                abs_idx = self.ordered_idx[idx]
            else:
                raise NotImplementedError
        return filename, abs_idx


    def load_image(self, filename):
        return cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)

    def load_mask(self, filename, mask_folder):
        # change dir name
        mask = cv2.imread(str(filename).replace('images', mask_folder), 0)
        return (mask / self.factor).astype(np.uint8)


    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())
