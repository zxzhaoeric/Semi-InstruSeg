import os
from pathlib import Path

"""
config file
"""

# number of classes for each type of problem
problem_class = {
    'binary': 2,
    'parts': 4,
    'instruments': 8
}

# segmentation
problem_factor = {
    'binary' : 255,
    'parts' : 85,
    'instruments' : 32
}

mask_folder = {
    'binary': 'binary_masks',
    'parts': 'parts_masks',
    'instruments': 'instruments_masks'
}

# directory
root_dir = Path('../../')
data_dir = root_dir / 'data'
train_dir = data_dir / 'train'
cropped_train_dir = data_dir / 'cropped_train'
pretrained_model_dir = root_dir / 'pretrained_model'
num_files_folder = 225

# data preprocessing
original_height, original_width = 1080, 1920
cropped_height, cropped_width = 1024, 1280
h_start, w_start = 28, 320



# updated on Apr 2
# optical flow testcase
optflow_testcase_dir = data_dir / 'optflow_testcase'
