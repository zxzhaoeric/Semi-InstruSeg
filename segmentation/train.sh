#!/bin/bash
# fully supervised training
CUDA_VISIBLE_DEVICES=0,1 python train.py --device_ids 0 1 --num_workers 8 --batch_size 4 --problem_type binary --folds 2 3 --lr 3e-5 --model UNet11 --jaccard_weight 0.3 --max_epochs 50 --train_dir ../data/cropped_train --input_height 1024 --input_width 1280 --tb_log False
# 30% semi supervised training
CUDA_VISIBLE_DEVICES=0,1 python train.py --device_ids 0 1 --num_workers 8 --batch_size 4 --problem_type binary --folds 2 3 --lr 3e-5 --model UNet11 --jaccard_weight 0.3 --max_epochs 50 --train_dir ../data/aug0.33 --input_height 1024 --input_width 1280 --tb_log False
