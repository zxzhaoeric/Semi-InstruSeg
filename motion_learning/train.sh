
FLOWNET2_CHECKPOINT=../pretrained_model/FlowNet2_checkpoint.pth.tar
#------training for motion prediction
CUDA_VISIBLE_DEVICES=3 python train_mp.py \
    --sequence_length 2 \
    --learning_rate 1e-4 \
    --save ../../model_cpkt \
    --batch_size 4\
    --epochs 1000 \
    --name motion_prediction \
    --val_n_batches 1 \
    --write_images \
    --model SDCNet2D \
    --train_file ../data/cropped_train \
    --flownet2_checkpoint ${FLOWNET2_CHECKPOINT}\
    --device_ids 0   \
    --save_freq 1 \
    --initial_eval

#------training for motion compensation
CUDA_VISIBLE_DEVICES=3 python train_mc.py \
    --sequence_length 2 \
    --learning_rate 2e-4 \
    --save ../../model_cpkt \
    --batch_size 2\
    --epochs 800 \
    --name motion_compensation \
    --val_n_batches 1 \
    --write_images \
    --model CycleNet \
    --train_file ../data/cropped_train \
    --flownet2_checkpoint ${FLOWNET2_CHECKPOINT}\
    --device_ids 0   \
    --save_freq 1 \
    --initial_eval
