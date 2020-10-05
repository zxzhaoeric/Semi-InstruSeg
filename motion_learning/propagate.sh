CUDA_VISIBLE_DEVICES=0 python propagate.py --sequence_length 1\
        --vis \
        --propagate 1 \
		--pretrained ../model_cpkt/_ckpt_epoch_800_iter_0062399_psnr_23.22.pt.tar\
		--flownet2_checkpoint ../pretrained_model/FlowNet2_checkpoint.pth.tar   \
		--source_dir ../data/cropped_train \
		--target_dir ../data/aug0.33/instrument_dataset_1 \
		--problem_type parts \
		--device_ids 0

CUDA_VISIBLE_DEVICES=0 python propagate.py --sequence_length 1\
        --vis \
        --propagate 1 \
		--pretrained ../model_cpkt/_ckpt_epoch_800_iter_0062399_psnr_23.22.pt.tar\
		--flownet2_checkpoint ../pretrained_model/FlowNet2_checkpoint.pth.tar   \
		--source_dir ../data/cropped_train \
		--target_dir ../data/aug0.33/instrument_dataset_2 \
		--problem_type parts \
		--device_ids 0

CUDA_VISIBLE_DEVICES=0 python propagate.py --sequence_length 1\
        --vis \
        --propagate 1 \
		--pretrained ../model_cpkt/_ckpt_epoch_800_iter_0062399_psnr_23.22.pt.tar\
		--flownet2_checkpoint ../pretrained_model/FlowNet2_checkpoint.pth.tar   \
		--source_dir ../data/cropped_train \
		--target_dir ../data/aug0.33/instrument_dataset_3 \
		--problem_type parts \
		--device_ids 0

CUDA_VISIBLE_DEVICES=0 python propagate.py --sequence_length 1\
        --vis \
        --propagate 1 \
		--pretrained ../model_cpkt/_ckpt_epoch_800_iter_0062399_psnr_23.22.pt.tar\
		--flownet2_checkpoint ../pretrained_model/FlowNet2_checkpoint.pth.tar   \
		--source_dir ../data/cropped_train \
		--target_dir ../data/aug0.33/instrument_dataset_4 \
		--problem_type parts \
		--device_ids 0

CUDA_VISIBLE_DEVICES=0 python propagate.py --sequence_length 1\
        --vis \
        --propagate 1 \
		--pretrained ../model_cpkt/_ckpt_epoch_800_iter_0062399_psnr_23.22.pt.tar\
		--flownet2_checkpoint ../pretrained_model/FlowNet2_checkpoint.pth.tar   \
		--source_dir ../data/cropped_train \
		--target_dir ../data/aug0.33/instrument_dataset_5 \
		--problem_type parts \
		--device_ids 0

CUDA_VISIBLE_DEVICES=0 python propagate.py --sequence_length 1\
        --vis \
        --propagate 1 \
		--pretrained ../model_cpkt/_ckpt_epoch_800_iter_0062399_psnr_23.22.pt.tar\
		--flownet2_checkpoint ../pretrained_model/FlowNet2_checkpoint.pth.tar   \
		--source_dir ../data/cropped_train \
		--target_dir ../data/aug0.33/instrument_dataset_6 \
		--problem_type parts \
		--device_ids 0

CUDA_VISIBLE_DEVICES=0 python propagate.py --sequence_length 1\
        --vis \
        --propagate 1 \
		--pretrained ../model_cpkt/_ckpt_epoch_800_iter_0062399_psnr_23.22.pt.tar\
		--flownet2_checkpoint ../pretrained_model/FlowNet2_checkpoint.pth.tar   \
		--source_dir ../data/cropped_train \
		--target_dir ../data/aug0.33/instrument_dataset_7 \
		--problem_type parts \
		--device_ids 0

CUDA_VISIBLE_DEVICES=0 python propagate.py --sequence_length 1\
        --vis \
        --propagate 1 \
		--pretrained ../model_cpkt/_ckpt_epoch_800_iter_0062399_psnr_23.22.pt.tar\
		--flownet2_checkpoint ../pretrained_model/FlowNet2_checkpoint.pth.tar   \
		--source_dir ../data/cropped_train \
		--target_dir ../data/aug0.33/instrument_dataset_8 \
		--problem_type parts \
		--device_ids 0
