#!bin/bash

DATASET_DIR="/data1/nys_new/audiohw/PANN/GTZAN/Data/genres_original"
WORKSPACE="/data1/nys_new/audiohw/PANN/finetune"

python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

#finetune
CUDA_VISIBLE_DEVICES=5 python3 pytorch/main.py train --dataset_dir="/data1/nys_new/audiohw/PANN/GTZAN/Data/genres_original" --workspace="/data1/nys_new/audiohw/PANN/finetune" --holdout_fold=3 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path="/data1/nys_new/audiohw/PANN/Cnn14_mAP=0.431.pth" --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000  --cuda

##pretrained feature
CUDA_VISIBLE_DEVICES=4 python3 pytorch/main.py train --dataset_dir="/data1/nys_new/audiohw/PANN/GTZAN/Data/genres_original" --workspace="/data1/nys_new/audiohw/PANN/finetune" --holdout_fold=3 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path="/data1/nys_new/audiohw/PANN/Cnn14_mAP=0.431.pth" --freeze_base --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --cuda
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --dataset_dir="/data1/nys_new/audiohw/PANN/GTZAN/Data/genres_original" --workspace="/data1/nys_new/audiohw/PANN/finetune" --holdout_fold=3 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path="/data1/nys_new/audiohw/PANN/Cnn14_mAP=0.431.pth" --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --freeze_base --layer_num 3 --cuda

##train from scratch
CUDA_VISIBLE_DEVICES=7 python3 pytorch/main.py train --dataset_dir="/data1/nys_new/audiohw/PANN/GTZAN/Data/genres_original" --workspace="/data1/nys_new/audiohw/PANN/finetune" --holdout_fold=1 --model_type="Cnn14" --pretrained_checkpoint_path="" --train_from_scratch --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --cuda


