# GTZAN classification finetuned on pretrained audio neural networks (PANNs)

This codebase is for audio signal processing assignment.

## Project
**0. Download dataset, code and feature** 

The code, dataset and extracted features can be downloaded from https://pan.baidu.com/s/1tiyqrAaLDOrO0A_2zotwHQ?pwd=20vz
提取码：20vz

(P.S. It seems the original dataset downloaded from kaggle has a corrupt file: jazz.00054.wav, I replace it with jazz.00053.wav, so there shoule be 2 same wav files)


**1. Download pretrained CNN14** 

run this to download pretrained CNN14:
<pre>
wget https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1
</pre>

## Run the code

**0. Prepare data** 

Download and upzip data, the data looks like:

<pre>
dataset_root
├── blues (100 files)
├── classical (100 files)
├── country (100 files)
├── disco (100 files)
├── hiphop (100 files)
├── jazz (100 files)
├── metal (100 files)
├── pop (100 files)
├── reggae (100 files)
└── rock (100 files)
</pre>

**1. Requirements**

<pre>
git clone https://github.com/sysu19351105/Audiohw-PANN_Transfer_GTZAN.git
conda create --name audiohw python=3.7
pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
</pre>


**2. Then simply run:**

$ Run the bash script ./runme.sh

Or run the commands in runme.sh line by line. The commands includes:

(1) Modify the paths of dataset and your workspace

(2) Extract features
(optional: if you have downloaded waveform.h5 from the link, you can skip this step)

(3) Train model

For example, to finetune CNN14, you can run
<pre>
CUDA_VISIBLE_DEVICES=5 python3 pytorch/main.py train --dataset_dir="/data1/nys_new/audiohw/PANN/GTZAN/Data/genres_original" --workspace="/data1/nys_new/audiohw/PANN/finetune" --holdout_fold=1 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path="/data1/nys_new/audiohw/PANN/Cnn14_mAP=0.431.pth" --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000  --cuda
</pre>

to use pretrained CNN14 as feature extractor(Freeze_L1 in paper), you can just add --freeze_base:
<pre>
CUDA_VISIBLE_DEVICES=4 python3 pytorch/main.py train --dataset_dir="/data1/nys_new/audiohw/PANN/GTZAN/Data/genres_original" --workspace="/data1/nys_new/audiohw/PANN/finetune" --holdout_fold=1 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path="/data1/nys_new/audiohw/PANN/Cnn14_mAP=0.431.pth" --freeze_base --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --cuda
</pre>

to use pretrained CNN14 as feature extractor(Freeze_L3 in paper), you can add --freeze_base, --layer_num 3:
<pre>
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --dataset_dir="/data1/nys_new/audiohw/PANN/GTZAN/Data/genres_original" --workspace="/data1/nys_new/audiohw/PANN/finetune" --holdout_fold=1 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path="/data1/nys_new/audiohw/PANN/Cnn14_mAP=0.431.pth" --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --freeze_base --layer_num 3 --cuda
</pre>

to train CNN14 from scratch, use --model_type="Cnn14" --train_from_scratch:
<pre>
CUDA_VISIBLE_DEVICES=7 python3 pytorch/main.py train --dataset_dir="/data1/nys_new/audiohw/PANN/GTZAN/Data/genres_original" --workspace="/data1/nys_new/audiohw/PANN/finetune" --holdout_fold=1 --model_type="Cnn14" --pretrained_checkpoint_path="" --train_from_scratch --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --cuda
</pre>

## Model
I use 10-fold cross validation for GTZAN classification following paper. That is, 900 audio clips are used for training, and 100 audio clips are used for validation.

## Results
TODO


## Citation
Thanks for the great work PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
https://github.com/qiuqiangkong/audioset_tagging_cnn

[1] Kong, Qiuqiang, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "PANNs: Large-scale pretrained audio neural networks for audio pattern recognition." arXiv preprint arXiv:1912.10211 (2019).
