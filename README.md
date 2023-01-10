# GTZAN classification finetuned on pretrained audio neural networks (PANNs)

Codebase for audiohw 

## Project
The code, dataset and extracted features can be downloaded from https://pan.baidu.com/s/1tiyqrAaLDOrO0A_2zotwHQ?pwd=20vz

提取码：20vz


(P.S. It seems the original dataset downloaded from kaggle has a corrupt file: jazz.00054.wav, I replace it with jazz.00053.wav, so there shoule be 2 same wav files)
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

(2) Extract features(optional: if you have downloaded waveform.h5 from the link, you can skip this step)

(3) Train model

## Model
A 14-layer CNN of PANNs is fine-tuned. I use 10-fold cross validation for GTZAN classification following paper. That is, 900 audio clips are used for training, and 100 audio clips are used for validation.

## Results


## Citation

[1] Kong, Qiuqiang, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "PANNs: Large-scale pretrained audio neural networks for audio pattern recognition." arXiv preprint arXiv:1912.10211 (2019).
