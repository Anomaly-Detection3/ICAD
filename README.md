
## Requirements
* numpy
* matplotlib
* pandas
* torch
* pytorch-lightning>=1.3.0
* torchmetrics>=0.3.0
* python-dotenv

## Model Training and Predict

bash

#=======# WADI 
#-------train-------  
python DDPM_main.py --epochs 50 --learning_rate 1e-3 --hidden_size 32 --batch_size 16 --noise_steps 100
--window_size 50 seed_train=42
#-----predict-----
python DDPM_main.py --batch_size 16 --window_size 50 seed_test=14


#=======# SWAT
#------train------- 
python DDPM_main.py --epochs 200 --learning_rate 1e-3 --hidden_size 32 --batch_size 16 --noise_steps 100
--window_size 50 seed_train=42 img_size=51
#------predict------
python DDPM_main.py --batch_size 16 --window_size 50 seed_test=80


#=======# EO
#------train------ 
python DDPM_main.py  --epochs 200 --learning_rate 1e-3 --hidden_size 32 --batch_size 16 --noise_steps 100
--window_size 40 seed_train=42
#------predict-----
python DDPM_main.py --batch_size 16 --window_size 40 seed_test=42


#=======# PSM
#-------train------ 
python DDPM_main.py  --epochs 300 --learning_rate 1e-3 --hidden_size 12 --batch_size 16 --noise_steps 100
--window_size 50 seed_train=42
#-------predict-------
python DDPM_main.py --batch_size 16 --window_size 50 seed_test=100


#=======# HAI
#-------train------ 
python DDPM_main.py  --epochs 400 --learning_rate 1e-3 --hidden_size 32 --batch_size 16 --noise_steps 100
--window_size 50 seed_train=42
#-------predict------
python DDPM_main.py --batch_size 16 --hidden_siz 32 --window_size 50 seed_test=7


Competing methods
All of the anomaly detectors in our paper are implemented in Python. 
We list their publicly available implementations below.

One-SVM, ECOD, ALAD, Lunar, IF, LOF: 
we directly use pyod (python library of anomaly detection approaches) (see baselines.py);

DSVDD: https://github.com/lukasruff/Deep-SVDD
COUTA: https://github.com/xuhongzuo/couta
COCA: https://github.com/lukasruff/Deep-SVDD
GDN: https://github.com/d-ailin/GDN
USAD: https://github.com/hoo2257/USAD-Anomaly-Detecting-Algorithm
MAD: https://github.com/mangushev/mtad-gat
GDN: https://link.zhihu.com/?target=https%3A//github.com/d-ailin/GDN
GRELEN: https://github.com/Vicky-51/GRELEN
GLUE: https://github.com/chufangao/glue
CST-GL: https://github.com/chufangao/glue
OMNIANOMALY: https://github.com/NetManAIOps/OmniAnomaly
USAD: https://github.com/manigalati/usad
INTERFUSION: https://github.com/zhhlee/InterFusion
A_T: https://github.com/thuml/Anomaly-Transformer
PeFAD: https://github.com/xu737/PeFAD
IMDiffusion: https://github.com/17000cyh/IMDiffusion
DDMT: https://github.com/Lelantos42/DDMT
DiffAD: https://github.com/ChunjingXiao/DiffAD

NOTE: EO dataset provides time-series data from 430 sensors (e.g., temperature, flow rate, liquid level, pressure) that characterize a complex chemical production process.
IMPORTANT: This is commercially sensitive data provided by our industrial partners and its use is strictly restricted to non-commercial academic research.