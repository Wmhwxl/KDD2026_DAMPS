# PROJECT NAME
This repository contains the PyTorch implementation for our research paper.
Title: Enhancing Multimodal Recommendation via Multimodal Representation Calibration in Spectral Domain (KDD 2026)

## INSTALLATION
Python 3.8.20
PyTorch 2.4.1
numpy  1.22.4  
spacy   3.7.2 
scikit-lear 1.3.2
torchvision 0.19.1   

## DATA
Download from Google Drive: [Baby/Sports/Elec](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing)
The data already contains text and image features extracted from Sentence-Transformers and CNN.

An alternative dataset for short-video recommendations: [MicroLens](https://drive.google.com/drive/folders/14UyTAh_YyDV8vzXteBJiy9jv8TBDK43w?usp=drive_link).

* Please move your downloaded data into this dir for model training.

## USAGE
1. Place the downloaded data (e.g., "baby") into the data/ directory.
2. Navigate to src/ and python main.py -m ${model} -d ${dataset} -g ${gpu_id}. (model: SMORE, MGCN ..) {dataset: baby, sports, clothing...} 
We have already inserted DAMPS into the MGCN model. If you want to see the original MGCN model for comparison, please comment out lines 105 and 153 in the MGCN.py file. The code for the DAMPS framework is in the DAMPS.py file
