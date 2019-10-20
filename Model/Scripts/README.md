# Guideline
## DatabaseBuilder.ipynb
Use this file for data pre-processing:  
- Index original label and sentence
- Divide document into 200-length chunks, pad or slice
- Data clean/ balance
- Group chunks into three sets, 80% for training and 10% for validation and 10% for test, sets are saved as h5py datasets

## Settings.py
Contains global parameters for this project

## FileUtils.py
Contains necessary functions to read files, serialize and deserialize python objects

## Dataloader.py
Generate pytorch dataloader used for model training

## TextCNN.py
CNN model for documents classification


## TextCRNN.py
GRU+CNN model for documents classification

## TrainUtils.py
Contains functions for calculate BCE loss, prediction accuracy

## Train.py
Contains functions used for train/validate/test model

## Evaluate.py
Contains functions to predict label with pretrained models

## TrainAndAnalysis(TextCNN).ipynb/ TrainAndAnalysis(RCNN).ipynb
- Train Models
- Print/Save logs
- Result Analysis
- Model Evaluate
