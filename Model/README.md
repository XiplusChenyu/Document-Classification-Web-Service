# Model Training and Analysis

I tried two model structure for this task. The first one is TextCNN model, a traditional model for document classification, and the other one is RCNN model, which add an extra GRU layer to catch the temporal feature.

## Project Structure
```
- Model: Root path for model training project
  - Data: contains original Data, and mapping files
  - LogSave: contains logs generated during model training
  - ModelSave: save model weights
  - Scripts: contains codes
```

## Data Preprocess
### Data Analysis
The first step is to figure out how the original data looks like
- Index label & words (store `index-word` / `index-label` dictionary into `Data/` folder)
  - Remove words only occurs once
  - ADD `<UNK>` and `<PAD>` words
  - Vocab size = 30997
- Find the distribution of documents length 

<img height=200 src=https://github.com/XiplusChenyu/HWProject/blob/master/ReadMePics/LengthD.png>

### Generate Chunk
Once found the length distribution, I divided documents into 200 words length chunks
- abort chunks where we need to pad 3/2 (i.e chunk length < 66)

### Balance data
This easily to found that we have a unbalance distribution between data, here is the `chunks num` vs `label` plot.  

<img height=200 src=https://github.com/XiplusChenyu/HWProject/blob/master/ReadMePics/ChunkD.png>

#### Balance methods
- limit there is max 2000 chunks for each label to train
- for labels contains words less than 1000 chunks, we reuse the chunks (i.e. over-sampling)

### Unbalance data
We also build a dataset which contains all chunks from all labels to compare model perfermance

## Model Design
Two Model is used for this project

### Model Structure
#### TextCNN model
```
Model(
  (embedding): Embedding(300997, 256, padding_idx=1)
  (convs): ModuleList(
    (0): Conv2d(1, 256, kernel_size=(2, 256), stride=(1, 1))
    (1): Conv2d(1, 256, kernel_size=(3, 256), stride=(1, 1))
    (2): Conv2d(1, 256, kernel_size=(4, 256), stride=(1, 1))
  )
  (dropout): Dropout(p=0.7, inplace=False)
  (fc): Linear(in_features=768, out_features=14, bias=True)
  (softmax): Softmax(dim=1)
)
```
### TextRCNN model
```
CRNNModel(
  (embedding): Embedding(300997, 256, padding_idx=1)
  (gruLayer): GRU(256, 128, num_layers=2, batch_first=True, bidirectional=True)
  (gruLayerF): Sequential(
    (0): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): Dropout(p=0.6, inplace=False)
  )
  (convs): ModuleList(
    (0): Conv2d(1, 256, kernel_size=(2, 256), stride=(1, 1))
    (1): Conv2d(1, 256, kernel_size=(3, 256), stride=(1, 1))
    (2): Conv2d(1, 256, kernel_size=(4, 256), stride=(1, 1))
  )
  (dropout): Dropout(p=0.7, inplace=False)
  (fc): Linear(in_features=768, out_features=14, bias=True)
  (softmax): Softmax(dim=1)
)
```

### Training Stage

#### GCP
Train models on GCP VM with cuda supported, use `tmux` with `jupyter-notebook` to coding on local browser and keep kernel running on background

#### Train Model
- Loss Function: BCE
- Model selection: prediction accuracy

Use validation dataset to select model during training, save model in `ModelSave/` folder. Use `LogSave/` folder to record train/test logs to visualize data.

##### Sample Train log:
<img height=200 src=https://github.com/XiplusChenyu/HWProject/blob/master/ReadMePics/Sample-T.png>

##### Train TextCNN model On balanced set (50 epochs)

<img height=200 src="https://github.com/XiplusChenyu/HWProject/blob/master/ReadMePics/CNNA.png"><img height=200 src="https://github.com/XiplusChenyu/HWProject/blob/master/ReadMePics/CNNL.png">


##### Train TextCNN model On non-balanced set (50 epochs)

<img height=200 src="https://github.com/XiplusChenyu/HWProject/blob/master/ReadMePics/FCNNA.png"><img height=200 src="https://github.com/XiplusChenyu/HWProject/blob/master/ReadMePics/FCNNL.png">


##### Train TextRCNN model On balanced set (20 epochs)

<img height=200 src="https://github.com/XiplusChenyu/HWProject/blob/master/ReadMePics/CRNNA.png"><img height=200 src="https://github.com/XiplusChenyu/HWProject/blob/master/ReadMePics/CRNNL.png">

##### Analysis

From the plot we can see the `GRU+CNN` Model has much faster training speed. However, all models suffer from overfitting somehow. So our dataset might be improved in the future

### Model Evaluation

I evaluate our models on
  - test set (chunks)
  - 1000 random full documents picked from orignal csv file

#### Full documents evaluation
  - divide document into chunks
  - predict labels of chunks, sum their score up and use the label has the largest score

#### Result

|   Model  |  Train Set | Accuracy (chunks) | Accuracy (documents) |
|:--------:|:----------:|:-----------------:|:----------------------:|
|  TextCNN |  Balanced  | 76.96% |95.3% |
| TextRCNN |  Balanced  | 73.49% |88.5% |
|  TextCNN | Unbalanced | 81.90% |91.6% |

- Thus we pickup the model weights trained on TextCNN model with balanced sets for our web service

#### Confusion Matrix


<img height=300 src="https://github.com/XiplusChenyu/HWProject/blob/master/ReadMePics/CNNM.png"><img height=300 src="https://github.com/XiplusChenyu/HWProject/blob/master/ReadMePics/CRNNM.png"><img height=300 src="https://github.com/XiplusChenyu/HWProject/blob/master/ReadMePics/FCNNM.png">

