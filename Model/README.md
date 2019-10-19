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

### Generate Chunk
Once found the length distribution, I divided documents into 200 words length chunks
- abort chunks where we need to pad 3/2 (i.e chunk length < 66)

### Balance data
This easily to found that we have a unbalance distribution between data, here is the `chunks num` vs `label` plot.  

#### Balance methods
- limit there is only 1000 chunks for each label to train
- for labels contains words less than 1000 chunks, we generate artifical data:
  - pick up 2 chunks from the label and combine them
  - add new chunk into list until we reach 1000 chunks

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
 

