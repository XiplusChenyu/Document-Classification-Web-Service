# HeavyWaterProject
> I start the project at Wed, Oct 16, the first stage is finished on Fri, Oct 18

## Project Steps
### Train a deep learning model
This part is saved on `Model/`. <a href=https://github.com/XiplusChenyu/HeavyWaterProject/blob/master/Model/README.md>Read me</a> for more details

### Build Rest-ful API
This part is saved on `FlaskApi/`. I choose to build Rest-ful API as a flask application. The API is hosted on a GCP instance. This is a temporary solution. I plan to transfer model to AWS SageMaker for further development

#### API Format
```
Method: Get
Required Parameter: document
Path: http://35.208.155.164:6516/top_labels
Example: http://35.208.155.164:6516/top_labels?document=xxxxxx
Response Format (json):
The top three most likely labels with along with their scores
[
  {
    'label': label1,
    'score': score1
  },
  {
    'label': label2,
    'score': score2
  },
  {
    'label': label3,
    'score': score3
  },
]
```

### Build frond end page
<a href=http://document-classification-buk.s3-website-us-west-1.amazonaws.com/> Sample Website </a>  

This part is saved on `FrontEnd/`. Use Bootstrap, jQuery, HTML/CSS to build a serverless web page. The page is hosted on AWS S3 bucket

```
Front End Logic

Get document -> Send to model API -> Get predictions and display them
```
