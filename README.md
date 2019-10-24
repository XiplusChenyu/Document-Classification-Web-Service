# HeavyWaterProject
Author: Chenyu Xi (cx2219@columbia.edu)

## Timeline
- Start the project at `Oct 16, 2019`     
- First stage is finished on `Oct 18, 2019`  
- Second stage is finished on `Oct 20, 2019` 

### Actions after submission
> not cheating

- I found that sometimes Heroku API needs time to init if it didn't get request for long time. Thus I add a small animation (waiting result) to improve user experience `Oct 22, 2019` 
- To use more AWS service, I replace the heroku API with API Gateway and update the readme `Oct 24, 2019`

## Project Steps

### Train a deep learning model
This part is saved on `Model/`. See <a href=https://github.com/XiplusChenyu/HeavyWaterProject/blob/master/Model/README.md>README</a> for more details

### Build Rest-ful API
This part is saved on `FlaskApi/`. See <a href=https://github.com/XiplusChenyu/HeavyWaterProject/blob/master/FlaskApi/README.md>README</a> for more details



### Build frond end page
This part is saved on `FrontEnd/`. Use Bootstrap, jQuery, HTML/CSS to build a serverless web page. The page is hosted on AWS S3 bucket. We also use **cloudfront** service to make it as Https website to ensure security

<a href=https://d3t9jbj88pt3ap.cloudfront.net/index.html> Sample Website</a>

#### UI and sample result
<img src=https://github.com/XiplusChenyu/HWProject/blob/master/ReadMePics/UI.png>

#### Wait animation
Sometimes you might need to wait seconds for result:
<img src=https://github.com/XiplusChenyu/HWProject/blob/master/ReadMePics/wait.png>

#### Front End Logic:
```
Once Load 
  -> get map files

User hit get result:
  -> Get document 
  -> Send document to Flask API 
  -> Get predictions and display them
```
