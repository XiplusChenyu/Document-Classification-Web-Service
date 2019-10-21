# HeavyWaterProject
Author: Chenyu Xi (cx2219@columbia.edu)

## Timeline
- Start the project at `Oct 16, 2019`     
- First stage is finished on `Oct 18, 2019`  
- Second stage is finished on `Oct 21, 2019`  

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

```
Front End Logic:
Once Load 
  -> get map files

User hit get result:
  -> Get document 
  -> Send document to Flask API 
  -> Get predictions and display them
```
