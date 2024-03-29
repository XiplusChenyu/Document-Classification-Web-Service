# Document Classification Web Service
Author: Chenyu Xi (cx2219@columbia.edu)

## Project Steps
### Train a deep learning model
This part is saved on `Model/`. See <a href=https://github.com/XiplusChenyu/HeavyWaterProject/blob/master/Model/README.md>README</a> for more details

### Build Rest-ful API
This part is saved on `FlaskApi/`. See <a href=https://github.com/XiplusChenyu/HeavyWaterProject/blob/master/FlaskApi/README.md>README</a> for more details



### Build frond end page
This part is saved on `FrontEnd/`. Use Bootstrap, jQuery, HTML/CSS to build a serverless web page. The page is hosted on AWS S3 bucket. We also use **cloudfront** service to make it as Https website to ensure security

<a href=https://d3t9jbj88pt3ap.cloudfront.net/index.html> Sample Website</a> **Closed**

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
  -> Send document to API Gateway 
  -> Get predictions and display them
```
