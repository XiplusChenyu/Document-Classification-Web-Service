# Flask API
## Structure
```
- FlaskApi

  -- static
    --- IndexToLabel.json (map)
    --- LabelToIndex.json (map)
    --- WordToIndex.json (map)
    --- Model.h5 (pretrained weight)
    
  -- FileUtils.py (process original sentence)
  -- Model.py (pytorch model)
  -- Evaluate.py (get prediction result)
  -- Settings.py (global variables)
  -- app.py (API & Main Function)
```

## API
I built two APIs with same function:

### Heroku API (Not in use now)
API Deploy on Heroku：  
Root path： https://document-classification.herokuapp.com/  
Since Heroku has a 500MB size limit, I deployed caculation endpoint on GCP VM

#### Get index to label map
- Path: `/label_map`
- Params: `None`
- Method: `GET`
##### Sample response:
```json
{
    "1": "DELETION OF INTEREST",
    "10": "REINSTATEMENT NOTICE",
    "11": "EXPIRATION NOTICE",
    "12": "INTENT TO CANCEL NOTICE",
    "13": "APPLICATION",
    ...
}
```

#### Get prediction/score for one document
- Path: `/top_labels`
- Params: `document`; Type: `Form-data`
- Method: `POST`
##### Sample response:
```json
[
    {
        "label": "4",
        "score": 0.901914632320404
    },
    {
        "label": "6",
        "score": 0.08254356980323792
    },
    ...
]
```


### API Gateway

Invoke URL： https://cshve8ptwi.execute-api.us-west-1.amazonaws.com/Apple

<img height=200 src=https://github.com/XiplusChenyu/HWProject/blob/master/ReadMePics/API.png>

Just use API Gateway JavaScript SDK

#### Get index to label map
- Path: `/label_map`
- Params: `None`
- Method: `GET`
##### Sample response body:
```json
{
    "1": "DELETION OF INTEREST",
    "10": "REINSTATEMENT NOTICE",
    "11": "EXPIRATION NOTICE",
    "12": "INTENT TO CANCEL NOTICE",
    "13": "APPLICATION",
    ...
}
```

#### Get prediction/score for one document
- Path: `/top_labels`
- Body Model
```json
{
  "type" : "object",
  "properties" : {
    "document" : {
      "type" : "string"
    }
  }
}
```
- Method: `POST`

##### Sample response body:
```json
[
    {
        "label": "4",
        "score": 0.901914632320404
    },
    {
        "label": "6",
        "score": 0.08254356980323792
    },
    ...
]
```
