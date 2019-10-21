import urllib
import urllib.request
from urllib.parse import quote
import json


def json_from_url(url):
    response = urllib.request.urlopen(url).read()
    json_data = json.loads(response)
    return json_data


def json_from_url2(url, document):
    url = url + quote(document, 'utf-8')
    response = urllib.request.urlopen(url).read()
    json_data = json.loads(response)
    return json_data


base_url = "http://35.208.155.164:6516/top_labels?document="