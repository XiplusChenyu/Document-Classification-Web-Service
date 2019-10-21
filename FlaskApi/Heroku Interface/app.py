from flask import Flask, request, jsonify
from flask_cors import CORS
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


app = Flask(__name__)
CORS(app)
base_url = "http://35.208.155.164:6516"


@app.route('/top_labels', methods=["POST"])
def get_label():
    """
    get result for each document
    :return:
    """
    sentence = request.form.get('document')
    new_url = base_url + "/top_labels?document="
    result = json_from_url2(new_url, document=sentence)
    return jsonify(result)


@app.route('/label_map', methods=['GET'])
def get_label_map():
    """
    Get a list of label-index map
    :return:
    """
    result = json_from_url(base_url+"/label_map")
    return jsonify(result)


if __name__ == '__main__':
    app.run()
