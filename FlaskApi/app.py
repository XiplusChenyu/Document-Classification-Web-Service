from Evaluate import evaluator
from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/label/<string:sentence>', methods=['GET'])
def get_label(sentence):
    result = evaluator.label_predict(sentence)
    return jsonify(result)


if __name__ == '__main__':
    app.run()
