from Evaluate import evaluator
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/top_labels', methods=['GET'])
def get_label():
    """
    get result for each document
    :return:
    """
    try:
        sentence = request.args.get('document')
        result = evaluator.label_predict(sentence)
    except AttributeError:
        result = {
            "error": "No Documents"
        }
    return jsonify(result)


@app.route('/label_map', methods=['GET'])
def get_label_map():
    """
    Get a list of label-index map
    :return:
    """
    result = evaluator.get_label_map()
    return jsonify(result)


if __name__ == '__main__':
    import click


    @click.command()
    @click.option('--debug', is_flag=True)
    @click.option('--threaded', is_flag=True)
    @click.argument('HOST', default='0.0.0.0')
    @click.argument('PORT', default=6516, type=int)  # default GCP port
    def run(debug, threaded, host, port):
        """
        This function handles command line parameters.
        Run the server using
            python server.py
        Show the help text using
            python server.py --help
        """
        app.run(host=host, port=port, debug=debug, threaded=threaded)


    run()
