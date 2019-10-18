from Evaluate import evaluator
from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/label/<string:sentence>', methods=['GET'])
def get_label(sentence):
    result = evaluator.label_predict(sentence)
    return jsonify(result)


if __name__ == '__main__':
    import click


    @click.command()
    @click.option('--debug', is_flag=True)
    @click.option('--threaded', is_flag=True)
    @click.argument('HOST', default='0.0.0.0')
    @click.argument('PORT', default=8111, type=int)
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
