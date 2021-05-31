from flask import Flask, jsonify, request
import logging
from flask_cors import CORS
import argparse
from utils.predict_util import load_learner, predict_on_text

logging.basicConfig(level=logging.DEBUG)

# create app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

parser = argparse.ArgumentParser()
parser.add_argument('--sent_config_path', type=str, default='model_done/bilstm-character/sentiment/config.json', help='')
parser.add_argument('--topic_config_path', type=str, default='model_done/bilstm-character/topic/config.json', help='')

args = parser.parse_args()


sent_learner = load_learner(args.sent_config_path)
topic_learner = load_learner(args.topic_config_path)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.json['text']
        prediction = predict_on_text(sent_learner, topic_learner, text)
        # logging.info(str(prediction))
        return jsonify(prediction)


if __name__ == '__main__':
    app.run()
