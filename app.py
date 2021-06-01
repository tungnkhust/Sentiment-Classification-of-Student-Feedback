from flask import Flask, jsonify, request
import logging
from flask_cors import CORS
import argparse
from utils.learner_util import load_learner, predict_on_text
from classification.learners.classification_learner import ClassificationLearner
import time
import json
logging.basicConfig(level=logging.DEBUG)

# create app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


serialization_dir = {
    'bilstm': {
        'sentiment': 'model_done/bilstm/sentiment',
        'topic': 'model_done/bilstm/topic'
    },
    'bilstm-character': {
        'sentiment': 'model_done/bilstm-character/sentiment',
        'topic': 'model_done/bilstm-character/topic'
    },
    'bilstm-attention': {
        'sentiment': 'model_done/bilstm-attention/sentiment',
        'topic': 'model_done/bilstm-attention/topic'
    },
    'bilstm-character-attention': {
        'sentiment': 'model_done/bilstm-character-attention/sentiment',
        'topic': 'model_done/bilstm-character-attention/topic'
    }
}

parser = argparse.ArgumentParser()
parser.add_argument('--sent_config_path', type=str, default='configs/sentiment_config.json', help='')
parser.add_argument('--topic_config_path', type=str, default='configs/topic_config.json', help='')
parser.add_argument('--sent_serialization_dir', type=str, default='model_done/bilstm/sentiment', help='')
parser.add_argument('--topic_serialization_dir', type=str, default='model_done/bilstm/topic', help='')
parser.add_argument('--all_model', type=bool, default=False, help='')
parser.add_argument('--debug', type=int, default=0, help='')
args = parser.parse_args()

if args.all_model:
    # load model bi-lstm
    bilstm_sent_learner = ClassificationLearner.from_serialization(
        serialization_dir['bilstm']['sentiment']
    )
    bilstm_topic_learner = ClassificationLearner.from_serialization(
        serialization_dir['bilstm']['topic']
    )

    # load model bi-lstm + character embedding
    bilstm_char_sent_learner = ClassificationLearner.from_serialization(
        serialization_dir['bilstm-character']['sentiment']
    )
    bilstm_char_topic_learner = ClassificationLearner.from_serialization(
        serialization_dir['bilstm-character']['topic']
    )
    try:
        # load model bi-lstm  + attention
        bilstm_att_sent_learner = ClassificationLearner.from_serialization(
            serialization_dir['bilstm-attention']['sentiment']
        )
        bilstm_att_topic_learner = ClassificationLearner.from_serialization(
            serialization_dir['bilstm-attention']['topic']
        )

        # load model bi-lstm + character embedding + attention
        bilstm_char_att_sent_learner = ClassificationLearner.from_serialization(
            serialization_dir['bilstm-character-attention']['sentiment']
        )
        bilstm_char_att_topic_learner = ClassificationLearner.from_serialization(
            serialization_dir['bilstm-character-attention']['topic']
        )
    except:
        print('')
else:
    if args.sent_serialization_dir == '':
        with open(args.sent_config_path, 'r') as pf:
            config = json.load(pf)
            sent_serialization_dir = config['leaner']['serialization_dir']
    else:
        sent_serialization_dir = args.sent_serialization_dir

    if args.topic_serialization_dir == '':
        with open(args.topic_config_path, 'r') as pf:
            config = json.load(pf)
            topic_serialization_dir = config['leaner']['serialization_dir']
    else:
        topic_serialization_dir = args.topic_serialization_dir

    sent_learner = ClassificationLearner.from_serialization(sent_serialization_dir)
    topic_learner = ClassificationLearner.from_serialization(topic_serialization_dir)


def predict_all_model(text, model_type):

    if model_type == 'bilstm':
        prediction = predict_on_text(bilstm_sent_learner, bilstm_topic_learner, text)
    elif model_type == 'bilstm-character':
        prediction = predict_on_text(bilstm_char_sent_learner, bilstm_char_topic_learner, text)
    # elif model_type == 'bilstm-attention':
    #     prediction = predict_on_text(bilstm_att_sent_learner, bilstm_att_topic_learner, text)
    # elif model_type == 'bilstm-character-attention':
    #     prediction = predict_on_text(bilstm_att_sent_learner, bilstm_att_topic_learner, text)
    else:
        prediction = predict_on_text(sent_learner, topic_learner, text)
    return prediction


logging.info(f'Load all model: {args.all_model}')


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    if request.method == 'POST':
        text = request.json['text']

        if args.all_model:
            model_type = request.json['model_type']
            prediction = predict_all_model(text, model_type)
        else:
            prediction = predict_on_text(sent_learner, topic_learner, text)
        if args.debug == 1:
            logging.info(f' Output of model: {prediction}')
            logging.info(f' Time predict: {time.time() - start_time}s')
        return jsonify(prediction)


if __name__ == '__main__':
    app.run()
