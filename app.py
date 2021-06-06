from flask import Flask, jsonify, request, abort
import logging
from flask_cors import CORS
import argparse
from utils.learner_util import predict_on_text
from classification.learners.classification_learner import ClassificationLearner
import time
import json
logging.basicConfig(level=logging.DEBUG)

# create app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


models_serialization_dir = {
    'bilstm': {
        'sentiment': 'model_done/bilstm/sentiment',
        'topic': 'model_done/bilstm/topic'
    },
    'character': {
        'sentiment': 'model_done/character/sentiment',
        'topic': 'model_done/character/topic'
    },
    'attention': {
        'sentiment': 'model_done/attention/sentiment',
        'topic': 'model_done/attention/topic'
    },
    'character-attention': {
        'sentiment': 'model_done/character_attention/sentiment',
        'topic': 'model_done/character_attention/topic'
    }
}

model_types = [model_type for model_type, _ in models_serialization_dir.items()]
logging.info('Model types: ', model_types)
parser = argparse.ArgumentParser()
parser.add_argument('--sent_config_path', type=str, default='configs/sentiment_config.json', help='')
parser.add_argument('--topic_config_path', type=str, default='configs/topic_config.json', help='')
parser.add_argument('--sent_serialization_dir', type=str, default='model_done/character_attention/sentiment', help='')
parser.add_argument('--topic_serialization_dir', type=str, default='model_done/character_attention/topic', help='')
parser.add_argument('--all_model', type=bool, default=False, help='')
parser.add_argument('--debug', type=int, default=0, help='')
args = parser.parse_args()


if args.all_model:
    # load model bi-lstm
    logging.info("Load all model")
    bilstm_sent_learner = ClassificationLearner.from_serialization(
        models_serialization_dir['bilstm']['sentiment']
    )
    bilstm_topic_learner = ClassificationLearner.from_serialization(
        models_serialization_dir['bilstm']['topic']
    )

    # load model bi-lstm + character embedding
    char_sent_learner = ClassificationLearner.from_serialization(
        models_serialization_dir['character']['sentiment']
    )
    char_topic_learner = ClassificationLearner.from_serialization(
        models_serialization_dir['character']['topic']
    )
    # load model bi-lstm  + attention
    att_sent_learner = ClassificationLearner.from_serialization(
        models_serialization_dir['attention']['sentiment']
    )
    att_topic_learner = ClassificationLearner.from_serialization(
        models_serialization_dir['attention']['topic']
    )

    # load model bi-lstm + character embedding + attention
    char_att_sent_learner = ClassificationLearner.from_serialization(
        models_serialization_dir['character-attention']['sentiment']
    )
    bilstm_char_att_topic_learner = ClassificationLearner.from_serialization(
        models_serialization_dir['character-attention']['topic']
    )

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

    logging.info(f"Load sentiment model from {sent_serialization_dir}")
    logging.info(f"Load topic model from {topic_serialization_dir}")
    sent_learner = ClassificationLearner.from_serialization(sent_serialization_dir)
    topic_learner = ClassificationLearner.from_serialization(topic_serialization_dir)



def predict_all_model(text, model_type):
    if model_type == 'bilstm':
        prediction = predict_on_text(bilstm_sent_learner, bilstm_topic_learner, text)
    elif model_type == 'character':
        prediction = predict_on_text(char_sent_learner, char_topic_learner, text)
    elif model_type == 'attention':
        prediction = predict_on_text(att_sent_learner, att_topic_learner, text)
    elif model_type == 'character-attention':
        prediction = predict_on_text(char_att_sent_learner, char_att_sent_learner, text)
    else:
        prediction = None
    return prediction


logging.info(f'Load all model: {args.all_model}')


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    if request.method == 'POST':
        if request.json is None or "text" not in request.json:
            abort(400, f"text is not found. You must pass text in body")     
        text = request.json['text']
        if args.all_model:
            if "model_type" not in request.json:
                abort(400, f"Model type is not found. You must pass model_type in [{' '.join(model_types)}]")
            model_type = request.json['model_type']
            prediction = predict_all_model(text, model_type)
        else:
            prediction = predict_on_text(sent_learner, topic_learner, text)
        if args.debug == 1:
            logging.info(f' Output of model: {prediction}')
            logging.info(f' Time predict: {time.time() - start_time}s')
        if prediction is None:
            abort(400, f"Model type is not found. You can choice in [{' '.join(model_types)}]")
        return jsonify(prediction)


if __name__ == '__main__':
    app.run()
