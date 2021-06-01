import json
import os
import shutil
from classification.learners.classification_learner import ClassificationLearner
from allennlp.data import Vocabulary
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from utils.utils import plot_confusion_matrix
import argparse
from utils.learner_util import predict_on_text, load_learner


if __name__ == '__main__':
    topic_config_path = 'configs/topic_config.json'
    sent_config_path = 'configs/sentiment_config.json'

    parser = argparse.ArgumentParser()
    parser.add_argument('--sent_config_path', type=str, default='configs/sentiment_config.json', help='')
    parser.add_argument('--topic_config_path', type=str, default='configs/topic_config.json', help='')
    parser.add_argument('--text', type=str, default='Thầy giáo giảng nhiệt tỉnh, hấp dẫn học sinh', help='')

    args = parser.parse_args()

    sent_learner = load_learner(args.sent_config_path)
    topic_learner = load_learner(args.topic_config_path)

    text = args.text
    output = predict_on_text(sent_learner, topic_learner, text)
    print(output)