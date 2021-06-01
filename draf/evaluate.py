import json
import os
import shutil
from classification.learners.classification_learner import ClassificationLearner
from allennlp.data import Vocabulary
import pandas as pd
from utils.learner_util import load_learner, evaluate


if __name__ == '__main__':
    sent_config_path = '../model_done/bilstm/sentiment/config.json'
    topic_config_path = '../model_done/bilstm/topic/config.json'

    # sent_config_path = 'model_done/bilstm-character/sentiment/config.json'
    # topic_config_path = 'model_done/bilstm-character/topic/config.json'

    print('topic_config_path', topic_config_path)
    print('sent_config_path', sent_config_path)
    test_df = pd.read_csv('../data/processed/test.csv')
    sent_learner = load_learner(sent_config_path)
    topic_learner = load_learner(topic_config_path)
    evaluate(sent_learner, test_df, result_path='../results')
    evaluate(topic_learner, test_df, result_path='../results')