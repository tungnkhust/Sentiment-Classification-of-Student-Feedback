import json
import os
import shutil
from classification.learners.classification_learner import ClassificationLearner
from allennlp.data import Vocabulary
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from utils.utils import plot_confusion_matrix
from utils.predict_util import load_learner


def evaluate(test_df, label_col, config_path):

    learner = load_learner(config_path)
    vocab = learner.vocab
    texts = test_df['text'].tolist()
    label_true = test_df[label_col].tolist()
    label_pred = learner.predict_on_texts(texts)
    y_true = [vocab.get_token_index(label, namespace='labels') for label in label_true]
    y_pred = [vocab.get_token_index(label, namespace='labels') for label in label_pred]

    # get metric scores
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='marco')
    recall = recall_score(y_true, y_pred, average='marco')
    f1 = f1_score(y_true, y_pred, average='marco')

    labels = list(vocab.get_token_to_index_vocabulary('labels').keys())
    report = classification_report(label_true, label_pred, labels=labels)

    print('Accuracy :', acc)
    print('Precision:', precision)
    print('Recall   :', recall)
    print('F1-Score :', f1)

    if os.path.exists('results') is False:
        os.mkdir('results')

    with open(f'results/{label_col}_score.txt', 'w') as pf:
        pf.write(f'Accuracy score :{acc}\n')
        pf.write(f'Precision score:{precision}\n')
        pf.write(f'Recall score   :{recall}\n')
        pf.write(f'F1 score       :{f1}\n')
        pf.write('Detail:\n')
        pf.write(report)

    cm = confusion_matrix(label_true, label_pred, labels=labels)
    plot_confusion_matrix(cm, target_names=labels, title=f'Confusion Matrix ({label_col})', normalize=False)
    plot_confusion_matrix(cm, target_names=labels, title=f'Confusion Matrix Normalize ({label_col})', normalize=True)

    test_df['sentiment_predict'] = label_pred
    false_pred_df = test_df[test_df['sentiment'] != test_df['sentiment_predict']]
    false_pred_df.to_csv('results/false_sentiment_predictions.csv', index=False)


if __name__ == '__main__':
    topic_config_path = 'configs/topic_config.json'
    sent_config_path = 'configs/sentiment_config.json'

    test_df = pd.read_csv('data/processed/test.csv')
    evaluate(test_df, label_col='sentiment', config_path=topic_config_path)
    evaluate(test_df, label_col='topic', config_path=sent_config_path)