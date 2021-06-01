import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from utils.learner_util import load_joint_learner


def evaluate(test_df, sent_col, topic_col, config_path):

    learner = load_joint_learner(config_path)
    vocab = learner.vocab
    texts = test_df['text'].tolist()
    sent_true = test_df[sent_col].tolist()
    topic_true = test_df[topic_col].tolist()
    predictions = learner.predict_on_texts(texts)
    sent_pred = [prediction['sentiment'][0] for prediction in predictions]
    topic_pred = [prediction['topic'][0] for prediction in predictions]

    y_sent_true = [vocab.get_token_index(sent, namespace='sentiment') for sent in sent_true]
    y_sent_pred = [vocab.get_token_index(sent, namespace='sentiment') for sent in sent_pred]

    y_topic_true = [vocab.get_token_index(topic, namespace='topic') for topic in topic_true]
    y_topic_pred = [vocab.get_token_index(topic, namespace='topic') for topic in topic_pred]

    # get metric scores
    sent_acc = accuracy_score(y_sent_true, y_sent_pred)
    sent_precision = precision_score(y_sent_pred, y_sent_pred, average='macro')
    sent_recall = recall_score(y_sent_pred, y_sent_pred, average='macro')
    sent_f1 = f1_score(y_sent_pred, y_sent_pred, average='macro')

    sents = list(vocab.get_token_to_index_vocabulary('sentiment').keys())
    sent_report = classification_report(sent_true, sent_pred, labels=sents)
    print(f'Evaluate Sentiment')
    print('Accuracy :', sent_acc)
    print('Precision:', sent_precision)
    print('Recall   :', sent_recall)
    print('F1-Score :', sent_f1)
    print('report')
    print(sent_report)

    print('-'*50)

    # get metric scores of topic
    topic_acc = accuracy_score(y_topic_pred, y_topic_true)
    topic_precision = precision_score(y_topic_pred, y_topic_true, average='macro')
    topic_recall = recall_score(y_topic_pred, y_topic_true, average='macro')
    topic_f1 = f1_score(y_topic_pred, y_topic_true, average='macro')

    topics = list(vocab.get_token_to_index_vocabulary('topic').keys())
    topic_report = classification_report(topic_true, topic_pred, labels=topics)
    print(f'Evaluate Sentiment')
    print('Accuracy :', topic_acc)
    print('Precision:', topic_precision)
    print('Recall   :', topic_recall)
    print('F1-Score :', topic_f1)
    print('report:')
    print(topic_report)
    print('-' * 50)


if __name__ == '__main__':
    config_path = '../configs/joint_config.json'

    test_df = pd.read_csv('../data/processed/test.csv')
    evaluate(test_df, sent_col='sentiment', topic_col='topic', config_path=config_path)
