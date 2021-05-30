import os
import shutil
from classification.learners.classification_learner import ClassificationLearner
from allennlp.data import Vocabulary
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from utils.utils import plot_confusion_matrix


if __name__ == '__main__':
    serialization_dir = 'models/SentimentCLF'
    vocab = Vocabulary.from_files(serialization_dir + '/vocabulary')
    sentiment_clf_learner = ClassificationLearner(
        vocab=vocab,
        max_tokens=80,
        embedding_dim=100,
        char_embedding_dim=30,
        ngram_filter_sizes=(3,),
        num_filters=64,
        hidden_size=256,
        dropout=0.4,
        num_layers=2
    )

    sentiment_clf_learner.load_weight(serialization_dir + '/best.th')
    sentiment_clf_learner.vocab.save_to_files(serialization_dir + '/vocabulary')

    test_df = pd.read_csv('data/processed/test_drop.csv')
    texts = test_df['text'].tolist()
    sent_true = test_df['sentiment'].tolist()
    sent_pred = sentiment_clf_learner.predict_on_texts(texts)

    y_true = [vocab.get_token_index(sent, namespace='labels') for sent in sent_true]
    y_pred = [vocab.get_token_index(sent, namespace='labels') for sent in sent_pred]

    # get metric scores
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='marco')
    recall = recall_score(y_true, y_pred, average='marco')
    f1 = f1_score(y_true, y_pred, average='marco')

    labels = list(vocab.get_token_to_index_vocabulary('labels').keys())

    report = classification_report(sent_true, sent_pred, labels=labels)

    print('Accuracy :', acc)
    print('Precision:', precision)
    print('Recall   :', recall)
    print('F1-Score :', f1)

    if os.path.exists('results') is False:
        os.mkdir('results')

    with open('results/sentiment_scores.txt', 'w') as pf:
        pf.write(f'Accuracy score :{acc}\n')
        pf.write(f'Precision score:{precision}\n')
        pf.write(f'Recall score   :{recall}\n')
        pf.write(f'F1 score       :{f1}\n')
        pf.write('Detail:\n')
        pf.write(report)

    cm = confusion_matrix(sent_true, sent_pred, labels=labels)
    plot_confusion_matrix(cm, target_names=labels, title='Sentiment Confusion Matrix', normalize=False)
    plot_confusion_matrix(cm, target_names=labels, title='Sentiment Confusion Matrix Normalize', normalize=True)

    test_df['sentiment_predict'] = sent_pred
    false_pred_df = test_df[test_df['sentiment'] != test_df['sentiment_predict']]
    false_pred_df.to_csv('results/false_predictions.csv', index=False)

