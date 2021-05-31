from allennlp.data import Vocabulary
from classification.learners.classification_learner import ClassificationLearner
from classification.learners.joint_learner import JointLearner
import json
import torch


def load_joint_learner(config_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(config_path, 'r') as pf:
        config = json.load(pf)
    serialization_dir = config['serialization_dir']
    vocab = Vocabulary.from_files(serialization_dir + '/vocabulary')
    learner = JointLearner(
        vocab=vocab,
        text_col=config['text_col'],
        sent_col=config['sent_col'],
        topic_col=config['topic_col'],
        max_tokens=config['max_tokens'],
        embedding_dim=config['embedding_dim'],
        char_embedding_dim=config['char_embedding_dim'],
        ngram_filter_sizes=config['ngram_filter_sizes'],
        num_filters=config['num_filters'],
        hidden_size=config['hidden_size'],
        dropout=config['dropout'],
        num_layers=config['num_layers'],
    )
    weight_path = config['weight_path']
    if weight_path is None:
        weight_path = serialization_dir + '/best.th'
    learner.load_weight(weight_path)
    return learner


def load_learner(config_path):
    with open(config_path, 'r') as pf:
        config = json.load(pf)
    serialization_dir = config['serialization_dir']
    vocab = Vocabulary.from_files(serialization_dir + '/vocabulary')
    learner = ClassificationLearner(
        vocab=vocab,
        label_col=config['label_col'],
        max_tokens=config['max_tokens'],
        embedding_dim=config['embedding_dim'],
        char_embedding_dim=config['char_embedding_dim'],
        ngram_filter_sizes=config['ngram_filter_sizes'],
        num_filters=config['num_filters'],
        hidden_size=config['hidden_size'],
        dropout=config['dropout'],
        num_layers=config['num_layers']
    )
    weight_path = config['weight_path']
    if weight_path is None:
        weight_path = serialization_dir + '/best.th'
    learner.load_weight(weight_path)
    return learner


def predict_on_text(sentiment_learner, topic_learner, text):
    sentiment, sentiment_confidence = sentiment_learner.predict(text)
    topic, topic_confidence = topic_learner.predict(text)
    return {
        'text': text,
        'sentiment': sentiment,
        'sentiment_confidence': str(sentiment_confidence),
        'topic': topic,
        'topic_confidence': str(topic_confidence)
    }
