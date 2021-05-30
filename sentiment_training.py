import os
import shutil
import json
from classification.learners.classification_learner import ClassificationLearner
import argparse


def train(config, checkpoint=False):
    if os.path.exists('models') is False:
        os.mkdir('models')

    serialization_dir = config['serialization_dir']

    if checkpoint is False:
        if os.path.exists(serialization_dir):
            shutil.rmtree(serialization_dir)

    sentiment_clf_learner = ClassificationLearner(
        train_path='data/processed/train.csv',
        text_col='text',
        label_col='sentiment',
        serialization_dir=serialization_dir,
        val_path='data/processed/val.csv',
        test_path='data/processed/test.csv',
        vocab=None,
        vocab_path={'tokens': 'pretrained/viki/viki_w2v_vocab.txt'},
        extend_vocab=True,
        max_tokens=config['max_tokens'],
        min_count=config['min_count'],
        embedding_dim=config['embedding_dim'],
        char_embedding_dim=config['char_embedding_dim'],
        ngram_filter_sizes=config['ngram_filter_sizes'],
        num_filters=config['num_filters'],
        hidden_size=config['hidden_size'],
        dropout=config['dropout'],
        num_layers=config['num_layers'],
        pretrained_w2v_path=config['pretrained_w2v_path'],
        pretrained_c2v_path=config['pretrained_c2v_path'],
        char_encoder_path=config['char_encoder_path'],
        seq_encoder_path=config['seq_encoder_path']
    )

    sentiment_clf_learner.train(
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        grad_clipping=config['grad_clipping'],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cf', '--config', type=str, default='configs/sentiment_config.json', help='')

    args = parser.parse_args()
    with open(args.config, 'r') as pf:
        config = json.load(pf)
        train(config, checkpoint=False)
