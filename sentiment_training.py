import os
import shutil
from classification.learners.classification_learner import ClassificationLearner

if __name__ == '__main__':
    if os.path.exists('models') is False:
        os.mkdir('models')

    serialization_dir = 'models/SentimentCLF'

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
        max_tokens=100,
        min_count={'tokens': 3},
        embedding_dim=100,
        char_embedding_dim=30,
        ngram_filter_sizes=(3,),
        num_filters=64,
        hidden_size=256,
        dropout=0.4,
        num_layers=2,
        pretrained_w2v_path='pretrained/viki/viki_w2v.txt',
        # pretrained_c2v_path='models/NextTokenLM/pretrained_lm/token_characters.txt',
        # char_encoder_path='models/NextTokenLM/pretrained_lm/char_encoder.pth',
        # seq_encoder_path='models/NextTokenLM/pretrained_lm/seq_encoder.pth'
    )

    sentiment_clf_learner.train(
        lr=0.001,
        weight_decay=0.001,
        batch_size=5,
        num_epochs=7,
        grad_clipping=5,
    )