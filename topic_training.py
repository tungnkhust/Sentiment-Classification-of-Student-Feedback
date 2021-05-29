import os
import shutil
from classification.learners.classification_learner import ClassificationLearner

if __name__ == '__main__':
    if os.path.exists('models') is False:
        os.mkdir('models')

    serialization_dir = 'models/TopicCLF'

    if os.path.exists(serialization_dir):
        shutil.rmtree(serialization_dir)

    sentiment_clf_learner = ClassificationLearner(
        train_path='data/processed/train_drop.csv',
        serialization_dir=serialization_dir,
        val_path='data/processed/val_drop.csv',
        test_path='data/processed/test_drop.csv',
        vocab=None,
        vocab_path={'tokens': 'pretrained/viki/viki_w2v_vocab.txt'},
        extend_vocab=True,
        max_tokens=80,
        min_count={'tokens': 2},
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
        batch_size=256,
        num_epochs=50,
        grad_clipping=5,
    )

    sentiment_clf_learner.vocab.save_to_files(serialization_dir + '/vocabulary')