import pandas as pd
import torch
from typing import List
from torch.utils.data import Dataset
from src.data.dictionary import Dictionary, TokenDictionary


class LanguageModelDataset(Dataset):
    def __init__(
            self,
            data: List[str],
            max_seq_len: int,
            max_char_len: int,
            seq_vocab: TokenDictionary,
            char_vocab: TokenDictionary,
    ):
        self.max_seq_len = max_seq_len
        self.max_char_len = max_char_len
        self.data = data
        self.seq_vocab = seq_vocab
        self.char_vocab = char_vocab

    def __len__(self):
        return len(self.data)

    def n_tokens(self):
        return len(self.seq_vocab)

    def n_chars(self):
        return len(self.char_vocab)

    def __getitem__(self, index):
        max_seq_len = self.max_seq_len
        max_char_len = self.max_char_len

        text = self.data[index]
        tokens = text.split(' ')
        if len(tokens) >= max_seq_len:
            tokens = tokens[: max_seq_len - 2]
        x_tokens = [self.seq_vocab.bos_token]
        y_tokens = tokens

        x_tokens.extend(tokens)
        x_tokens.append(self.seq_vocab.eos_token)

        x_vector, x_mask = self.seq_vocab.encode(x_tokens, max_len=max_seq_len)
        y_vector, y_mask = self.seq_vocab.encode(y_tokens, max_len=max_seq_len)
        x_chars_vector = torch.zeros((max_seq_len, max_char_len), dtype=torch.int64)
        mask_chars = torch.zeros((max_seq_len, max_char_len), dtype=torch.int64)
        for i, token in enumerate(tokens):
            char_vector, mask_c = self.char_vocab.encode(list(token), max_len=max_char_len)
            x_chars_vector[i + 1] = char_vector
            mask_chars[i + 1] = mask_c
            
        return x_vector, y_vector, x_mask, x_chars_vector

    @classmethod
    def from_csv(cls, file_path: str, text_col='text'):
        data_df = pd.read_csv(file_path, usecols=[text_col])

        seq_vocab = TokenDictionary()
        char_vocab = TokenDictionary()

        max_seq_len = 0
        max_token_len = 0
        for i, row in data_df.iterrows():
            tokens = row['text'].split(' ')
            chars = list(set([c for c in row['text']]))

            max_seq_len = max(max_seq_len, len(tokens))
            max_token_len = max(max_token_len, max([len(token) for token in tokens]))

            seq_vocab.add_items(tokens)
            char_vocab.add_items(chars)

        return cls(
            data_df,
            max_seq_len,
            max_token_len,
            seq_vocab,
            char_vocab
        )


class LabelDataset(Dataset):
    def __init__(
            self,
            data_df,
            seq_vocab: TokenDictionary,
            sent_dict: Dictionary,
            topic_dict: Dictionary,
            char_vocab: TokenDictionary = None,
            max_seq_len: int = None,
            max_char_len: int = None,
    ):

        self.data_df = data_df
        self.seq_vocab = seq_vocab
        self.sent_dict = sent_dict
        self.topic_dict = topic_dict
        self.char_vocab = char_vocab

        if max_seq_len is not None:
            self.max_seq_len = max_seq_len
        else:
            self.max_seq_len = self.compute_max_seq_len() + 1

        if max_char_len is not None:
            self.max_char_len = max_char_len
        else:
            self.max_char_len = self.compute_max_char_len() + 1

    def __len__(self):
        return len(self.data_df)

    def n_tokens(self):
        return len(self.seq_vocab)

    def n_sent(self):
        return len(self.sent_dict)

    def n_topic(self):
        return len(self.topic_dict)

    def compute_max_seq_len(self):
        texts = self.data_df['text'].tolist()
        max_seq_len = 0
        for text in texts:
            max_seq_len = max(max_seq_len, len(text.split(' ')))
        return max_seq_len

    def compute_max_char_len(self):
        texts = self.data_df['text'].tolist()
        max_char_len = 0
        for text in texts:
            max_char_len = max(max_char_len, max(len(token) for token in text.split(' ')))
        return max_char_len

    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        x_tokens = row['text'].split(' ')
        if len(x_tokens) > self.max_seq_len:
            x_tokens = x_tokens[:self.max_seq_len]
        n_tokens = len(x_tokens)
        sent_label = row['sentiment']
        topic_label = row['topic']
        x_vector, x_mask = self.seq_vocab.encode(x_tokens, max_len=self.max_seq_len)
        sent_vector = self.sent_dict.index(sent_label)
        topic_vector = self.topic_dict.index(topic_label)

        if self.char_vocab is not None:

            chars_vector = torch.zeros((self.max_seq_len, self.max_char_len), dtype=torch.int64)
            mask_chars = torch.zeros((self.max_seq_len, self.max_char_len), dtype=torch.int64)
            for i, token in enumerate(x_tokens):
                char_vector, mask_c = self.char_vocab.encode(list(token), max_len=self.max_char_len)
                chars_vector[i] = char_vector
                mask_chars[i] = mask_c

            return x_vector, sent_vector, topic_vector, x_mask, chars_vector

        return x_vector, sent_vector, topic_vector, x_mask

    @classmethod
    def from_csv(cls, file_path: str, text_col='text', sent_col='sentiment', topic_col='topic', multi_label=False):
        data_df = pd.read_csv(file_path, usecols=[text_col, sent_col, topic_col])
        data_df = data_df.rename({text_col: 'text', sent_col: 'sentiment', topic_col: 'topic'}, axis='columns')

        seq_vocab = TokenDictionary()
        sent_dict = Dictionary()
        topic_dict = Dictionary()
        char_vocab = TokenDictionary()

        max_seq_len = 0
        for i, row in data_df.iterrows():
            tokens = row['text'].split(' ')
            max_seq_len = max(max_seq_len, len(tokens))

            sentiment = row['sentiment'].split(' ')
            sent_dict.add_items(sentiment)

            topic = row['topic'].split(' ')
            topic_dict.add_items(topic)

            for token in tokens:
                char_vocab.add_items(list(token))

        return cls(
            data_df,
            seq_vocab=seq_vocab,
            sent_dict=sent_dict,
            topic_dict=topic_dict,
            char_vocab=char_vocab,
            max_seq_len=max_seq_len,
        )