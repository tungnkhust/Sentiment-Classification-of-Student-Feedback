import torch
from typing import Union, List
from collections import Counter


class Dictionary(object):
    def __init__(self, item2idx: dict = None):
        """
        :param item2idx: A dict map token to index.
        """
        if item2idx is None:
            item2idx = {}
        self.item2idx = item2idx
        self.idx2item = {index: item for item, index in item2idx.items()}

    def check_in(self, item):
        if item in self.item2idx:
            return True
        else:
            return False

    def add_item(self, item):
        """
        Add item into Dictionary.
        :param item: (str) item.
        :return: The index of item was added.
        """
        if item in self.item2idx:
            return self.item2idx[item]
        else:
            index = len(self)
            self.item2idx[item] = index
            self.idx2item[index] = item
            return index

    def add_items(self, items):
        """
        Add many items into Dictionary.
        :param items: (list) list of tokens.
        :return: List indies of tokens were added.
        """
        indies = [self.add_item(item) for item in items]
        return indies

    def index(self, item):
        """
        Get index of specified item.
        :param item: (str) item.
        :return: The index of item.
        """
        if item in self.item2idx:
            return self.item2idx[item]
        else:
            raise KeyError(f"Item '{item}' is not existed in dictionary!")

    def item(self, index):
        """
        Get item at index position.
        :param index: (int) index of item
        :return: item
        """
        if int(index) >= len(self):

            raise IndexError(f"The index {index} is out of dictionary!")
        else:
            return self.idx2item[index]

    def __len__(self):
        return len(self.item2idx)

    def save(self, file_path: str, delimiter='\t'):
        """
        Save dictionary as file (.txt).
        :param file_path: file path.
        :param delimiter: (str) default='\t'.
        :return:
        """
        with open(file_path, 'w') as pf:
            for item, index in self.item2idx.items():
                pf.write(item + delimiter + str(index) + '\n')
            pf.close()

    def get_items(self):
        return list(self.item2idx.keys())

    @classmethod
    def load(cls, file_path: str, delimiter='\t'):
        """
        Load from file (.txt).
        :param file_path: path of file.
        :param delimiter: (str) default='\t'.
        :return: Dictionary.
        """
        with open(file_path, 'r') as pf:
            lines = pf.readlines()
            lines = [line.replace('\n', '') for line in lines]
            item2idx = {}
            for line in lines:
                item = line.split(delimiter)
                item2idx[item[0]] = int(item[1])
        return cls(item2idx)

    @classmethod
    def from_items(cls, items: list):
        items = list(set(items))
        item2idx = {item: idx for idx, item in enumerate(items)}
        return cls(item2idx)

    def encode(self, items: List[str]):
        vector = torch.zeros(len(self), dtype=torch.int64)
        for item in items:
            vector[self.item2idx[item]] = 1
        return vector

    def decode(self, indices: list):
        return [self.item(idx) for idx in indices]


class TokenDictionary(Dictionary):
    def __init__(self, item2idx=None, padding_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
        """
        :param token2idx: A dict map token to index.
        :param unk_token: Unknown token.
        :param bos_token: Begin of sequence token.
        :param eos_token: End of sequence token.
        """
        super(TokenDictionary, self).__init__(item2idx)

        if padding_token:
            self.padding_token = padding_token
            self.padding_idx = self.add_item(padding_token)

        if unk_token:
            self.unk_token = unk_token
            self.unk_idx = self.add_item(unk_token)
        else:
            self.unk_token = None

        if bos_token:
            self.bos_token = bos_token
            self.bos_idx = self.add_item(bos_token)

        if eos_token:
            self.eos_token = eos_token
            self.eos_idx = self.add_item(eos_token)

    def index(self, item):
        if item in self.item2idx:
            return self.item2idx[item]
        else:
            return self.unk_idx

    def encode(self, tokens: list, max_len: int):
        """
        Convert string to index vector
        :param tokens: list of tokens
        :param max_len: max len of sequence
        :return:
        """
        indices = [self.index(token) for token in tokens]

        seq_len = len(indices)
        if max_len is None:
            max_len = seq_len

        if max_len < seq_len:
            indices = indices[:max_len]

        vector = torch.zeros(max_len, dtype=torch.int64)
        vector[:len(indices)] = torch.tensor(indices, dtype=torch.int64)
        vector[len(indices):] = self.padding_idx
        mask = torch.zeros(max_len, dtype=torch.uint8)
        mask[:len(indices)] = torch.ones(len(indices), dtype=torch.int64)
        return vector, mask

    def decode(self, x: Union[torch.Tensor, list], padding: bool = False):
        """
        Convert a tensor of token indices to string.
        :param x:
            if type list: list of index.
            if type Tensor: shape is (bs, seq_len)
        :param padding: if True is convert include padding_token else no.
        :return: str
        """
        if isinstance(x, list):
            if padding:
                tokens = [self.item(idx) for idx in x]
            else:
                tokens = [self.item(idx) for idx in x if idx != self.padding_idx]
            return tokens
        elif isinstance(x, torch.Tensor):
            if len(x.shape) != 2:
                raise Exception(f"Num of demension have to be 2 not is {len(x.shape)}!")
            else:
                if padding:
                    tokens = [[self.item(int(idx)) for idx in items] for items in x]
                else:
                    tokens = [[self.item(int(idx)) for idx in items if idx != self.padding_idx] for items in x]
                return tokens
        raise TypeError(f"Type of input have to list or Tensor not is {type(x)}")


def build_vocab(texts: List[str], thresh_num=5):
    count = Counter()
    seq_vocab = TokenDictionary()
    max_seq_len = 0
    max_char_len = 0
    for text in texts:
        text = text.replace('\n', '')
        tokens = text.split(' ')
        for token in tokens:
            count[token] += 1
        max_seq_len = max(max_seq_len, len(tokens))
    for token in count:
        if count[token] > 2:
            max_char_len = max(max_char_len, len(token))
        if count[token] >= thresh_num:
            if any(char.isdigit() for char in token) is False:
                seq_vocab.add_item(token)
    return seq_vocab, max_seq_len, max_char_len


def build_char_vocab(tokens):
    char_vocab = TokenDictionary()
    for token in tokens:
        char_vocab.add_items(list(token))
    return char_vocab


def get_label_dict(labels: List[str], delimiter=';'):
    label_dict = Dictionary()
    for label in labels:
        label = label.replace('\n', '')
        ls = label.split(delimiter)
        if '' in ls:
            ls.remove('')
        if ' ' in ls:
            ls.remove(' ')
        ls = [l.replace(' ', '') for l in ls]
        label_dict.add_items(ls)
    return label_dict


def get_tag_dict(bio_tags: List[str], delimiter=' '):
    tag_dict = TokenDictionary(unk_token=None, bos_token=None, eos_token=None)
    for bio in bio_tags:
        bio = bio.replace('\n', '')
        tags = bio.split(delimiter)
        for tag in tags:
            if tag != ' ' and tag != ''  and tag != '\n':
                tag_dict.add_item(tag.strip(' '))
    return tag_dict