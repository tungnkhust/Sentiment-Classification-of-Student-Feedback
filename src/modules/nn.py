import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor
from torch.nn import init


class Attention(nn.Module):
    def __init__(self, hidden_size, method='general', cuda='cpu'):
        super(Attention, self).__init__()
        self.method = 'general'
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

        elif self.method == 'concat':
            self.linear = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size)).to(cuda)

        self.init_weight()

    def init_weight(self):
        if self.method == 'general':
            init.xavier_uniform_(self.linear.weight)
        elif self.method == 'concat':
            init.xavier_uniform_(self.linear.weight)
            init.xavier_uniform_(self.weight)

    def score(self, hidden, encoder_outputs):

        if self.method == 'dot':
            score = encoder_outputs.bmm(hidden.view(1, -1, 1)).squeeze(-1)
            return score

        elif self.method == 'general':
            out = self.linear(hidden)
            score = encoder_outputs.bmm(out.unsqueeze(-1)).squeeze(-1)
            return score

        elif self.method == 'concat':
            out = self.linear(torch.cat((hidden, encoder_outputs), 1))
            score = out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)
            return score

    def forward(self, hidden, encoder_outputs, mask=None):
        score = self.score(hidden, encoder_outputs)
        if mask is not None:
            score = score * mask
        att_w = f.softmax(score, -1)
        if mask is not None:
            att_w = att_w * mask
        att_w = att_w.unsqueeze(-1)
        out = encoder_outputs.transpose(-1, -2).bmm(att_w).squeeze(-1)
        return out, att_w


class CNNPoolingLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(CNNPoolingLayer, self).__init__()
        self.cnn = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)


class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_embed_size, char_padding_idx=1, char_len=24):
        super(CharCNN, self).__init__()
        if char_len != 24:
            raise Exception('To use CNN for char embedding, char_len must be 16')

        self.char_embed_size = char_embed_size
        self.embed = nn.Embedding(char_vocab_size, char_embed_size, padding_idx=char_padding_idx)

        self.cnn_3 = nn.Conv1d(char_embed_size, char_embed_size, kernel_size=3, stride=1, padding=char_padding_idx)
        self.cnn_2 = nn.Conv1d(char_embed_size, char_embed_size, kernel_size=2, stride=1, padding=char_padding_idx)
        self.cnn_1 = nn.Conv1d(char_embed_size, char_embed_size, kernel_size=1, stride=1)

        self.cnn_m = torch.Sequential([
            CNNPoolingLayer(in_channel=3 * char_embed_size, out_channel=char_embed_size, kernel_size=3, stride=1,
                            padding=char_padding_idx)
        ])

        self.cnn_f = nn.Conv1d(char_embed_size, char_embed_size, kernel_size=3)

        self.init_weight()

    def init_weight(self):
        torch.nn.init.xavier_uniform(self.cnn_1.weight)
        torch.nn.init.xavier_uniform(self.cnn_2.weight)
        torch.nn.init.xavier_uniform(self.cnn_3.weight)
        for p in self.cnn_m.parameters():
            torch.nn.init.xavier_uniform_(p)
        torch.nn.init.xavier_uniform(self.cnn_f.weight)

    def forward(self, x: Tensor):
        """
        :param x: shape(bs, word_len, word_len)
        :return: word vector (bs, embed_word_size)
        """
        bs, seq_len, word_len = x.shape
        _x = x.view(bs * seq_len, word_len)
        embed = self.embed(_x)  # shape (bs*seq_len, word_len, char_embed_size)
        t = torch.transpose(embed, -2, -1)
        x_3 = self.cnn_3()
        x_2 = self.cnn_2()
        x_1 = self.cnn_1(embed)
        out = torch.cat([x_1, x_2, x_3], dim=1)
        out = self.cnn_m(out)
        out = self.cnn_f(out)
        out = out.view(bs, seq_len, self.char_embed_size * 2)
        return out


class CharRNN(nn.Module):
    def __init__(self, char_vocab_size, char_embed_size, n_rnn_char=1, rnn_dropout=0.1, char_padding_idx=1):
        super(CharRNN, self).__init__()
        self.embed = nn.Embedding(char_vocab_size, char_embed_size, padding_idx=char_padding_idx)
        self.rnn = nn.LSTM(char_embed_size, char_embed_size, n_rnn_char,
                           dropout=rnn_dropout, bias=True, bidirectional=True)
        self.n_rnn_layer = n_rnn_char
        self.char_embed_size = char_embed_size

        self.init_weight()

    def init_weight(self):
        for layer_p in self.rnn._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    init.xavier_uniform_(self.rnn.__getattr__(p))

    def init_hidden(self, batch_size):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        h0 = torch.randn(self.n_rnn_layer * 2, batch_size, self.char_embed_size)
        c0 = torch.randn(self.n_rnn_layer * 2, batch_size, self.char_embed_size)
        return h0.to(device), c0.to(device)

    def forward(self, x: Tensor):
        """
        :param x: shape(bs, seq_len, char_len)
        :return:
        """
        bs, seq_len, char_len = x.shape
        _x = x.view(bs * seq_len, char_len)
        embed = self.embed(_x)  # shape (bs*seq_len, char_len, char_embed_size)
        embed = torch.transpose(embed, 0, 1)
        h0, c0 = self.init_hidden(bs * seq_len)
        hidden_state, _ = self.rnn(embed, (h0, c0))
        hidden_state = hidden_state.view(char_len, bs * seq_len, 2, self.char_embed_size)
        forward = hidden_state[-1, :, 0, :]
        backward = hidden_state[0, :, 1, :]
        out = torch.cat([forward, backward], dim=-1)
        out = out.view(bs, seq_len, -1)
        return out


class WordEmbedding(nn.Module):
    def __init__(
            self,
            word_vocab_size,
            char_vocab_size,
            word_embed_size=100,
            char_embed_size=25,
            word_padding_idx=1,
            dropout=0.1,
            rnn_dropout=0.1,
            use_char_embedding=False,
            char_embed_type='rnn',
            n_rnn_char=1,
            char_padding_idx=1,
            word2vec=None,

    ):
        super(WordEmbedding, self).__init__()

        self.use_char_embedding = use_char_embedding
        if use_char_embedding is False:
            self.embed_size = word_embed_size
        else:
            self.embed_size = word_embed_size + 2 * char_embed_size

        # add char embedding if use_char_embedding is True
        if use_char_embedding:
            if char_embed_type == 'rnn':
                self.char_embed = CharRNN(
                    char_embed_size=char_embed_size,
                    char_vocab_size=char_vocab_size,
                    n_rnn_char=n_rnn_char,
                    rnn_dropout=rnn_dropout,
                    char_padding_idx=char_padding_idx
                )
            elif char_embed_type == 'cnn':
                self.char_embed = CharCNN(
                    char_embed_size=char_embed_size,
                    char_vocab_size=char_vocab_size,
                    char_padding_idx=char_padding_idx
                )
            else:
                self.char_embed = None
        else:
            self.char_embed = None

        self.word_embed = nn.Embedding(word_vocab_size, word_embed_size, padding_idx=word_padding_idx)
        if word2vec is not None:
            self.word_embed.from_pretrained(word2vec, freeze=False, padding_idx=word_padding_idx)

    def forward(self, word_vector: Tensor, chars_vector: Tensor = None):
        """
        :param word_vector: shape(bs, seq_len),
        :param chars_vector: shape(bs, seq_len, word_len)
        :return:
        """
        bs, seq_len = word_vector.shape
        word_embed = self.word_embed(word_vector)
        if chars_vector is not None and self.use_char_embedding:
            char_embed = self.char_embed(chars_vector)
            # chars_mask = word_embed[:, :, :self.char_embed_size * 2] != 0.0
            # char_embed = char_embed * chars_mask
            embed = torch.cat([word_embed, char_embed], dim=-1)
        else:
            embed = word_embed
        return embed


class AttentionDecoder(nn.Module):
    def __init__(self, dropout, hidden_size, n_labels, att_method='general'):
        super(AttentionDecoder, self).__init__()
        self.att = Attention(hidden_size, method=att_method)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size * 2, n_labels)
        self.init_weight()
        self.hidden_size = hidden_size

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, encoder_state, mask=None):
        encoder_vector = encoder_state
        hidden_state = encoder_state
        bs, seq_len, _ = encoder_vector.shape

        hidden_state = hidden_state.transpose(0, 1)
        hidden_state = hidden_state.view(seq_len, bs, 2, int(self.hidden_size / 2))
        forward = hidden_state[-1, :, 0, :]
        backward = hidden_state[0, :, 1, :]
        context = torch.cat([forward, backward], dim=-1)
        att_context, att_w = self.att(context, encoder_vector, mask)
        hidden = torch.cat([att_context, context], -1)
        # hidden = context
        out = self.linear(self.drop(hidden))
        # out = torch.sigmoid(out)
        return out


class LMLSTMDecoder(nn.Module):
    def __init__(
            self,
            hidden_size,
            n_labels
    ):
        super(LMLSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_labels = n_labels
        self.forward_decoder = nn.LSTM(hidden_size, n_labels, bias=True)
        # self.backward_decoder = nn.LSTM(hidden_size, n_labels, bias=True)
        self.init_weight

    def init_weight(self):
        for layer_p in self.forward_decoder._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    init.xavier_uniform_(self.rnn.__getattr__(p))

    def forward(self, x, mask=None):
        out_state, (h_n, c_n) = self.forward_decoder(x)
        out = out_state.transpose(0, 1)
        return out


class SeqLinearDecoder(nn.Module):
    def __init__(
            self,
            hidden_size,
            n_labels
    ):
        super(SeqLinearDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_labels = n_labels
        self.linear_1 = nn.Linear(hidden_size, n_labels, bias=True)

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_1.weight)

    def forward(self, x, mask=None):
        out = self.linear_1(x)
        return out
    

class LSTMEncoder(nn.Module):
    def __init__(
            self,
            embedding,
            embed_size=100,
            hidden_size=128,
            n_rnn_layers=2,
            dropout=0.1,
    ):
        super(LSTMEncoder, self).__init__()
        self.n_rnn_layer = n_rnn_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.rnn = nn.LSTM(embed_size, int(hidden_size / 2), n_rnn_layers,
                           dropout=dropout, bias=True, bidirectional=True)

    def init_weight(self):
        for layer_p in self.rnn._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    init.xavier_uniform_(self.rnn.__getattr__(p))

    def init_hidden(self, batch_size):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        h0 = torch.randn(self.n_rnn_layer * 2, batch_size, int(self.hidden_size / 2))
        c0 = torch.randn(self.n_rnn_layer * 2, batch_size, int(self.hidden_size / 2))
        return h0.to(device), c0.to(device)

    def forward(self, word_vector: Tensor, chars_vector: Tensor = None, mask: Tensor = None):
        """
        :param word_vector: (bs, seq_len)
        :param chars_vector:
        :param mask:
        :return:
        """
        bs, seq_len = word_vector.shape
        embed = self.embedding(word_vector, chars_vector)
        h_0, c_0 = self.init_hidden(bs)
        embed = torch.transpose(embed, 0, 1)
        hidden_state, (h_n, c_n) = self.rnn(embed, (h_0, c_0))
        hidden_state = torch.transpose(hidden_state, 0, 1)
        return hidden_state

    def from_pretrained(self, model_path):
        self.load_state_dict(torch.load(model_path))

class SelfAttentiveLSTMEncoder(nn.Module):
    def __init__(
            self,
            embedding,
            embed_size=100,
            hidden_size=128,
            n_rnn_layers=2,
            rnn_dropout=0.3,
            dropout=0.3,
            num_heads=6,
            n_self_atts=1,
    ):
        super(SelfAttentiveLSTMEncoder, self).__init__()
        self.n_rnn_layer = n_rnn_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.rnn = nn.LSTM(embed_size, int(hidden_size / 2), n_rnn_layers,
                           dropout=rnn_dropout, bias=True, bidirectional=True)
        self.multi_head_atts = nn.ModuleList([nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
                                            for i in range(n_self_atts)])
    def init_weight(self):
        for layer_p in self.rnn._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    init.xavier_uniform_(self.rnn.__getattr__(p))

    def init_hidden(self, batch_size):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        h0 = torch.randn(self.n_rnn_layer * 2, batch_size, int(self.hidden_size / 2))
        c0 = torch.randn(self.n_rnn_layer * 2, batch_size, int(self.hidden_size / 2))
        return h0.to(device), c0.to(device)

    def forward(self, word_vector: Tensor, chars_vector: Tensor = None, mask: Tensor = None):
        """
        :param word_vector: (bs, seq_len)
        :param chars_vector:
        :param mask:
        :return:
        """
        bs, seq_len = word_vector.shape
        embed = self.embedding(word_vector, chars_vector)
        h_0, c_0 = self.init_hidden(bs)
        embed = torch.transpose(embed, 0, 1)
        hidden_state, (h_n, c_n) = self.rnn(embed, (h_0, c_0))
        q = k = v = hidden_state
        for layer in self.multi_head_atts:
            attn_out, att_w = layer(q, k, v)
            q = k = v = attn_out
        out = attn_out.transpose(0, 1)
        return out

    def from_pretrained(self, model_path):
        self.load_state_dict(torch.load(model_path))