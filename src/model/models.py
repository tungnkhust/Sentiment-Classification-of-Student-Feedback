import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from src.modules.nn import Attention
from src.training.metrics import get_metrics
from torch.utils.data import DataLoader


class BaseModel(nn.Module):
    def compute_loss(self, sent_out, topic_out, sent_label, topic_label):
        sent_loss = - nn.NLLLoss(reduction='sum')(sent_out, sent_label)
        topic_loss = - nn.NLLLoss(reduction='sum')(topic_out, topic_label)
        return sent_loss + topic_loss

    def evaluate(self, test_dataset, batch_size=32):
        self.eval()

        device = self.device

        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        val_loss = 0.0
        sent_pred = []
        sent_true = []
        topic_pred = []
        topic_true = []

        for i, (x_vector, sent_vector, topic_vector, x_mask) in enumerate(test_loader):
            out = self.forward(x_vector.to(device), x_mask.to(device))
            sent_out = out[0]
            topic_out = out[1]
            loss = self.compute_loss(sent_out, topic_out, sent_vector.to(device), topic_vector.to(device))
            sent_true.extend(sent_vector.tolist())
            topic_true.extend(topic_vector.tolist())

            sent_pred.extend(sent_out.argmax(-1).tolist())
            topic_pred.extend(topic_out.argmax(-1).tolist())

            # compute val loss
            val_loss += (loss.item() - val_loss) / (i + 1)
        # evaluate report
        f1, precision, recall, acc, report = get_metrics(y_true=sent_true, y_pred=sent_pred)

        sent_report = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': acc,
            'report': report
        }
        print('Evaluated sentiment report:')
        print('+----------+-----------+----------+---------+')
        print('|f1_score  |precision  |recall    |accuracy |')
        print('+----------+-----------+----------+---------+')
        print('|{:.4f}    |{:.4f}     |{:.4f}    |{:.4f}   |'.format(f1, precision, recall, acc, ))
        print('+----------+-----------+----------+---------+')

        f1, precision, recall, acc, report = get_metrics(y_true=topic_true, y_pred=topic_pred)

        topic_report = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': acc,
            'report': report
        }

        print('Evaluated topic report:')
        print('+----------+-----------+----------+---------+')
        print('|f1_score  |precision  |recall    |accuracy |')
        print('+----------+-----------+----------+---------+')
        print('|{:.4f}    |{:.4f}     |{:.4f}    |{:.4f}   |'.format(f1, precision, recall, acc, ))
        print('+----------+-----------+----------+---------+')
        return val_loss, sent_report, topic_report

    def from_pretrained(self, model_path):
        self.load_state_dict(torch.load(model_path))


class BasicModel(BaseModel):
    def __init__(
            self,
            vocab_size,
            n_sent=3,
            n_topic=4,
            embed_size=100,
            hidden_size=128,
            n_rnn_layers=2,
            dropout=0.3,
            att_method='general',
            word2vec=None,
            padding_idx=1,
            device='cpu',
    ):
        super(BasicModel, self).__init__()
        self.n_rnn_layer = n_rnn_layers
        self.hidden_size = hidden_size
        self.device = device
        if word2vec is None:
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=int(padding_idx))
        else:
            self.embedding = nn.Embedding.from_pretrained(word2vec, freeze=False, padding_idx=padding_idx)
        self.rnn = nn.LSTM(embed_size, hidden_size, n_rnn_layers,
                           dropout=dropout, bias=True, bidirectional=True)
        self.sent_att = Attention(hidden_size * 2, method=att_method)
        self.topic_att = Attention(hidden_size * 2, method=att_method)
        self.drop = nn.Dropout(dropout)
        self.sent_clf = nn.Linear(hidden_size * 4, n_sent)
        self.topic_clf = nn.Linear(hidden_size * 4, n_topic)
        self.init_weight()

        self.to(device)

    def init_weight(self):
        for layer_p in self.rnn._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    init.xavier_uniform_(self.rnn.__getattr__(p))

        self.sent_att.init_weight()
        torch.nn.init.xavier_uniform_(self.sent_clf.weight)

        self.topic_att.init_weight()
        torch.nn.init.xavier_uniform_(self.topic_clf.weight)

    def init_hidden(self, batch_size):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        h0 = torch.randn(self.n_rnn_layer * 2, batch_size, self.hidden_size)
        c0 = torch.randn(self.n_rnn_layer * 2, batch_size, self.hidden_size)
        return h0.to(device), c0.to(device)

    def forward(self, word_vector: Tensor, mask: Tensor = None):
        bs, seq_len = word_vector.shape

        # compute embedding
        embed = self.embedding(word_vector)
        h_0, c_0 = self.init_hidden(bs)
        embed = torch.transpose(embed, 0, 1)

        # forward to lstm
        encoder_outputs, (h_n, c_n) = self.rnn(embed, (h_0, c_0))
        hidden_state = encoder_outputs
        encoder_outputs = encoder_outputs.transpose(0, 1)
        hidden_state = hidden_state.view(seq_len, bs, 2, self.hidden_size)

        # get context vector of sentence
        forward = hidden_state[-1, :, 0, :]
        backward = hidden_state[0, :, 1, :]
        context = torch.cat([forward, backward], dim=-1)

        # compute attention hidden state for sentiment
        sent_att, sent_att_w = self.sent_att(context, encoder_outputs, mask)
        sent_hidden = torch.cat([sent_att, context], -1)

        # compute attention hidden state for topics
        topic_att, topic_att_w = self.sent_att(context, encoder_outputs, mask)
        topic_hidden = torch.cat([topic_att, context], -1)

        # compute output for classify sentiment
        sent_out = self.sent_clf(self.drop(sent_hidden))
        sent_out = nn.Softmax(dim=-1)(sent_out)

        # compute output for classify topic
        topic_out = self.topic_clf(self.drop(topic_hidden))
        topic_out = nn.Softmax(dim=-1)(topic_out)
        return sent_out, topic_out, sent_att_w, topic_att_w


class GeneralModel(BaseModel):
    def __init__(
            self,
            encoder,
            decoder,
            device=None,
    ):
        super(GeneralModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.to(device)

    def forward(self, word_vector: Tensor, char_vector: Tensor = None, mask: Tensor = None):
        hidden_state = self.encoder(word_vector, char_vector, mask)
        out = self.decoder(hidden_state, mask)
        return out