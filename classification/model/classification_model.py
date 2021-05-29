import torch
from typing import Dict
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.nn import util


class TextClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_feild_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self._text_feild_embedder = text_feild_embedder
        self._encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self._classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(1)

    def forward(
            self,
            tokens,
            label: torch.Tensor=None,
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self._text_feild_embedder(tokens)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(tokens)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self._encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self._classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {
            'probs': probs,
            'logits': logits
        }
        if label is not None:
            loss = torch.nn.functional.cross_entropy(logits, label)
            output['loss'] = loss
            self.accuracy(logits, label)
            self.f1_measure(logits, label)

        return output
    
    def get_metrics(self, reset: bool = False):

        metrics_result = self.f1_measure.get_metric(reset)
        metrics_result['accuracy'] = self.accuracy.get_metric(reset)
        
        return metrics_result
