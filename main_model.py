'''
author: Sounak Mondal
'''
# inbuilt lib imports:
from typing import List, Dict, Tuple
import os

# external libs
import torch
import torch.nn as nn

# project imports
from sequence_to_vector import DanSequenceToVector, GruSequenceToVector

torch.manual_seed(1337)

class MainClassifier(nn.Module):
    def __init__(self,
                 seq2vec_choice: str,
                 vocab_size: int,
                 embedding_dim: int,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 device: str = 'cpu') -> 'MainClassifier':
        """
        It is a wrapper model for DAN or GRU sentence encoder.
        The initializer typically stores configurations in private/public
        variables, which need to accessed during the call (forward pass).
        We also define the trainable variables in the initializer.

        Parameters
        ----------
        seq2vec_choice : ``str``
            Name of sentence encoder: "dan" or "gru".
        vocab_size : ``int``
            Vocabulary size used to index the data instances.
        embedding_dim : ``int``
            Embedding matrix dimension
        num_layers : ``int``
            Number of layers of sentence encoder to build.
        num_classes : ``int``
            Number of classes that this Classifier chooses from.
        """
        super(MainClassifier, self).__init__()
        self.device = device
        # Construct and setup sequence_to_vector model

        if seq2vec_choice == "dan":
            self._seq2vec_layer = DanSequenceToVector(embedding_dim, num_layers, device = device).to(device)
        else:
            self._seq2vec_layer = GruSequenceToVector(embedding_dim, num_layers, device = device).to(device)

        # Trainable Variables
        torch.manual_seed(42)
        self._embedding_layer = nn.Embedding(vocab_size, embedding_dim).to(device)
        self._classification_layer = nn.Linear(embedding_dim, num_classes).to(device)

    def forward(self,
             inputs,
             training=False):
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        inputs : ``str``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, max_tokens_num) and entries are indices of tokens
            in to the vocabulary. 0 means that it's a padding token. max_tokens_num
            is maximum number of tokens in any text sequence in this batch.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.
        """
        inputs = inputs.to(self.device)
        embedded_tokens = self._embedding_layer(inputs)
        tokens_mask = torch.where(inputs !=0, 1,0).float()
        outputs = self._seq2vec_layer(embedded_tokens, tokens_mask, training)
        classification_vector = outputs["combined_vector"]
        layer_representations = outputs["layer_representations"]
        logits = self._classification_layer(classification_vector)
        return {"logits": logits, "layer_representations": layer_representations}
