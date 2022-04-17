'''
author: Sounak Mondal
'''

# std lib imports
from typing import Dict
from pydantic import SequenceError

# external libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

torch.manual_seed(1337)

class SequenceToVector(nn.Module):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``torch.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``torch.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : torch.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : torch.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2, device = 'cpu'):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = "relu"
        self.layers = [nn.Linear(input_dim, input_dim) for layer in range(self.num_layers)]
        # TODO(students): end

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:
        # TODO(students): start
        batch, tokens, embed = vector_sequence.shape
        #print(vector_sequence.shape)
        layer_representations = []
        sequence_mask = torch.reshape(sequence_mask, (batch, tokens))
        word_count = torch.sum(sequence_mask, axis = 1)
       
    
        if training:
            dropoutM = torch.rand(batch, tokens)
            torch.where(dropoutM >= self.dropout, 1, 0)
            word_count = torch.sum(sequence_mask * dropoutM, axis = 1)
            #print(vector_sequence.shape)
            #print(dropoutM.shape)
            sequence_mask *= dropoutM
            
        #print(vector_sequence.shape)
        #print(sequence_mask.shape)
        #print(word_count.shape)

        sequence_mask = sequence_mask.unsqueeze(-1).expand(vector_sequence.size())
        #print(sequence_mask.shape)

        vector_sequence *= sequence_mask

        combined_vector = torch.nan_to_num(torch.sum(vector_sequence, axis=1))
        for i in self.layers:
            combined_vector = i(combined_vector)
            layer_representations.append(combined_vector)
        layer_representations = torch.stack(layer_representations, axis = 1)

        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int, device = 'cpu'):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(input_size = input_dim, hidden_size = input_dim, batch_first = True, num_layers = self.num_layers)
        # TODO(students): end

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:
        # TODO(students): start
        batch, tokens, embed = vector_sequence.shape
        #print(sequence_mask.shape)

        # print(vector_sequence.shape)
        layer_representations = []
        
        sequences, length = sequence_mask.shape
        # (64, 209) - sequence mask. 64 = sequences, 209 = seqeunce length

        sequence_mask = torch.sum(sequence_mask, dim = 1)
        padded = pack_padded_sequence(vector_sequence, sequence_mask, batch_first = True, enforce_sorted = False)
        # lengths = torch.sum()
        # padded = pack_padded_sequence(vector_sequence, sequence_mask batch_first = True, enforced_sorted = False))
        # print(sequence_mask.shape)
        
        gru_out, gru_hidden = self.gru(padded)

        #print(gru_hidden.shape)
        # [4, 64, 50]
        # [64, 4, 50]
        # 64 - batch_size, 4 - num layers, 50 - num embeds
        # batch_size, num_layers, embedding_dim
        combined_vector = gru_hidden[-1]
        layer_representations = gru_hidden.permute(1, 0, 2)
        # TODO(students): end
        
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
