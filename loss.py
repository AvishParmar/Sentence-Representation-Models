'''
author: Sounak Mondal
'''
import torch
import torch.nn as nn

torch.manual_seed(1337)

def cross_entropy_loss(logits, labels):
    labels = torch.Tensor(labels).long()
    return nn.CrossEntropyLoss()(logits.cpu(), labels)
