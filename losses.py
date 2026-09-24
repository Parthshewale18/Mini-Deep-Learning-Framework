""" Loss Functions for Neural Networks """

import math
from tensor import Tensor

def mse(preds, targets):
    loss = sum([(p - t)**2 for p, t in zip(preds, targets)])
    return loss / len(preds)

def binary_cross_entropy(pred, target):
    loss = -(target * pred.log() + (1 - target) * (1 - pred).log())
    return loss

def softmax(logits):
    exps = [Tensor(math.exp(logit.data)) for logit in logits]
    sum_exps = sum(exps)
    return [exp / sum_exps for exp in exps]