from tensor import Tensor
import math

# Sigmoid
def sigmoid(x):
    out = Tensor(1 / (1 + math.exp(-x.data)), (x,), 'sigmoid')

    def _backward():

        s = out.data
        x.grad += s * (1 - s) * out.grad

    out._backward = _backward

    return out

# Tanh
def tanh(x):
    out = Tensor(math.tanh(x.data), (x,), 'tanh')

    def _backward():
        t = math.tanh(x.data)
        x.grad += (1 - t**2) * out.grad

    out._backward = _backward

    return out