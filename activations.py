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

# ReLU
def relu(x):
    out = Tensor(max(0, x.data), (x,), 'relu')

    def _backward():
        x.grad += (1.0 if x.data > 0 else 0.0) * out.grad

    out._backward = _backward

    return out

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    out = Tensor(x.data if x.data > 0 else alpha * x.data, (x,), 'leaky_rely')

    def _backward():
        x.grad += (1.0 if x.data > 0 else alpha) * out.grad

    out._backward = _backward

    return out

# ELU
def elu(x, alpha=1.0):
    out = Tensor(x.data if x.data > 0 else alpha * (math.exp(x.data) - 1), (x,), 'elu')

    def _backward():
        x.gard = (1.0 if x.data > 0 else alpha * math.exp(x.data)) * out.grad

    out._backward = _backward

    return out

def selu(x):
    alpha = 1.6732632423543772
    scale = 1.0507009873554805

    out = Tensor(scale * (x.data if x.data > 0 else alpha * (math.exp(x.data) - 1)), (x,), 'selu')

    def _backward():
        x.grad += (scale * (1.0 if x.data > 0 else alpha * math.exp(x.data))) * out.grad

    out._backward = _backward

    return out

