import random
from tensor import Tensor
from activations import relu, selu

class Neuron:
    def __init__(self, nin, activation=relu):
        self.w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Tensor(0.0)
        self.activation = activation

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w,x)), self.b) # z = WX + B
        out = self.activation(act) # a = f(z)
        return out

    def parameters(self):
        return self.w + [self.b]

class Linear:
    def __init__(self, nin, nout, activation=relu):
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())

        return params

class MLP:
    def __init__(self, nin, layers, activation=relu):
        sizes = [nin] + layers
        self.layer = [Linear(sizes[i], sizes[i+1], activation=activation) for i in range(len(layers))]

    def __call__(self, x):
        for layer in self.layer:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layer:
            params.extend(layer.parameters())
        return params

if __name__ == "__main__":

    model = MLP( 2, [4,4,1], activation=selu)
    x = [2,3]
    pred = model(x)[0]
    # suppose we have a target value for the prediction
    target = Tensor(1.0)
    loss = (pred - target)**2
    loss.backward()
    for p in model.parameters():
        print(p.grad)