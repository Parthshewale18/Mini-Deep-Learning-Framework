from tensor import Tensor
import math

class Optimizer:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

class SGD(Optimizer):
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

class Momentum(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params, lr)
        self.momentum = momentum
        self.v = [Tensor(0.0) for _ in params]

    def step(self):
        for i,p in enumerate(self.params):
            self.v[i] = self.momentum * self.v[i] + p.grad

            p.data -= (self.lr * self.v[i])

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [Tensor(0.0) for _ in params]
        self.v = [Tensor(0.0) for _ in params]
        self.t = 0 # Time step

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= (self.lr * m_hat.data) / (math.sqrt(v_hat.data) + self.epsilon)

        