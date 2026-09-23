""" Tensor class for automatic differentiation"""

import math
class Tensor:
    def __init__(self, data, _children=(), op=''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda:None
        self.op = op

    def __repr__(self):
        return f'Tensor(data={self.data}, grad={self.grad})'

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward

        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        return self + (-other) # reuses add and neg

    def __rsub__(self, other): # Reverse Substraction
        other = other if isinstance(other, Tensor) else Tensor(other)

        return other + (-self)

    def __pow__(self, power):
        assert isinstance(power, (int, float))


        out = Tensor(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += (power * (self.data ** (power - 1)) ) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        return self * (other ** -1)

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        return other * (self ** -1)

    def exp(self):
        out = Tensor(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def log(self):
        out = Tensor(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward

        return out

    def abs(self):
        out = Tensor(abs(self.data), (self,), 'abs')

        def _backward():
            if self.data > 0:
                self.grad += out.grad
            elif self.data < 0:
                self.grad -= out.grad

        out._backward = _backward

        return out

    def tanh(self):
        out = Tensor(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward

        return out

    def zero_grad(self):
        self.grad = 0.0
                
    def backward(self):

        topo = []
    
        visited = set()
    
        def build(v):
    
            if v not in visited:
    
                visited.add(v)
    
                for child in v._prev:
                    build(child)
    
                topo.append(v)
    
        build(self)
    
        self.grad = 1.0
    
        for node in reversed(topo):
            node._backward()