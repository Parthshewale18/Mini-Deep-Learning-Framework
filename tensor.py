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
        # Forward pass
        out = Tensor(self.data + other.data, (self, other), '+')
        # Gradient rule
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        # Forward pass
        out = Tensor(self.data * other.data, (self, other), '*')
        # Gradient rule
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

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
            node._backward 