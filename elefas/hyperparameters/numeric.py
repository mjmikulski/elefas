from ..engine import *

class Linear(BoundedH):
    def __init__(self, name, a, b, n=None, step=None):
        super().__init__(name, a, b, n, step)


class Geometric(BoundedH):
    def __init__(self, name, a, b, n=None, step=None):
        super().__init__(name, a, b, n, step)
