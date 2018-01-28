from ..engine import *


class NumericHyperParameter(BaseHyperParameter):
    def __init__(self, name, start, stop):
        super().__init__(name)
        self.start = start
        self.stop = stop

class Linear(NumericHyperParameter):
    def __init__(self, name, start, stop):
        super().__init__(name, start, stop)


class Exponential(NumericHyperParameter):
    def __init__(self, name, start, stop):
        super().__init__(name, start, stop)
