from ..engine import *


class NumericHyperParameter(BaseHyperParameter):
    def __init__(self, name, start, stop):
        super().__init__(name)
        self.start = start
        self.stop = stop

    def is_int(self):
        return isinstance(self.start, int) and isinstance(self.stop, int)

class Linear(NumericHyperParameter):
    def __init__(self, name, start, stop):
        super().__init__(name, start, stop)


class Exponential(NumericHyperParameter):
    def __init__(self, name, start, stop):
        super().__init__(name, start, stop)
