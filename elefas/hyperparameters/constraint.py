from inspect import getfullargspec

from ..engine import BaseH


class Constraint(BaseH):
    def __init__(self, name, f):
        super().__init__(name)
        self.hparams = getfullargspec(f).args
        self.f = f

