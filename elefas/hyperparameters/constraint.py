from inspect import getfullargspec

from ..engine import BaseH


class Constraint(BaseH):
    def __init__(self, name, f):
        super().__init__(name)
        self.f = f
        self.hparams = getfullargspec(f).args

    def __str__(self):
        return f'{self.name} <Constraint that depends on {self.hparams}>'

