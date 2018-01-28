from inspect import getfullargspec

from ..engine import BaseHyperParameter


class Constraint(BaseHyperParameter):
    def __init__(self, name, f):
        super().__init__(name)
        self.f = f
        self.constrained_h_params = getfullargspec(f).args

    def __str__(self):
        return f'{self.name} <Constraint that depends on {self.constrained_h_params}>'

