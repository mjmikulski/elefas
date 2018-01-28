from inspect import getfullargspec

from ..engine import BaseHyperParameter


class Dependent(BaseHyperParameter):
    def __init__(self, name, f):
        super().__init__(name)
        self.f = f
        self.superior_h_params = getfullargspec(f).args