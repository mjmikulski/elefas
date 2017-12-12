from inspect import getfullargspec

from ..engine import BaseH


class Dependent(BaseH):
    def __init__(self, name, f):
        super().__init__(name)
        self.f = f
        self.hparams = getfullargspec(f).args