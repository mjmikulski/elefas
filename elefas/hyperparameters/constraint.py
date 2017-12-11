from ..engine import BaseH


class Constraint(BaseH):
    def __init__(self, name, hparams, f):
        super().__init__(name)
        self.hparams = hparams
        self.f = f

