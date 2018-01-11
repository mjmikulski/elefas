from copy import deepcopy

from ..engine import Search


class Random(Search):
    def __init__(self, num_points):
        super().__init__()

    def add(self, hparam, n=None, step=None):
        if self.compiled:
            raise RuntimeError('You cannot add hyper-parameters after space was compiled')

        if isinstance(hparam.name, list):
            names = hparam.name
            for name in names:
                h = deepcopy(hparam)
                h.name = name
                self._add(h, n, step)
        else:
            h = deepcopy(hparam)
            self._add(h, n, step)

    def __call__(self, *args, **kwargs):
        pass

    def summary(self, print_fn=print):
        pass

    def _add(self):
        pass

    def _compile(self):
        pass
