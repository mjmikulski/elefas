from collections import OrderedDict
from copy import deepcopy
from math import log10, floor

import numpy as np

from ..hyperparameters import *


class Grid(Search):
    def __init__(self):
        super().__init__()
        self.ranges = []
        self.n_total = 0
        self.hp = None

    def add(self, hparam, *, n=None, step=None):
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
        if not self.compiled: raise RuntimeError('Compile space before accessing points')

        while not self.hp.done:
            pos = self.hp.get()
            d = OrderedDict()
            for i, h in enumerate(self.hs):
                d[h.name] = h.values[pos[i]]
            for h in self.dependent:
                d[h.name] = h.f(**{k: d[k] for k in d if k in h.hparams})
            if self._satisfy_constraints(d):
                self.n_accessed += 1
                yield d
            self.hp.move()

    def summary(self, print_fn=print):
        s = '_' * 80 + '\n'
        s += 'Hyper-parameters:\n'
        for h in self.hs:
            s += '  {:>4}  {:20} {} \n'.format(len(h.values), h.name, str(list(h.values)))

        if len(self.constrains) > 0:
            from inspect import signature
            s += 'Constraints:\n'
            for c in self.constrains:
                s += '{:>6}  {:20} {}\n'.format(c.n_points_rejected, c.name, signature(c.f))

        s += '=' * 80 + '\n'
        s += 'Total number of points: {}'.format(self.n_total) + '\n'
        s += 'Points accessed: {}'.format(self.n_accessed) + '\n'
        s += '_' * 80 + '\n'
        print_fn(s)

    def _add(self, h, n, step):
        if isinstance(h, NumericH):
            self._add_numeric(h, n, step)

        elif isinstance(h, Choice):
            self.hs.append(h)
            self.ranges.append(len(h.values))

        elif isinstance(h, Constraint):
            self.constrains.append(h)
            h.n_points_rejected = 0

        elif isinstance(h, Dependent):
            self.dependent.append(h)
        else:
            raise TypeError('Unexpected hyperparameter added to GridSearch')

    def _add_numeric(self, h, n, step):
        def mold(x):
            return round(x, scale)

        points = []
        if n is not None and step is not None:
            raise ValueError('You can pass either number of points to explore or step, not both.')
        if n is None and step is None:
            n = 5
        if n is not None:
            if isinstance(h, Linear):
                scale = 3 - int(floor(log10((h.stop - h.start) / n)))
                if isinstance(h.start, int) and isinstance(h.stop, int):
                    points = np.around(np.linspace(h.start, h.stop, num=n)).astype(int).tolist()
                else:
                    points = np.linspace(h.start, h.stop, num=n).tolist()
                    points = list(map(mold, points))
            elif isinstance(h, Exponential):
                scale = 3 - int(floor(log10(h.start)))
                if isinstance(h.start, int) and isinstance(h.stop, int):
                    points = np.around(np.geomspace(h.start, h.stop, num=n)).astype(int).tolist()
                else:
                    points = np.geomspace(h.start, h.stop, num=n).tolist()
                    points = list(map(mold, points))

        elif step is not None:
            scale = 3 - int(floor(log10(step)))

            if isinstance(h, Linear):
                points = [h.start]
                x = mold(h.start + step)
                while x < h.stop:
                    points.append(x)
                    x = mold(x + step)
                points.append(h.stop)
            elif isinstance(h, Exponential):
                points = [h.start]
                x = h.start * step
                while x < h.stop:
                    points.append(x)
                    x *= step
                points.append(h.stop)
        h.values = points
        self.hs.append(h)
        self.ranges.append(len(h.values))

    def _compile(self):
        self.hp = HyperPointer(self.ranges)
        self.n_total = np.prod(self.ranges)

    def _satisfy_constraints(self, d):
        for c in self.constrains:
            kwargs = {k: d[k] for k in d if k in c.hparams}
            if not c.f(**kwargs):
                c.n_points_rejected += 1
                return False
        return True
