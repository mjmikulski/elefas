from collections import OrderedDict
from copy import deepcopy

import numpy as np

from math import log10, floor

from ..engine import Search, HyperPointer, elegant_numbers as en
from ..hyperparameters import *


class GridSearch(Search):
    def __init__(self):
        self.hs = []
        self.ranges = []
        self.compiled = False

        self.n_accessed = 0
        self.n_total = 0

    def add(self, hparam, n=None, step=None):
        if self.compiled:
            raise RuntimeError('You cannot add hyper-parameters after space was compiled')

        h = deepcopy(hparam)

        points = []

        if isinstance(h, NumericH):
            if n is not None and step is not None: raise ValueError('You can pass either number of points to explore or step, not both.')
            if n is None and step is None: n = 5
            if n is not None:
                if isinstance(h, Linear):
                    if isinstance(h.start, int) and isinstance(h.stop, int):
                        points = np.around(np.linspace(h.start, h.stop, num=n)).astype(int)
                    else:
                        points = np.linspace(h.start, h.stop, num=n)
                elif isinstance(h, Exponential):
                    if isinstance(h.start, int) and isinstance(h.stop, int):
                        points = np.around(np.geomspace(h.start, h.stop, num=n)).astype(int)
                    else:
                        points = np.geomspace(h.start, h.stop, num=n)

            elif step is not None:
                scale = 3 - int(floor(log10(step)))
                def mold(x):
                    return round(x, scale)

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
        elif isinstance(h, Choice):
            pass
        else:
            raise TypeError('Unexpected hyperparameter added to GridSearch')

        self.hs.append(h)
        self.ranges.append(len(h.values))

    def compile(self):
        self.compiled = True
        self.hp = HyperPointer(self.ranges)
        self.n_total = np.prod(self.ranges)

    def __call__(self, *args, **kwargs):
        if not self.compiled: raise RuntimeError('Compile space before accessing points')

        while not self.hp.done:
            self.n_accessed += 1
            pos = self.hp.get()
            d = OrderedDict()
            for i, h in enumerate(self.hs):
                d[h.name] = h.values[pos[i]]
            yield d
            self.hp.move()

    def summary(self, print_fn=print):
        s = '=' * 80 + '\n'
        s += 'Hyper-parameters:\n'
        for h in self.hs:
            s += '  {:>3}  {:20} {} \n'.format(len(h.values), h.name, str(list(h.values)))

        s += '-' * 80 + '\n'
        s += 'Total number of points: {}'.format(self.n_total) + '\n'
        s += 'Points accessed: {}'.format(self.n_accessed) + '\n'
        s += '=' * 80 + '\n'
        print_fn(s)
