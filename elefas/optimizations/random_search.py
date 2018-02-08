import math
import random
from collections import OrderedDict
from copy import deepcopy

from ..hyperparameters import *


class Random(Search):
    def __init__(self, points=math.inf):
        super().__init__()
        self.n_points = points

    def add(self, h_param):
        if self.compiled:
            raise RuntimeError('You cannot add hyper-parameters after space was compiled')

        if isinstance(h_param.name, list):
            names = h_param.name
            for name in names:
                h = deepcopy(h_param)
                h.name = name
                self._add(h)
        else:
            h = deepcopy(h_param)
            self._add(h)

    def _add(self, h):
        if isinstance(h, NumericHyperParameter):
            self.h_params.append(h)

        elif isinstance(h, Choice):
            self.h_params.append(h)

        elif isinstance(h, Constraint):
            self.constrains.append(h)
            h.n_points_rejected = 0

        elif isinstance(h, Dependent):
            self.dependent.append(h)
        else:
            raise TypeError('Unexpected hyperparameter added to Random search')

    def __call__(self, *args, **kwargs):
        if not self.compiled:
            raise RuntimeError('Compile space before accessing points')

        while self.n_accessed < self.n_points:
            d = OrderedDict()
            for h in self.h_params:
                if isinstance(h, Choice):
                    d[h.name] = random.choice(h.values)
                elif isinstance(h, Linear):
                    if isinstance(h.start, int) and isinstance(h.stop, int):
                        d[h.name] = random.randint(a=h.start, b=h.stop)  # both endpoints included
                    else:
                        d[h.name] = random.uniform(a=h.start, b=h.stop)
                elif isinstance(h, Exponential):
                    v = h.start * ( (h.stop/h.start) ** random.random() )
                    if isinstance(h.start, int) and isinstance(h.stop, int):
                        d[h.name] = round(v)
                    else:
                        d[h.name] = v

            for h in self.dependent:
                d[h.name] = h.f(**{k: d[k] for k in d if k in h.superior_h_params})
            if self._satisfy_constraints(d):
                self.n_accessed += 1
                yield d

    def _proper_summary(self):
        s = ''
        for h in self.h_params:
            s += '        {:20} {} \n'.format(h.name, h.__class__.__name__)
        return s
