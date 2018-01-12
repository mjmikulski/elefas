import math
import random
from collections import OrderedDict
from copy import deepcopy

from ..hyperparameters import *


class Random(Search):
    def __init__(self, num_points=math.inf):
        super().__init__()
        self.num_points = num_points

    def add(self, hparam):
        if self.compiled:
            raise RuntimeError('You cannot add hyper-parameters after space was compiled')

        if isinstance(hparam.name, list):
            names = hparam.name
            for name in names:
                h = deepcopy(hparam)
                h.name = name
                self._add(h)
        else:
            h = deepcopy(hparam)
            self._add(h)

    def __call__(self, *args, **kwargs):
        if not self.compiled:
            raise RuntimeError('Compile space before accessing points')

        while self.n_accessed < self.num_points:
            d = OrderedDict()
            for h in self.hs:
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
                d[h.name] = h.f(**{k: d[k] for k in d if k in h.hparams})
            if self._satisfy_constraints(d):
                self.n_accessed += 1
                yield d



    def summary(self, print_fn=print):
        pass

    def _add(self, h):
        if isinstance(h, NumericH):
            self.hs.append(h)

        elif isinstance(h, Choice):
            self.hs.append(h)

        elif isinstance(h, Constraint):
            self.constrains.append(h)
            h.n_points_rejected = 0

        elif isinstance(h, Dependent):
            self.dependent.append(h)
        else:
            raise TypeError('Unexpected hyperparameter added to Random search')


    def _compile(self):
        pass
