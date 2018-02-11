import math
import random
from collections import OrderedDict
from copy import deepcopy

from ..hyperparameters import *


class Random(Search):
    MAX_TRIALS = 10000
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

    def __next__(self):
        if self.n_explored < self.n_points:
            self._next()
            return self.current_point
        else:
            raise StopIteration

    def _next(self):
        trials = 0
        while True:
            self.current_point = OrderedDict()
            for h in self.h_params:
                if isinstance(h, Choice):
                    self.current_point[h.name] = random.choice(h.values)
                elif isinstance(h, Linear):
                    if isinstance(h.start, int) and isinstance(h.stop, int):
                        self.current_point[h.name] = random.randint(a=h.start, b=h.stop)  # both endpoints included
                    else:
                        self.current_point[h.name] = random.uniform(a=h.start, b=h.stop)
                elif isinstance(h, Exponential):
                    v = h.start * ((h.stop / h.start) ** random.random())
                    if isinstance(h.start, int) and isinstance(h.stop, int):
                        self.current_point[h.name] = round(v)
                    else:
                        self.current_point[h.name] = v

            self._process_dependent()

            if self._satisfy_constraints(self.current_point):
                self.n_explored += 1
                break

            trials +=1
            if trials > Random.MAX_TRIALS:
                warnings.warn('Could not satisfy constraints in {} trials. Check if your constraints are correct. '
                              'You can change number of trials by setting Random.MAX_TRIALS. '
                              'If you are not afraid of infinite loop, set it to math.inf')
                raise StopIteration  # aka return

    def _show_h_params(self):
        s = ''
        for h in self.h_params:
            s += '        {:20} {} \n'.format(h.name, h.__class__.__name__)
        return s
