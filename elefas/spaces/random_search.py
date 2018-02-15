import math
import random
import time
import warnings
from collections import OrderedDict
from datetime import timedelta

from elefas.hyperparameters import *
from .search import Search


class Random(Search):
    MAX_TRIALS = 100000

    def __init__(self, points=math.inf, time_limit=math.inf):
        super().__init__()
        self.n_points = points
        self.time_limit = time_limit.total_seconds() if isinstance(time_limit, timedelta) else time_limit

    def add(self, h_param):
        self._add(h_param)

    def __next__(self):
        if self.n_explored >= self.n_points:
            raise StopIteration('All points done')
        if not self.enough_time():
            raise StopIteration('Time limit')

        self._next()
        return self.current_point

    def _next(self):
        trials = 0
        while True:
            self.current_point = OrderedDict()
            for h in self.h_params:
                if isinstance(h, Choice):
                    self.current_point[h.name] = random.choice(h.values)
                elif isinstance(h, Linear):
                    if h.is_int():
                        self.current_point[h.name] = random.randint(a=h.start, b=h.stop)  # both endpoints included
                    else:
                        self.current_point[h.name] = random.uniform(a=h.start, b=h.stop)
                elif isinstance(h, Exponential):
                    v = h.start * ((h.stop / h.start) ** random.random())
                    if h.is_int():
                        self.current_point[h.name] = round(v)
                    else:
                        self.current_point[h.name] = v

            self._update_with_constants()
            self._update_with_dependent()

            if self._satisfy_constraints():
                self.n_explored += 1
                break

            trials += 1
            if trials > Random.MAX_TRIALS:
                warnings.warn(
                    f'Could not satisfy constraints in {trials} trials. Check if your constraints are correct. '
                    'You can change number of trials by setting Random.MAX_TRIALS. '
                    'If you are not afraid of infinite loop, set it to math.inf')
                raise StopIteration('Number of MAX_TRIALS exceeded')

    def _show_h_params(self):
        s = ''
        for h in self.h_params:
            s += '        {:20} {} \n'.format(h.name, h.__class__.__name__)
        return s

    def enough_time(self):
        if self.n_explored < 2:
            return True
        mean_time_for_point = (time.time() - self.time_start) / self.n_explored
        if time.time() + mean_time_for_point < self.time_start + self.time_limit:
            return True
        return False
