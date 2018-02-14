import time
import warnings

import math

from .utils import rough_timedelta


class Search:
    """
    This is a base class for all searches. It should not be instantiated. Use concrete implementations instead.
    """

    def __init__(self):
        self.h_params = []
        self.dependent = []
        self.constrains = []

        self.current_point = None

        self.n_points = math.inf
        self.n_explored = 0
        self.compiled = False
        self.time_start = 0
        self.time_elapsed = 0

    def compile(self):
        if self.compiled:
            warnings.warn('Model has been already compiled', RuntimeWarning, stacklevel=2)
        else:
            self._compile()
            self.compiled = True
            self.time_start = time.time()

    def _compile(self):
        pass

    def summary(self, print_fn=print):
        s = self._begin_summary()
        s += self._show_h_params()
        s += self._show_dependent()
        s += self._show_constraints()
        s += self._end_summary()
        print_fn(s)

    def _begin_summary(self):
        s = '_' * 80 + '\n'
        if not self.compiled:
            s += '<NOT COMPILED!>\n'
        if len(self.h_params) > 0:
            s += 'Hyper-parameters:\n'
        else:
            s += 'No hyper-parameters.\n'
        return s

    def _show_h_params(self):
        return '.\n'

    def _show_dependent(self):
        s = ''
        if len(self.dependent) > 0:
            from inspect import signature
            s += 'Dependent:\n'
            for d in self.dependent:
                s += '        {:20} {}\n'.format(d.name, signature(d.f))
        return s

    def _show_constraints(self):
        s = ''
        if len(self.constrains) > 0:
            from inspect import signature
            s += 'Constraints:\n'
            for c in self.constrains:
                s += '{:>6}  {:20} {}\n'.format(c.n_points_rejected, c.name, signature(c.f))
        return s

    def _end_summary(self):
        self.time_elapsed = time.time() - self.time_start
        s = 'Explored {} points in {}\n'.format(self.n_explored, rough_timedelta(self.time_elapsed))
        s += '_' * 80 + '\n'
        return s

    def _satisfy_constraints(self, d):
        for c in self.constrains:
            kwargs = {k: d[k] for k in d if k in c.constrained_h_params}
            if not c.f(**kwargs):
                c.n_points_rejected += 1
                return False
        return True

    def _process_dependent(self):
        for h in self.dependent:
            self.current_point[h.name] = h.f(
                **{k: self.current_point[k] for k in self.current_point if k in h.superior_h_params})

    def __iter__(self):
        if not self.compiled:
            raise RuntimeError('Compile space before exploring it.')
        return self

    def status(self, print_fn=print):
        s = 'Current point: {:4} of {} : {}'.format(self.n_explored, self.n_points, list(self.current_point.items()))
        print_fn(s)
