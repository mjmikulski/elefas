from collections import OrderedDict
import numpy as np
import time

from elefas.engine.utils import magnitude, rough_timedelta
from elefas.engine.hyper_pointer import HyperPointer
from elefas.hyperparameters import *
from .search import Search


class Grid(Search):
    def __init__(self):
        super().__init__()
        self.spectra = []
        self.hyper_pointer = None

    def add(self, h_param, *, n=None, step=None):
        self._add(h_param, n=n, step=step)

    def _add_numeric(self, h, **kwargs):
        n = kwargs.get('n', None)
        step = kwargs.get('step', None)
        if n is not None and step is not None:
            raise ValueError('You can pass either number of points to explore or step, not both.')
        if n is None and step is None:
            n = 5

        points = []

        def mold(x):
            return round(x, magn)

        if n is not None:
            if isinstance(h, Linear):
                magn = magnitude((h.stop - h.start) / n)
                if h.is_int():
                    points = np.around(np.linspace(h.start, h.stop, num=n)).astype(int).tolist()
                else:
                    points = np.linspace(h.start, h.stop, num=n).tolist()
                    points = list(map(mold, points))

            elif isinstance(h, Exponential):
                magn = magnitude(h.start)
                if h.is_int():
                    points = np.around(np.geomspace(h.start, h.stop, num=n)).astype(int).tolist()
                else:
                    points = np.geomspace(h.start, h.stop, num=n).tolist()
                    points = list(map(mold, points))

        elif step is not None:
            magn = magnitude(step)

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
                if h.is_int():
                    points = np.unique(np.around(points).astype(int)).tolist()
                points.append(h.stop)

        h.values = points
        self.h_params.append(h)

    def _compile(self):
        for h in self.h_params:
            self.spectra.append(len(h.values))
        self.hyper_pointer = HyperPointer(self.spectra)
        self.n_points = np.prod(self.spectra)

    def __next__(self):
        self._next()
        return self.current_point

    def _next(self):
        while not self.hyper_pointer.done:
            self.current_point = OrderedDict()
            pos = self.hyper_pointer.get()
            for i, h in enumerate(self.h_params):
                self.current_point[h.name] = h.values[pos[i]]
            self.hyper_pointer.move()

            self._update_with_dependent()
            self._update_with_constants()

            if self._satisfy_constraints():
                self.n_explored += 1
                return
        raise StopIteration

    def _show_h_params(self):
        s = ''
        for h in self.h_params:
            s += '  {:>4}  {:20} {} \n'.format(len(h.values), h.name, str(list(h.values)))
        return s

    def _end_summary(self):
        self.time_elapsed = time.time() - self.time_start
        s = '=' * 80 + '\n'
        s += 'Explored {}/{} points in {}\n'.format(self.n_explored, self.n_points, rough_timedelta(self.time_elapsed))
        s += '_' * 80 + '\n'
        return s
