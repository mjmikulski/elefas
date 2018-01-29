class Search:
    """
    This is a base class for all searches. It should not be instantiated. Use concrete implementations instead.
    """
    def __init__(self):
        self.h_params = []
        self.dependent = []
        self.constrains = []

        self.n_accessed = 0

        self.compiled = False

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

    def _satisfy_constraints(self, d):
        for c in self.constrains:
            kwargs = {k: d[k] for k in d if k in c.constrained_h_params}
            if not c.f(**kwargs):
                c.n_points_rejected += 1
                return False
        return True

    def _compile(self):
        pass

    def compile(self):
        if self.compiled:
            raise RuntimeError('Already compiled')
        else:
            self.compiled = True
            self._compile()

    def _begin_summary(self):
        s = '_' * 80 + '\n'
        if not self.compiled:
            s += '<NOT COMPILED!>\n'
        if len(self.h_params) > 0:
            s += 'Hyper-parameters:\n'
        else:
            s += 'No hyper-parameters.\n'
        return s

    def _end_summary(self):
        s = 'Points accessed: {}'.format(self.n_accessed) + '\n'
        s += '_' * 80 + '\n'
        return s
