class Search:
    def __init__(self):
        self.dependent = []
        self.constrains = []
        self.hs = []
        self.n_accessed = 0

        self.compiled = False

    def _satisfy_constraints(self, d):
        for c in self.constrains:
            kwargs = {k: d[k] for k in d if k in c.hparams}
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
