from ..engine import BaseH

class Choice(BaseH):
    def __init__(self, name, vals):
        super().__init__(name)
        if not isinstance(vals, list):
            self.vals = [vals]
        else:
            self.vals = vals
        self.n = len(self.vals)

