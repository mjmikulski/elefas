class BaseH:
    def __init__(self, name):
        self.name = name


class NumericH(BaseH):
    def __init__(self, name, start, stop):
        super().__init__(name)
        self.start = start
        self.stop = stop