from .base_hyper import BaseHyperParameter



class Constant(BaseHyperParameter):
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value


class Choice(BaseHyperParameter):
    def __init__(self, name, values:list):
        super().__init__(name)
        self.values = values


class Boolean(Choice):
    def __init__(self, name):
        super().__init__(name, values=[False, True])
