from ..engine import BaseH


class Constant(BaseH):
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value


class Choice(BaseH):
    def __init__(self, name, values):
        super().__init__(name)
        self.values = values

class Boolean(Choice):
    def __init__(self, name):
        super().__init__(name, values=[False, True])