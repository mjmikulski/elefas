class BaseH:
    def __init__(self, name):
        self.name = name


# class NumericH(BaseH):
#     def __init__(self, name):
#         super().__init__(name)


class BoundedH(BaseH):
    def __init__(self, name, a, b, n=None, step=None):
        super().__init__(name, a, b)
        self.n = n
        self.step = step