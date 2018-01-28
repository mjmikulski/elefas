class BaseHyperParameter:
    """
    This is a base class for all hyper-parameters. It should not be instantiated. Use concrete implementations instead.
    """
    def __init__(self, name):
        self.name = name

