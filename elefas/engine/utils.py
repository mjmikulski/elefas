from math import floor, log10


def magnitude(x) -> int:
    return 3 - floor(log10(x))
