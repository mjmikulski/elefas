import math
import random
from collections import OrderedDict
from copy import deepcopy

from ..hyperparameters import *
from . import Random


class Sobol(Random):
    def __init__(self, points=math.inf):
        super().__init__(points)
