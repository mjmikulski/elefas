import math
import time
from collections import OrderedDict


class Score:
    def __init__(self, index, hyper_point, scores_dict):
        self.index = index
        self.hyper_point = hyper_point
        self.scores_dict = scores_dict
        self.timestamp = time.time()


class Scores:
    def __init__(self):
        self.scores = []

    def add(self, s):
        self.scores.append(s)

    def best(self, metrics, highest_is_best=True):
        if highest_is_best:
            return max(self.scores, key=lambda x: x.scores_dict.get(metrics, -math.inf), default=-math.inf)
        else:
            return min(self.scores, key=lambda x: x.scores_dict.get(metrics, math.inf), default=math.inf)  # note different default

    def best_p(self, metrics, highest_is_best=True):  # maybe delete
        return self.best(metrics, highest_is_best).hyper_point

    def best_sp(self, metrics, highest_is_best=True):
        b = self.best(metrics, highest_is_best)
        return (b.scores_dict[metrics], b.hyper_point)