import unittest

from elefas.engine import Scores, Score
from elefas.engine.utils import rough_timedelta as rt


class TestScores(unittest.TestCase):
    def test_best(self):
        scores = Scores()

        hyper_point = {'momentum': 0.8}
        s = Score(1, hyper_point, {'accu': 0.4})
        scores.add(s)

        hyper_point = {'momentum': 0.98}
        s = Score(1, hyper_point, {'accu': 0.49})
        scores.add(s)

        hyper_point = {'momentum': 0.9}
        s = Score(1, hyper_point, {'accu': 0.45})
        scores.add(s)

        best = scores.best('accu')
        self.assertEquals(best['momentum'], 0.98)


    def test_best_with_missing(self):
        scores = Scores()

        hyper_point = {'momentum': 0.8}
        s = Score(1, hyper_point, {'accu': 0.4})
        scores.add(s)

        hyper_point = {'momentum': 0.98}
        s = Score(1, hyper_point, {'accu': 0.49, 'loss': 1.2})
        scores.add(s)

        hyper_point = {'momentum': 0.9}
        s = Score(1, hyper_point, {'loss': 1.1})
        scores.add(s)

        best = scores.best('loss', highest_is_best=False)
        self.assertEquals(best['momentum'], 0.9)


if __name__ == '__main__':
    unittest.main()
