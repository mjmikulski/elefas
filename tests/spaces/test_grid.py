import unittest
from math import sqrt

from elefas.hyperparameters import Exponential
from elefas.spaces import Grid


class Test_Grid(unittest.TestCase):
    def test_int_with_float_step(self):
        space = Grid()

        space.add(Exponential('a', 2, 10), step=1.5)
        space.compile()

        points = [p['a'] for p in space]

        self.assertListEqual(points, [2, 3, 4, 7, 10 ])


    def test_int_with_small_float_step(self):
        space = Grid()

        space.add(Exponential('a', 1, 4), step=sqrt(2))
        space.compile()

        points = [p['a'] for p in space]

        self.assertListEqual(points, [1, 2, 3, 4])

if __name__ == '__main__':
    unittest.main()
