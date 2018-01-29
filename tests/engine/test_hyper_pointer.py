import unittest

from elefas.engine import *


class TestHyperPointer(unittest.TestCase):
    def test_content(self):
        hp = HyperPointer([7, 12, 3])

        results = []
        while not hp.done:
            results.append(hp.get())
            hp.move()

        self.assertEqual(len(results), 7*12*3)  # number of points is ok

        self.assertEqual(len(results), len(set(results)))  # all unique

        self.assertIn((0,0,0), results)  # first
        self.assertIn((3,3,1), results)  # sth in the middle
        self.assertIn((6,11,2), results) # last


    def test_construction_errors(self):
        self.assertRaises(ValueError, HyperPointer, [3,1,2])
        self.assertRaises(ValueError, HyperPointer, [7,0])
        self.assertRaises(ValueError, HyperPointer, [1])


if __name__ == '__main__':
    unittest.main()