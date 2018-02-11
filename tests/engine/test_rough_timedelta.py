import unittest

from elefas.engine.utils import rough_timedelta as rt


class TestRoughTimedelta(unittest.TestCase):
    def test_seconds(self):
        self.assertEquals(rt(0.6), 'less than 1 second')
        self.assertEquals(rt(2.1), '2.1 seconds')
        self.assertEquals(rt(11.111), '11 seconds')
        self.assertEquals(rt(183), '3 minutes 3 seconds')

    def test_negative(self):
        self.assertEquals(rt(-12345), '3 hours 26 minutes')


if __name__ == '__main__':
    unittest.main()
