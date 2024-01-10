import unittest

from CalculateObject import CalculateObject


class CalculateObjectTest(unittest.TestCase):

    def test_bonus_method(self):
        calcObject: CalculateObject = CalculateObject()
        self.assertEquals(calcObject.check_for_bonus_condition())


if __name__ == '__main__':
    unittest.main()