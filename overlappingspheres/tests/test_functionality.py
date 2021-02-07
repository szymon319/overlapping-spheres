from overlappingspheres.functionality import randompoint_on
from overlappingspheres.functionality import randomly_scatter
from overlappingspheres.functionality import forces_total
from overlappingspheres.functionality import advance
from overlappingspheres.functionality import shift
from shapely.geometry import Point, Polygon

# import overlappingspheres
import numpy as np
import unittest

from math import log10, floor


def round_sig(x, sig=2):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


class TestFunctionality(unittest.TestCase):

    def test_randompoint_on(self):
        unitsquare = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        y = randompoint_on(unitsquare, 0)
        # print(type(y))
        assert type(y) == np.ndarray

    def test_randomly_scatter(self):
        unitsquare = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        y = randomly_scatter(100, unitsquare)
        # print(type(y))
        assert type(y) == list
        assert len(y) == 100

    def test_forces_total(self):
        pt = [0, 0, 0]
        pts = [[1, 1, 1]]
        y = forces_total(pt, pts)
        assert type(y) == list
        assert len(y) == 2
        # self.assertEqual(overlappingspheres.minimum(1, 2, 3), 1)
        # self.assertEqual(overlappingspheres.minimum(1.2, 2.3), 1.2)
        # self.assertEqual(overlappingspheres.minimum(-1.2, -3), -3)

    def test_advance(self):
        testtest = np.array([[0, 0, 1]])
        y = advance(testtest, 0.0001)
        # print(type(y))
        assert type(y) == np.ndarray

        board = np.array([[0, 0, 1], [5, 0, 1]])
        for i in range(int(1e3)):
            y5 = advance(board, 0.15)
            if y5[0][0] == board[0][0]:
                break
            else:
                board = y5
            print(i)
        expectations = np.array([[2.1, 0, 1], [2.9, 0, 1]])
        assert round_sig(y5[0][0]) == expectations[0][0]
        assert round_sig(y5[1][0]) == expectations[1][0]

        board6 = np.array([[2.4, 0, 1], [2.6, 0, 1]])
        for i in range(int(1e4)):
            y6 = advance(board6, 0.015)
            if y6[0][0] == board6[0][0]:
                break
            else:
                board6 = y6
            print(i)
        expectations6 = np.array([[2.1, 0, 1], [2.9, 0, 1]])
        assert round_sig(y6[0][0]) == expectations6[0][0]
        assert round_sig(y6[1][0]) == expectations6[1][0]

    def test_shift(self):
        y = shift(100)
        # print(type(y))
        assert type(y) == np.ndarray


if __name__ == '__main__':
    unittest.main()
