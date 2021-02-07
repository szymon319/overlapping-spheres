from overlappingspheres.functionality import randompoint_on
from overlappingspheres.functionality import randomly_scatter
from overlappingspheres.functionality import forces_total
from overlappingspheres.functionality import advance
from overlappingspheres.functionality import shift
from shapely.geometry import Point, Polygon

# import overlappingspheres
import numpy as np
import unittest


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

    def test_shift(self):
        y = shift(100)
        # print(type(y))
        assert type(y) == np.ndarray


if __name__ == '__main__':
    unittest.main()
