from overlappingspheres.functionality import randompoint_on
from overlappingspheres.functionality import forces_total
from shapely.geometry import Point, Polygon

# import overlappingspheres
import numpy as np
import unittest


class TestFunctionality(unittest.TestCase):

    # def test_randompoint_on(self):
    #     unitsquare = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    #     y = randompoint_on(unitsquare)
    #     assert type(y) == np.array

    def test_forces_total(self):
        pt = [0, 0, 0]
        pts = [[1, 1, 1]]
        y = forces_total(pt, pts)
        assert len(y) == 2
        # self.assertEqual(overlappingspheres.minimum(1, 2, 3), 1)
        # self.assertEqual(overlappingspheres.minimum(1.2, 2.3), 1.2)
        # self.assertEqual(overlappingspheres.minimum(-1.2, -3), -3)


if __name__ == '__main__':
    unittest.main()
