from shapely.geometry import Point, Polygon

import overlappingspheres
import unittest


class TestFunctionality(unittest.TestCase):

    def test_randompoint_on(self):
        unitsquare = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        y = overlappingspheres.functionality.randompoint_on(unitsquare)
        assert type(y) == Point
        # self.assertEqual(overlappingspheres.greet("Fergus"), "Hello Fergus")

    def test_forces_total(self):
        pt = Point([0, 0])
        pts = [Point([1, 1])]
        y = overlappingspheres.functionality.forces_total(pt, pts)
        assert len(y) == 2
        # self.assertEqual(overlappingspheres.minimum(1, 2, 3), 1)
        # self.assertEqual(overlappingspheres.minimum(1.2, 2.3), 1.2)
        # self.assertEqual(overlappingspheres.minimum(-1.2, -3), -3)


if __name__ == '__main__':
    unittest.main()
