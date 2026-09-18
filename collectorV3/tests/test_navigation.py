from types import SimpleNamespace
import unittest

from b2d_collector.navigation import RouteProgress


def point(x, y):
    return SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=0.0))


class NavigationTest(unittest.TestCase):
    def test_navigation_advances_and_returns_world_coordinates(self):
        progress = RouteProgress()
        progress.set_plan([(point(float(i), 0.0), SimpleNamespace(value=4)) for i in range(20)])
        signal = progress.update(SimpleNamespace(x=5.1, y=0.0, z=0.0))
        self.assertEqual(signal.near_xy, (7.0, 0.0))
        self.assertEqual(signal.far_xy, (17.0, 0.0))
        self.assertEqual(signal.near_command, 4)
