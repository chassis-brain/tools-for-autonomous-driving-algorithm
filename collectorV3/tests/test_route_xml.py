from pathlib import Path
import unittest

from b2d_collector.route_xml import RouteCatalog


class RouteXmlTest(unittest.TestCase):
    def test_example_route_is_valid(self):
        root = Path(__file__).resolve().parents[1]
        catalog = RouteCatalog(str(root / "routes" / "custom_parking_exit.xml"))
        self.assertEqual(catalog.validate(), [])
        route = catalog.get("custom-0001")
        self.assertEqual(route.town, "Town13")
        self.assertEqual(route.scenario_name, "ParkingExit")
        self.assertGreaterEqual(len(route.waypoints), 2)
