import unittest

from b2d_collector.sensors import auxiliary_sensor_specs, collection_sensors


class SensorSplitTest(unittest.TestCase):
    def test_agent_side_suite_is_leaderboard_legal(self):
        sensors = collection_sensors("full")
        types = [item["type"] for item in sensors]
        self.assertEqual(types.count("sensor.camera.rgb"), 6)
        self.assertEqual(types.count("sensor.other.gnss"), 1)
        self.assertEqual(types.count("sensor.other.imu"), 1)
        self.assertEqual(types.count("sensor.speedometer"), 1)
        self.assertFalse(any("depth" in item or "segmentation" in item for item in types))

    def test_rich_suite_is_auxiliary(self):
        sensors = auxiliary_sensor_specs("full")
        ids = {item["id"] for item in sensors}
        self.assertIn("LIDAR_TOP", ids)
        self.assertIn("RADAR_BACK_RIGHT", ids)
        self.assertIn("CAM_FRONT_DEPTH", ids)
        self.assertIn("TOP_DOWN", ids)

