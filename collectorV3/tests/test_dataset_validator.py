import gzip
import json
import tempfile
import unittest
from pathlib import Path

from b2d_collector.validate_dataset import (
    PRIMARY_RGB_CALIBRATION_IDS,
    REQUIRED_ANNO,
    REQUIRED_CAMERA_CALIBRATION,
    REQUIRED_MEASUREMENT,
    RGB_IDS,
    validate_clip,
)


def _camera_calibration():
    payload = {key: 0 for key in REQUIRED_CAMERA_CALIBRATION}
    payload["location"] = [0.0, 0.0, 0.0]
    payload["rotation"] = [0.0, 0.0, 0.0]
    payload["intrinsic"] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    payload["world2cam"] = [[1.0, 0.0, 0.0, 0.0]] * 4
    payload["cam2ego"] = [[1.0, 0.0, 0.0, 0.0]] * 4
    payload["fov"] = 90.0
    payload["image_size_x"] = 1600
    payload["image_size_y"] = 900
    return payload


class DatasetValidatorTest(unittest.TestCase):
    def test_minimal_clip(self):
        """The minimal fixture follows the current Base-v1 camera_only contract."""
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            (tmp_path / "anno").mkdir(parents=True)
            (tmp_path / "measurements").mkdir(parents=True)
            (tmp_path / "_collector_meta").mkdir(parents=True)

            (tmp_path / "_collector_meta" / "clip.json").write_text(
                json.dumps({"sensor_profile": "camera_only"}), encoding="utf-8"
            )

            for name in RGB_IDS:
                rgb = tmp_path / "camera" / ("rgb_%s" % name)
                rgb.mkdir(parents=True)
                (rgb / "00000.jpg").write_bytes(b"fake")

            anno_payload = {key: 0 for key in REQUIRED_ANNO}
            anno_payload["sensors"] = {
                sensor_id: _camera_calibration()
                for sensor_id in PRIMARY_RGB_CALIBRATION_IDS
            }
            anno_payload["bounding_boxes"] = []
            with gzip.open(
                str(tmp_path / "anno" / "00000.json.gz"), "wt", encoding="utf-8"
            ) as handle:
                json.dump(anno_payload, handle)

            measurement = {key: {} for key in REQUIRED_MEASUREMENT}
            measurement.update(
                {
                    "schema": "b2d-base-e2e-measurement-v1",
                    "frame_index": 0,
                    "timestamp": 0.0,
                    "sensor_frames": {},
                    "ego": {
                        "location": [0.0, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "acceleration": [0.0, 0.0, 0.0],
                        "speed": 0.0,
                    },
                    "control": {},
                    "navigation": {},
                    "collector": {},
                }
            )
            with gzip.open(
                str(tmp_path / "measurements" / "00000.json.gz"),
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(measurement, handle)

            errors, counts = validate_clip(tmp_path)
            self.assertEqual(errors, [])
            self.assertEqual(counts["anno"], 1)
            self.assertEqual(counts["measurements"], 1)
            for name in RGB_IDS:
                self.assertEqual(counts["camera/rgb_%s" % name], 1)
