import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.database.database as database
import backend.main as main


class ApiWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        database.DB_PATH = self.temp_path / "test.sqlite"
        database.init_db()
        main.UPLOAD_DIR = self.temp_path / "uploads"
        main.UPLOAD_DIR.mkdir()
        self.ready_patch = patch.object(main.sensor_service, "ready", True)
        self.predict_patch = patch.object(
            main.sensor_service,
            "predict",
            return_value={
                "prediction": "Normal",
                "confidence": 0.92,
                "probabilities": {"Normal": 0.92, "pH_High": 0.08},
            },
        )
        self.ready_patch.start()
        self.predict_patch.start()
        self.client = TestClient(main.app)

    def tearDown(self):
        self.predict_patch.stop()
        self.ready_patch.stop()
        self.temp_dir.cleanup()

    def test_manual_analysis_is_saved_and_cnn_stays_pending(self):
        response = self.client.post(
            "/api/predict",
            data={
                "ph": "6.2",
                "ec": "1.8",
                "water_temp": "24",
                "humidity": "70",
                "air_temp": "26",
            },
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["sensor"]["prediction"], "Normal")
        self.assertEqual(result["image"]["status"], "pending_cnn_model")
        self.assertEqual(result["fusion"]["status"], "normal_condition")
        self.assertGreater(result["observation_id"], 0)

        history = self.client.get("/api/history").json()["items"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["sensor_probabilities"]["Normal"], 0.92)
        self.assertEqual(self.client.get("/api/sensor/latest").json()["sensor_prediction"], "Normal")

    def test_esp32_reading_keeps_timestamp_and_source(self):
        response = self.client.post(
            "/api/esp32/sensor",
            json={
                "device_id": "lettuce-rig-01",
                "timestamp": "2026-09-28T10:15:00Z",
                "ph": 6.2,
                "ec": 1.8,
                "water_temp": 24,
                "humidity": 70,
                "air_temp": 26,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["device_id"], "lettuce-rig-01")
        latest = self.client.get("/api/sensor/latest").json()
        self.assertEqual(latest["source"], "esp32")
        self.assertEqual(latest["timestamp"], "2026-09-28T10:15:00Z")

    def test_image_upload_is_saved_without_inventing_visual_result(self):
        png_header = b"\x89PNG\r\n\x1a\n" + b"test-image"
        response = self.client.post(
            "/api/predict",
            data={
                "ph": "6.2",
                "ec": "1.8",
                "water_temp": "24",
                "humidity": "70",
                "air_temp": "26",
            },
            files={"image": ("leaf.png", png_header, "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["image"]["predicted_class"], "unknown")
        self.assertEqual(result["image"]["status"], "pending_cnn_model")
        saved_path = Path(self.client.get("/api/history").json()["items"][0]["image_path"])
        self.assertTrue(saved_path.is_file())

    def test_invalid_image_content_and_sensor_bounds_are_rejected(self):
        invalid_image = self.client.post(
            "/api/image/predict",
            files={"image": ("leaf.png", b"not an image", "image/png")},
        )
        invalid_sensor = self.client.post(
            "/api/predict/combined",
            json={"ph": 18, "ec": 1.8, "water_temp": 24, "humidity": 70, "air_temp": 26},
        )

        self.assertEqual(invalid_image.status_code, 415)
        self.assertEqual(invalid_sensor.status_code, 422)


if __name__ == "__main__":
    unittest.main()
