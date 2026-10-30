import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.pipeline import TranslationPipeline

class TestTranslationPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = TranslationPipeline()

    def test_pipeline_with_empty_input(self):
        result = self.pipeline.process([])
        self.assertEqual(result["sentence"], "")
        self.assertEqual(result["glosses"], [])

    def test_pipeline_with_mock_frames(self):
        # Generate 20 dummy landmark frames
        mock_frames = []
        for _ in range(20):
            mock_frames.append({
                "pose": [{"x": 0.5, "y": 0.5, "z": 0.0}] * 33,
                "leftHand": [{"x": 0.4, "y": 0.6, "z": 0.0}] * 21,
                "rightHand": [{"x": 0.6, "y": 0.6, "z": 0.0}] * 21
            })

        result = self.pipeline.process(mock_frames, mode="isolated")
        self.assertIn("sentence", result)
        self.assertIn("glosses", result)
        self.assertIsInstance(result["processing_time_ms"], float)
        print("Pipeline execution result:", result)

if __name__ == "__main__":
    unittest.main()
