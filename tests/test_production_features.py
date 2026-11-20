import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from src.production_features import (
    FeaturePipeline,
    extract_features,
    extract_feature_groups,
    extract_hybrid_features,
    preprocess_image,
)


class ProductionFeaturesTests(unittest.TestCase):
    def test_feature_groups_match_combined_feature_vector(self):
        image = np.full((224, 224, 3), 127, dtype=np.uint8)

        groups = extract_feature_groups(image)
        combined = extract_features(image)

        self.assertEqual(combined.shape, (sum(group.size for group in groups.values()),))
        self.assertEqual(groups["color"].shape, (96,))
        self.assertEqual(groups["lbp"].shape, (26,))
        self.assertEqual(groups["glcm"].shape, (40,))
        self.assertTrue(np.isfinite(combined).all())

    def test_hybrid_features_combine_mobilenet_and_hsv(self):
        class StubFeatureExtractor:
            def extract(self, image):
                return np.ones(1280, dtype=np.float32)

        image = np.full((224, 224, 3), 127, dtype=np.uint8)
        features = extract_hybrid_features(image, StubFeatureExtractor())

        self.assertEqual(features.shape, (1376,))
        self.assertTrue(np.isfinite(features).all())

    def test_preprocess_accepts_path_and_pil_image(self):
        image = Image.new("RGB", (80, 120), color=(200, 80, 40))
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            image.save(image_path)

            from_path = preprocess_image(image_path)
            from_image = preprocess_image(image)

        self.assertEqual(from_path.shape, (224, 224, 3))
        self.assertEqual(from_path.dtype, np.uint8)
        np.testing.assert_array_equal(from_path, from_image)

    def test_feature_pipeline_joblib_round_trip(self):
        rng = np.random.default_rng(42)
        training_features = rng.normal(size=(20, 8)).astype(np.float32)
        pipeline = FeaturePipeline(n_components=0.95)
        pipeline.fit_transform(training_features)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "feature_pipeline.joblib"
            joblib.dump(pipeline, model_path)
            restored = joblib.load(model_path)

        transformed = restored.transform(training_features[:2])
        self.assertEqual(transformed.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
