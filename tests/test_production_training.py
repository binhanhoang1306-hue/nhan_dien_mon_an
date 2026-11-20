import tempfile
import unittest
from pathlib import Path

from train_production_model import collect_training_items


class ProductionTrainingDataTests(unittest.TestCase):
    def test_collection_removes_duplicates_and_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            class_a = root / "class_a"
            class_b = root / "class_b"
            class_a.mkdir()
            class_b.mkdir()

            (class_a / "duplicate_1.jpg").write_bytes(b"same image")
            (class_a / "duplicate_2.jpg").write_bytes(b"same image")
            (class_a / "conflict.jpg").write_bytes(b"conflicting image")
            (class_a / "unique_a.jpg").write_bytes(b"unique A")
            (class_b / "conflict.jpg").write_bytes(b"conflicting image")
            (class_b / "unique_b.jpg").write_bytes(b"unique B")

            classes, class_to_idx, items = collect_training_items(root)

        self.assertEqual(classes, ["class_a", "class_b"])
        self.assertEqual(len(items), 3)
        self.assertEqual(
            {Path(path).name for path, _ in items},
            {"duplicate_1.jpg", "unique_a.jpg", "unique_b.jpg"},
        )
        self.assertEqual({label for _, label in items}, set(class_to_idx.values()))


if __name__ == "__main__":
    unittest.main()
