import pickle
import tempfile
import unittest
from pathlib import Path

from scripts import run_geometry_split_v1 as split
from scripts import run_replacement_pool as replacement


class GeometrySplitTests(unittest.TestCase):
    def records(self, count=20):
        records = []
        for index in range(count):
            geometry = (180.0 + index, 90.0 + index / 2.0, 200.0 + index, 450.0 + index)
            identifier = replacement.geometry_id(geometry)
            for pol in ("p", "s"):
                records.append(
                    {
                        "geometry_id": identifier,
                        "L": geometry[0],
                        "W": geometry[1],
                        "H": geometry[2],
                        "P": geometry[3],
                        "pol": pol,
                    }
                )
        return records

    def test_geometry_pairs_and_split_are_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp) / "pool.pkl"
            with pool.open("wb") as handle:
                pickle.dump({"records": self.records()}, handle)
            identifiers = split.load_geometry_ids(pool)
            first = split.build_assignments(identifiers, "A" * 64)
            second = split.build_assignments(list(reversed(identifiers)), "A" * 64)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 20)
            self.assertEqual(sum(item["split"] == "train" for item in first), 16)
            self.assertEqual(sum(item["split"] == "validation" for item in first), 2)
            self.assertEqual(sum(item["split"] == "test" for item in first), 2)
            self.assertEqual(len({item["geometry_id"] for item in first}), 20)

    def test_missing_polarization_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp) / "pool.pkl"
            records = self.records()
            records.pop()
            with pool.open("wb") as handle:
                pickle.dump({"records": records}, handle)
            with self.assertRaisesRegex(ValueError, "exact p/s"):
                split.load_geometry_ids(pool)


if __name__ == "__main__":
    unittest.main()
