import unittest
import numpy as np
import tempfile
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_joint_convergence_v1_1 import atomic_pickle, labels_1nm, orders_recorded, select_supplemental, summarize, threshold_summary

class V11Tests(unittest.TestCase):
    def _g(self, i):
        return {"L": 80+i, "W": 70+i/2, "H": 100+i, "P": 300+i, "r": (80+i)/(70+i/2), "fill": .05+i*.001, "sharpness_5nm": float(i)}
    def test_global_top4_and_boundaries(self):
        out = select_supplemental([self._g(i) for i in range(20)])
        self.assertEqual(sum(g["selection"] == "global_sharpness_top4" for g in out), 4)
        self.assertTrue({"high_H", "high_fill", "large_P", "aspect_ratio_boundary"} <= {g["selection"] for g in out})
    def test_complete_401_point_integration_changes_for_intermediate_peak(self):
        base = np.zeros(401); narrow = base.copy(); narrow[123] = 1
        self.assertGreater(np.linalg.norm(labels_1nm(narrow)-labels_1nm(base)), 1e-6)
    def test_1nm_label_has_401_samples(self):
        self.assertEqual(labels_1nm(np.ones(401)).shape, (3,))

    def test_evidence_generation_is_stable(self):
        selected = [dict(self._g(i), selection=("global_sharpness_top4" if i < 4 else ["high_H", "high_fill", "large_P", "aspect_ratio_boundary"][i-4])) for i in range(8)]
        checkpoint = {"meta": {"selected_geometries": selected, "expected_tasks": 32}, "results": {}}
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "checkpoint.pkl"
            cp.write_bytes(b"fixed-checkpoint")
            v1 = Path(td) / "v1.pkl"
            import pickle
            v1.write_bytes(pickle.dumps({"meta": {"expected_tasks": 400}, "results": {}}))
            a = summarize(checkpoint, cp, "POOLHASH", v1)
            b = summarize(checkpoint, cp, "POOLHASH", v1)
            self.assertEqual(a, b)

    def test_over_threshold_comparison_fails(self):
        result = threshold_summary([0.1, 2.31])
        self.assertFalse(result["all_lt_2_3"])
        self.assertFalse(result["mean_lt_1_15"])

    def test_requested_and_retained_orders_must_match(self):
        self.assertTrue(orders_recorded({"a": {"requested_nG": 131, "retained_nG": 121}}))
        self.assertFalse(orders_recorded({"a": {"requested_nG": 131, "retained_nG": 225}}))

    def test_checkpoint_writer_produces_readable_pickle(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "checkpoint.pkl"
            atomic_pickle(path, {"results": {"one": {"status": "ok"}}})
            import pickle
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            self.assertEqual(payload["results"]["one"]["status"], "ok")
            self.assertEqual(list(Path(td).glob("*.tmp")), [])

if __name__ == "__main__": unittest.main()
