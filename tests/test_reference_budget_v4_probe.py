import json
import unittest
from pathlib import Path
from unittest import mock

from pipeline_supervisor import file_digest
from scripts import probe_reference_budget_v4 as probe


ROOT = Path(__file__).resolve().parents[1]


class ReferenceBudgetV4ProbeTests(unittest.TestCase):
    def test_protocol_sources_are_frozen_and_tasks_are_exact(self):
        protocol = probe.load_protocol()
        tasks = probe.build_tasks(protocol)
        self.assertEqual(len(tasks), 6)
        self.assertEqual(len({task["id"] for task in tasks}), 6)
        self.assertEqual({task["pol"] for task in tasks}, {"p", "s"})
        self.assertEqual(
            {(task["requested_nG"], task["retained_nG"], task["step_nm"]) for task in tasks},
            {(750, 729, 0.5), (850, 841, 1.0), (850, 841, 0.5)},
        )
        for key in ("source_v3_probe", "source_v3_checkpoint", "source_v3_protocol", "source_v3_runner"):
            item = protocol[key]
            self.assertEqual(file_digest(ROOT / item["path"]), item["sha256"])

    def test_protocol_keeps_gate_and_training_closed(self):
        protocol = json.loads(probe.PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertFalse(protocol["training_allowed"])
        self.assertFalse(protocol["gate_registration_allowed"])
        self.assertTrue(protocol["formal_validation_allowed_only_after_independent_audit"])
        self.assertEqual(protocol["thresholds"]["mean_joint_dE00_lt"], 1.15)
        self.assertEqual(protocol["thresholds"]["all_joint_dE00_lt"], 2.3)

    def test_only_final_order_and_spectral_comparisons_are_decisive(self):
        failed = {"passed": False}
        passed = {"passed": True}
        protocol = probe.load_protocol()
        tasks = probe.build_tasks(protocol)
        with mock.patch.object(probe, "compare", side_effect=[failed, passed, passed]), mock.patch.object(
            probe, "validate_result"
        ), mock.patch.object(probe, "load_source_anchor", return_value={"p": {}, "s": {}}), mock.patch.object(
            probe, "binding", return_value={"path": "x", "sha256": "A"}
        ), mock.patch.object(probe, "file_digest", return_value="A"):
            results = {task["id"]: {} for task in tasks}
            evidence = probe.summarize(results, tasks, Path("checkpoint.pkl"), protocol)
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["classification"], "candidate_budget_supported")


if __name__ == "__main__":
    unittest.main()
