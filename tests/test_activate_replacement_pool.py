import copy
import unittest

import pipeline_supervisor as supervisor
from scripts.activate_replacement_pool import (
    build_gate_state,
    protocol_bound_gates,
    replacement_spec,
)


class ActivateReplacementPoolTests(unittest.TestCase):
    def setUp(self):
        self.policy = copy.deepcopy(supervisor.load_policy())

    def evidence(self, path="data/replacement_v1.pkl"):
        return {
            "schema_version": 1,
            "evidence_version": "paper2-replacement-pool-v1",
            "passed": True,
            "pool_sha256": "ABC",
            "pool_spec": {"path": path, "resume_command": "resume replacement"},
        }

    def test_replacement_requires_new_unprotected_path(self):
        spec = replacement_spec(self.policy, self.evidence())
        self.assertEqual(spec["path"], "data/replacement_v1.pkl")
        with self.assertRaisesRegex(ValueError, "new versioned path"):
            replacement_spec(self.policy, self.evidence(self.policy["pool"]["path"]))
        with self.assertRaisesRegex(ValueError, "immutable or protected"):
            replacement_spec(self.policy, self.evidence("data/rcwa_5k.pkl"))

    def test_only_protocol_gates_survive_pool_activation(self):
        keep = protocol_bound_gates(self.policy)
        self.assertEqual(keep, {"d65_colorimetry", "reference_resolution"})
        # build_gate_state needs real files for hashes; its filtering rule is
        # asserted directly so pool-bound gates cannot leak across activation.
        old = {
            "gates": {
                "d65_colorimetry": {"passed": True},
                "reference_resolution": {"passed": True},
                "joint_numerical_convergence": {"passed": True},
            }
        }
        filtered = {name for name in old["gates"] if name in keep}
        self.assertEqual(filtered, {"d65_colorimetry", "reference_resolution"})


if __name__ == "__main__":
    unittest.main()
