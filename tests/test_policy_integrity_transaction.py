import tempfile
import unittest
from pathlib import Path

from pipeline_supervisor import atomic_json, file_digest, load_json
from scripts.policy_integrity_transaction import (
    apply_policy_integrity_transaction,
    recover_policy_integrity_transaction,
)


class InjectedCrash(RuntimeError):
    pass


class PolicyIntegrityTransactionTests(unittest.TestCase):
    def test_crash_after_policy_replace_recovers_idempotently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy_path = root / "pipeline_policy.json"
            integrity_path = root / ".state/pipeline_integrity.json"
            before_policy = {"strategy_override": {"revision": 1}}
            atomic_json(policy_path, before_policy)
            before_integrity = {
                "schema_version": 1,
                "policy_sha256": file_digest(policy_path),
                "supervisor_sha256": "A" * 64,
                "protected_assets_revision": 7,
            }
            atomic_json(integrity_path, before_integrity)
            after_policy = {"strategy_override": {"revision": 2}}
            expected_policy_path = root / "expected-policy.json"
            atomic_json(expected_policy_path, after_policy)
            after_integrity = {
                **before_integrity,
                "policy_sha256": file_digest(expected_policy_path),
                "protected_assets_revision": 8,
            }

            def crash(stage):
                if stage == "after_policy_replace":
                    raise InjectedCrash(stage)

            with self.assertRaisesRegex(InjectedCrash, "after_policy_replace"):
                apply_policy_integrity_transaction(
                    policy_path,
                    integrity_path,
                    before_policy,
                    before_integrity,
                    after_policy,
                    after_integrity,
                    fault_injector=crash,
                )

            self.assertEqual(load_json(policy_path), after_policy)
            self.assertEqual(load_json(integrity_path), before_integrity)
            journals = list(
                (integrity_path.parent / "policy_integrity_transactions").glob("*.json")
            )
            self.assertEqual(len(journals), 1)
            journal = load_json(journals[0])
            self.assertEqual(journal["status"], "policy_replaced")
            self.assertEqual(journal["before"]["policy"]["sha256"], before_integrity["policy_sha256"])
            self.assertEqual(journal["after"]["policy"]["sha256"], after_integrity["policy_sha256"])

            recovered = recover_policy_integrity_transaction(policy_path, integrity_path)
            self.assertEqual(recovered["status"], "recovered")
            self.assertEqual(load_json(policy_path), after_policy)
            self.assertEqual(load_json(integrity_path), after_integrity)
            self.assertEqual(load_json(journals[0])["status"], "committed")
            self.assertIsNone(
                recover_policy_integrity_transaction(policy_path, integrity_path)
            )
            self.assertEqual(load_json(policy_path), after_policy)
            self.assertEqual(load_json(integrity_path), after_integrity)


if __name__ == "__main__":
    unittest.main()
