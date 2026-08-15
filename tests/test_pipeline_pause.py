import unittest

from scripts import set_pipeline_pause as pause


class PipelinePauseTests(unittest.TestCase):
    def test_pause_is_exact_request_scoped_and_refreshes_runtime_hashes(self):
        before = {
            "strategy_override": {
                "evidence": [
                    {"path": "pipeline_supervisor.py", "sha256": "OLD"},
                    {"path": "tests/test_pipeline_supervisor.py", "sha256": "OLD"},
                ]
            }
        }
        after = pause.build_after(before, "request-1", True)
        bound = after["operations"]["pause_after_request"]
        self.assertTrue(bound["enabled"])
        self.assertEqual(bound["request_id"], "request-1")
        self.assertEqual(bound["resume_requires"], "explicit_user_authorization")
        self.assertNotEqual(after["strategy_override"]["evidence"][0]["sha256"], "OLD")
        self.assertNotEqual(after["strategy_override"]["evidence"][1]["sha256"], "OLD")

        resumed = pause.build_after(after, "request-1", False)
        self.assertFalse(resumed["operations"]["pause_after_request"]["enabled"])


if __name__ == "__main__":
    unittest.main()
