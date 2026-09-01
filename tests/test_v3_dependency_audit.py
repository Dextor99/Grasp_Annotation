import unittest


class V3DependencyAuditTests(unittest.TestCase):
    def test_audit_confirms_v3_is_auxiliary_and_v4_controls_merge(self):
        from scripts.audit_v3_dependency import run_audit

        report = run_audit()

        self.assertTrue(all(item["passed"] for item in report))
        checks = {item["check"]: item for item in report}
        self.assertTrue(checks["v3_geometry_input_for_refinement"]["passed"])
        self.assertTrue(checks["v3_not_used_for_merge_selection"]["passed"])
        self.assertTrue(checks["v4_score_used_for_merge"]["passed"])
        self.assertTrue(checks["v4_score_used_for_final_export"]["passed"])


if __name__ == "__main__":
    unittest.main()
