from __future__ import annotations

import unittest

from audit_astra_model import classify_structure


class AstraModelAuditTest(unittest.TestCase):
    def test_known_statuses(self):
        ids = [
            "S1_partial_gap_lifetime",
            "S2_temporal_path_coexistence",
            "S3_point_vs_context",
            "S4_blank_reentry_parallel_persistence",
            "S5_partial_shape_scope",
        ]
        for sid in ids:
            out = classify_structure({"id": sid, "representation": {}})
            self.assertIn(out["status"], {"PARTIAL", "NOT_YET"})


if __name__ == "__main__":
    unittest.main()
