from __future__ import annotations

import unittest

import torch

from scripts.audit_validation_false_positives import classify_false_positives, summarize_records


class ValidationFalsePositiveAuditTests(unittest.TestCase):
    def test_fp_categories_are_disjoint_complete_and_gate_blocks_concentrated_texture(self) -> None:
        probability = torch.zeros(1, 1, 21, 21)
        target = torch.zeros_like(probability)
        target[0, 0, 5, 2] = 1.0
        target[0, 0, 5, 8] = 1.0
        probability[0, 0, 5, 2:9] = 1.0  # One predicted component bridges two GT components.
        target[0, 0, 15, 15] = 1.0
        probability[0, 0, 15, 15] = 1.0
        probability[0, 0, 15, 17] = 1.0  # Protected boundary overextension.
        probability[0, 0, 18, 1:6] = 1.0  # Far texture; three center pixels meet 3x3 cohesion.
        probability[0, 0, 19, 10] = 1.0  # Far isolated false positive.

        categories = classify_false_positives(probability, target)
        self.assertEqual(int(categories["bridge"].sum()), 5)
        self.assertEqual(int(categories["boundary_overextension"].sum()), 1)
        self.assertEqual(int(categories["far_texture"].sum()), 3)
        self.assertEqual(int(categories["far_isolated"].sum()), 3)
        partition = sum(categories[name].to(torch.int64) for name in ("boundary_overextension", "far_isolated", "far_texture", "bridge"))
        self.assertTrue(torch.equal(partition, categories["all_false_positives"].to(torch.int64)))

        concentrated = [
            {
                "stem": "only_texture_source",
                "false_positive_pixels": 100,
                "boundary_overextension_pixels": 0,
                "far_isolated_pixels": 0,
                "far_texture_pixels": 100,
                "bridge_pixels": 0,
                "boundary_overextension_ratio": 0.0,
                "far_isolated_ratio": 0.0,
                "far_texture_ratio": 1.0,
                "bridge_ratio": 0.0,
            }
        ]
        summary = summarize_records(concentrated)
        self.assertEqual(summary["gate"]["decision"], "BLOCK")
        self.assertFalse(summary["gate"]["conditions"]["far_texture_present_in_at_least_3_images"])
        self.assertFalse(summary["gate"]["conditions"]["top2_images_contribute_at_most_50_percent_of_far_texture"])

        distributed = [
            {
                "stem": f"texture_{index}",
                "false_positive_pixels": 20,
                "boundary_overextension_pixels": 0,
                "far_isolated_pixels": 0,
                "far_texture_pixels": 20,
                "bridge_pixels": 0,
                "boundary_overextension_ratio": 0.0,
                "far_isolated_ratio": 0.0,
                "far_texture_ratio": 1.0,
                "bridge_ratio": 0.0,
            }
            for index in range(5)
        ]
        self.assertEqual(summarize_records(distributed)["gate"]["decision"], "PASS")


if __name__ == "__main__":
    unittest.main()
