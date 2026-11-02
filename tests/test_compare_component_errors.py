from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.compare_component_errors import (
    FIXED_THRESHOLD,
    binary_prediction,
    comparison_masks,
    make_image_record,
    rank_component_regressions,
    select_positive_regressions,
    validate_fixed_protocol,
    write_outputs,
)


def _record(name: str, delta: float) -> dict:
    native_error = 2.0
    return make_image_record(
        name,
        f"DeepCrack/test/images/{name}",
        native_f1=0.8,
        ema_f1=0.81,
        native_topology=(0.9, native_error),
        ema_topology=(0.91, native_error + delta),
        native_prediction=np.array([[False, True], [False, False]]),
        ema_prediction=np.array([[True, True], [False, False]]),
    )


class ComponentErrorComparisonTests(unittest.TestCase):
    def test_only_positive_component_regressions_are_ranked_and_exported(self):
        records = [_record("z.png", 2.0), _record("b.png", 1.0), _record("a.png", 1.0), _record("n.png", 0.0), _record("m.png", -1.0)]
        ranked = rank_component_regressions(records)
        self.assertEqual([record["image_name"] for record in ranked], ["z.png", "a.png", "b.png", "n.png", "m.png"])
        selected = select_positive_regressions(records, top_k=2)
        self.assertEqual([record["image_name"] for record in selected], ["z.png", "a.png"])

        with tempfile.TemporaryDirectory() as directory:
            summary = {
                "threshold": FIXED_THRESHOLD,
                "image_size": [384, 544],
                "native": {},
                "ema": {},
                "component_error_delta": 0.0,
                "positive_component_error_images": 3,
            }
            exported = write_outputs(directory, records, summary, visualizations=[], top_k=2)
            self.assertEqual([record["image_name"] for record in exported], ["z.png", "a.png"])
            self.assertTrue((Path(directory) / "visualizations").is_dir())
            with open(Path(directory) / "top_component_regressions.csv", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["image_name"] for row in rows], ["z.png", "a.png"])
            self.assertTrue(all(float(row["delta_component_error"]) > 0.0 for row in rows))

    def test_fixed_threshold_and_difference_masks_are_strict(self):
        validate_fixed_protocol([384, 544], 0.5, 1)
        with self.assertRaises(ValueError):
            validate_fixed_protocol([544, 384], 0.5, 1)
        with self.assertRaises(ValueError):
            validate_fixed_protocol([384, 544], 0.51, 1)

        prediction = binary_prediction(torch.tensor([[0.49, 0.50, 0.51]]))
        self.assertEqual(prediction.tolist(), [[False, False, True]])
        ema_only, native_only = comparison_masks(
            np.array([[False, True], [True, False]]),
            np.array([[True, True], [False, False]]),
        )
        self.assertEqual(ema_only.tolist(), [[True, False], [False, False]])
        self.assertEqual(native_only.tolist(), [[False, False], [True, False]])

    def test_record_delta_is_ema_minus_native(self):
        record = _record("sample.png", 1.5)
        self.assertEqual(record["delta_component_error"], 1.5)
        self.assertAlmostEqual(record["delta_f1"], 0.01)
        self.assertAlmostEqual(record["delta_cldice"], 0.01)
