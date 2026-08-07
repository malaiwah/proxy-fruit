import unittest

import torch

from export_fruit import _checkpoint_conventions, _set_rope_theta
from fruit_serve_mtp import _acceptance_stats


class ExportConventionTests(unittest.TestCase):
    def test_legacy_checkpoint_requires_explicit_theta(self):
        with self.assertRaisesRegex(RuntimeError, "set FRUIT_ROPE_THETA"):
            _checkpoint_conventions({}, {})

    def test_legacy_checkpoint_uses_explicit_theta(self):
        native, theta = _checkpoint_conventions(
            {}, {"FRUIT_ROPE_THETA": "500000"})
        self.assertFalse(native)
        self.assertEqual(theta, 500000.0)

    def test_theta_replaces_both_parent_config_defaults(self):
        config = {
            "rope_theta": 8000000,
            "rope_parameters": {
                "rope_theta": 8000000,
                "rope_type": "default",
            },
        }
        _set_rope_theta(config, 500000.0)
        self.assertEqual(config["rope_theta"], 500000.0)
        self.assertEqual(config["rope_parameters"]["rope_theta"], 500000.0)
        self.assertEqual(config["rope_parameters"]["rope_type"], "default")

    def test_native_checkpoint_uses_paired_markers(self):
        state = {
            "serve_conv_v": torch.tensor([2], dtype=torch.int32),
            "rope_theta_trained": torch.tensor([500000.0], dtype=torch.float64),
        }
        native, theta = _checkpoint_conventions(state, {})
        self.assertTrue(native)
        self.assertEqual(theta, 500000.0)
        self.assertEqual(state, {})

    def test_native_checkpoint_rejects_theta_override_mismatch(self):
        state = {
            "serve_conv_v": torch.tensor([2], dtype=torch.int32),
            "rope_theta_trained": torch.tensor([500000.0], dtype=torch.float64),
        }
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            _checkpoint_conventions(
                state, {"FRUIT_ROPE_THETA": "8000000"})

    def test_checkpoint_rejects_unpaired_markers(self):
        state = {"serve_conv_v": torch.tensor([2], dtype=torch.int32)}
        with self.assertRaisesRegex(RuntimeError, "both be present"):
            _checkpoint_conventions(state, {})


class MTPAcceptanceTests(unittest.TestCase):
    @staticmethod
    def counters(accepted=790, draft_tokens=1000, drafts=1000):
        return {
            "vllm:spec_decode_num_drafts": drafts,
            "vllm:spec_decode_num_draft_tokens": draft_tokens,
            "vllm:spec_decode_num_accepted_tokens": accepted,
        }

    def test_acceptance_above_threshold_passes(self):
        rate, mean = _acceptance_stats(self.counters(), 1, 0.50)
        self.assertAlmostEqual(rate, 0.79)
        self.assertAlmostEqual(mean, 1.79)

    def test_structural_mtp_regression_fails(self):
        with self.assertRaisesRegex(ValueError, "below required"):
            _acceptance_stats(
                self.counters(accepted=4, draft_tokens=1016, drafts=1016),
                1,
                0.50,
            )

    def test_missing_metrics_fail(self):
        with self.assertRaisesRegex(ValueError, "missing spec-decode counters"):
            _acceptance_stats({}, 1, 0.50)

    def test_impossible_counter_values_fail(self):
        with self.assertRaisesRegex(ValueError, "invalid accepted-token count"):
            _acceptance_stats(
                self.counters(accepted=1001, draft_tokens=1000), 1, 0.50)

    def test_zero_drafts_fail(self):
        with self.assertRaisesRegex(ValueError, "produced no drafts"):
            _acceptance_stats(
                self.counters(accepted=0, draft_tokens=0, drafts=0), 1, 0.0)


if __name__ == "__main__":
    unittest.main()
