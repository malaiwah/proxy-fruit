import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from checkpoint_contract import checkpoint_model_state
from export_fruit import (
    _checkpoint_conventions,
    _fruit_metadata,
    _resolve_stored_moe_inter,
    _set_rope_theta,
)
from fruit_kld import _dense_served_log_probs
from fruit_serve_mtp import _acceptance_stats
from parity_test import _reference_conventions
from publish_fruit import validate_export_manifest
from train_fruit import (
    _restore_trained_theta_marker_precision,
    _resume_checkpoint_theta,
)


class ExportConventionTests(unittest.TestCase):
    def test_checkpoint_model_state_accepts_raw_and_schema_wrapped_payloads(self):
        state = {"weight": torch.tensor([1.0])}
        self.assertIs(checkpoint_model_state(state), state)
        wrapped = {"schema": 2, "model": state, "step": 249}
        self.assertIs(checkpoint_model_state(wrapped), state)

    def test_checkpoint_model_state_rejects_nonmapping_model(self):
        with self.assertRaisesRegex(TypeError, "model state"):
            checkpoint_model_state({"model": torch.tensor([1.0])})

    def test_legacy_checkpoint_requires_explicit_theta(self):
        with self.assertRaisesRegex(RuntimeError, "set FRUIT_ROPE_THETA"):
            _checkpoint_conventions({}, {})

    def test_legacy_checkpoint_uses_explicit_theta(self):
        native, theta = _checkpoint_conventions({}, {"FRUIT_ROPE_THETA": "500000"})
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

    def test_subtile_pilot_intermediate_defaults_to_exact_padding(self):
        self.assertEqual(_resolve_stored_moe_inter(128, None), (256, 256))
        self.assertEqual(_resolve_stored_moe_inter(512, None), (0, 512))
        with self.assertRaisesRegex(ValueError, "divisible by 256"):
            _resolve_stored_moe_inter(128, "0")

    def test_release_metadata_distinguishes_pilot_and_trained_indexer(self):
        pilot = _fruit_metadata("pilot", 128, 256)
        self.assertEqual(pilot["name"], "GLM-5.2-SIQ-Fruit-pilot (Clémentine)")
        self.assertEqual(pilot["logical_moe_intermediate_size"], 128)
        self.assertEqual(pilot["stored_moe_intermediate_size"], 256)
        self.assertTrue(pilot["exact_zero_padding"])
        base = _fruit_metadata("base", 512, 512)
        self.assertEqual(base["name"], "GLM-5.2-SIQ-Fruit")
        self.assertIn("KL-distilled", base["indexer_weights"])

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
            _checkpoint_conventions(state, {"FRUIT_ROPE_THETA": "8000000"})

    def test_checkpoint_rejects_unpaired_markers(self):
        state = {"serve_conv_v": torch.tensor([2], dtype=torch.int32)}
        with self.assertRaisesRegex(RuntimeError, "both be present"):
            _checkpoint_conventions(state, {})

    def test_checkpoint_rejects_nonintegral_marker(self):
        state = {
            "serve_conv_v": torch.tensor([2.5]),
            "rope_theta_trained": torch.tensor([500000.0]),
        }
        with self.assertRaisesRegex(RuntimeError, "unsupported serve_conv_v"):
            _checkpoint_conventions(state, {})

    def test_parity_legacy_checkpoint_uses_rope_theta(self):
        native, theta = _reference_conventions({}, {"ROPE_THETA": "500000"})
        self.assertFalse(native)
        self.assertEqual(theta, 500000.0)

    def test_parity_native_checkpoint_reads_marker_without_consuming_it(self):
        state = {
            "serve_conv_v": torch.tensor([2], dtype=torch.int32),
            "rope_theta_trained": torch.tensor([500000.0], dtype=torch.float64),
        }
        native, theta = _reference_conventions(state, {})
        self.assertTrue(native)
        self.assertEqual(theta, 500000.0)
        self.assertEqual(set(state), {"serve_conv_v", "rope_theta_trained"})

    def test_native_resume_derives_checkpoint_theta(self):
        state = {
            "serve_conv_v": torch.tensor([2], dtype=torch.int32),
            "rope_theta_trained": torch.tensor([500000.0], dtype=torch.float64),
        }
        self.assertEqual(
            _resume_checkpoint_theta(state, {}, True),
            500000.0,
        )

    def test_native_resume_rejects_effective_theta_mismatch(self):
        state = {
            "serve_conv_v": torch.tensor([2], dtype=torch.int32),
            "rope_theta_trained": torch.tensor([500000.0], dtype=torch.float64),
        }
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            _resume_checkpoint_theta(state, {"ROPE_THETA": "10000"}, True)

    def test_legacy_resume_requires_explicit_theta(self):
        with self.assertRaisesRegex(RuntimeError, "set FRUIT_ROPE_THETA"):
            _resume_checkpoint_theta({}, {}, False)

    def test_resume_rejects_layout_mismatch(self):
        state = {
            "serve_conv_v": torch.tensor([2], dtype=torch.int32),
            "rope_theta_trained": torch.tensor([500000.0], dtype=torch.float64),
        }
        with self.assertRaisesRegex(RuntimeError, "SERVE_CONV=0"):
            _resume_checkpoint_theta(state, {}, False)

    def test_theta_marker_survives_bf16_model_conversion_exactly(self):
        model = torch.nn.Module()
        model.register_buffer(
            "rope_theta_trained", torch.tensor([500000.0], dtype=torch.float64)
        )
        model.to(dtype=torch.bfloat16)
        self.assertNotEqual(model.rope_theta_trained.item(), 500000.0)
        _restore_trained_theta_marker_precision(model, 500000.0)
        self.assertEqual(model.rope_theta_trained.dtype, torch.float64)
        self.assertEqual(model.rope_theta_trained.item(), 500000.0)


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
            _acceptance_stats(self.counters(accepted=1001, draft_tokens=1000), 1, 0.50)

    def test_impossible_draft_token_count_fails(self):
        with self.assertRaisesRegex(ValueError, "impossible draft-token count"):
            _acceptance_stats(
                self.counters(accepted=50, draft_tokens=100, drafts=1),
                1,
                0.50,
            )

    def test_fractional_counter_fails(self):
        with self.assertRaisesRegex(ValueError, "must be integral"):
            _acceptance_stats(
                self.counters(accepted=0.5, draft_tokens=1, drafts=1),
                1,
                0.0,
            )

    def test_zero_drafts_fail(self):
        with self.assertRaisesRegex(ValueError, "produced no drafts"):
            _acceptance_stats(
                self.counters(accepted=0, draft_tokens=0, drafts=0), 1, 0.0
            )


class FullVocabularyKLDTests(unittest.TestCase):
    def test_dense_served_distribution_closes_full_vocabulary(self):
        values = {
            2: SimpleNamespace(logprob=-2.0),
            0: SimpleNamespace(logprob=0.0),
            1: SimpleNamespace(logprob=-1.0),
        }
        log_probs = _dense_served_log_probs(values, 3)
        self.assertEqual(log_probs.argmax().item(), 0)
        torch.testing.assert_close(
            log_probs.exp().sum(), torch.tensor(1.0, dtype=torch.float64)
        )

    def test_dense_served_distribution_rejects_truncation(self):
        values = {0: SimpleNamespace(logprob=0.0)}
        with self.assertRaisesRegex(ValueError, "expected full vocabulary"):
            _dense_served_log_probs(values, 2)


class PublicationManifestTests(unittest.TestCase):
    @staticmethod
    def seed_export(root: Path) -> None:
        payload = b'{"model_type":"glm_moe_dsa"}\n'
        (root / "config.json").write_bytes(payload)
        (root / "README.md").write_text("# card\n", encoding="utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        (root / "MANIFEST.sha256").write_text(
            f"{digest}  config.json\n",
            encoding="utf-8",
        )

    def test_manifest_accepts_authenticated_closed_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.seed_export(root)
            self.assertEqual(
                validate_export_manifest(root),
                {
                    "config.json": hashlib.sha256(
                        (root / "config.json").read_bytes()
                    ).hexdigest()
                },
            )

    def test_manifest_rejects_unmanifested_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.seed_export(root)
            (root / "tier_bitmap.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unmanifested"):
                validate_export_manifest(root)

    def test_manifest_rejects_payload_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.seed_export(root)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_export_manifest(root)


if __name__ == "__main__":
    unittest.main()
