#!/usr/bin/env python3
"""Unit test suite for Regulated Voice ASR, 77-Intent NLU & LLM Governance Engine."""

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_asr_nlu_pipeline import (
    compute_sha256,
    EXPECTED_SHA256,
    word_edit_distance,
    run_pipeline,
)


class TestVoiceAsrNluGovernance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_pipeline(ROOT)

    def test_01_dataset_sha256_and_scale(self):
        audit = self.report["dataset_audit"]
        self.assertEqual(audit["train_sha256"], EXPECTED_SHA256["banking77_train.csv"])
        self.assertEqual(audit["test_sha256"], EXPECTED_SHA256["banking77_test.csv"])
        self.assertEqual(audit["train_utterances"], 10003)
        self.assertEqual(audit["test_utterances"], 3080)
        self.assertEqual(audit["total_utterances"], 13083)
        self.assertEqual(audit["unique_intents"], 77)

    def test_02_word_edit_distance_exact(self):
        self.assertEqual(word_edit_distance(["card", "payment", "declined"], ["card", "payment", "declined"]), 0)
        self.assertEqual(word_edit_distance(["card", "payment", "declined"], ["card", "paymint", "deklined"]), 2)

    def test_03_asr_wer_and_turntaking_latency_reduction(self):
        asr = self.report["asr_and_turntaking_metrics"]
        self.assertLess(asr["rescored_asr_wer_pct"], asr["raw_asr_wer_pct"])
        self.assertGreater(asr["relative_wer_reduction_pct"], 50.0)
        self.assertLess(asr["optimized_turn_latency_p95_ms"], asr["baseline_turn_latency_p95_ms"])
        self.assertGreater(asr["p95_turn_latency_reduction_pct"], 35.0)

    def test_04_nlu_77_intent_accuracy_and_asr_recovery(self):
        nlu = self.report["nlu_intent_classification_metrics"]
        self.assertGreaterEqual(nlu["Clean_Test_Top1_Accuracy_Pct"], 88.0)
        self.assertGreaterEqual(nlu["Clean_Test_Top3_Accuracy_Pct"], 96.0)
        self.assertGreater(
            nlu["Rescored_ASR_Test_Top1_Accuracy_Pct"],
            nlu["Raw_ASR_Test_Top1_Accuracy_Pct"],
        )

    def test_05_llm_structured_json_and_sqlite_governance(self):
        gov = self.report["llm_structured_output_and_governance"]
        self.assertEqual(gov["evaluated_call_summaries"], 3080)
        self.assertEqual(gov["schema_compliance_rate_pct"], 100.0)
        self.assertEqual(self.report["sqlite_integrity_check"], "ok")


if __name__ == "__main__":
    unittest.main()
