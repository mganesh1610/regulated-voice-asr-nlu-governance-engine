# Regulated Voice ASR Error Rescoring, 77-Intent NLU, Structured LLM Tool-Calling & Model Governance Engine

End-to-end voice and conversational AI engineering repository executing on the public **PolyAI Banking77** regulated customer call dataset (**13,083 utterances** across **77 fine-grained intents**: `10,003` train + `3,080` holdout test). Designed for regulated financial and healthcare telephony environments requiring low-latency turn-taking, ASR error recovery, calibrated NLU intent recognition, deterministic JSON tool calling, and auditable model governance.

## Architecture & Core Capabilities

1. **Automated Dataset Labeling & Quality Pipeline (`load_and_audit_banking77`):**
   - Verifies cryptographic SHA-256 dataset provenance (`b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` train; `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` test).
   - Audits duplicate transcripts (`4` train duplicates quarantined before fitting), token distributions (`11.71` mean tokens), and Shannon label-balance entropy (`6.2169 bits`, `0.9920` normalized balance ratio across `77` intents).
2. **Telephony ASR Error Root-Causing, Domain-Lexicon Shallow Fusion & Turn-Taking Latency Optimization (`simulate_and_rescore_asr`):**
   - Evaluates exact Levenshtein Word Error Rate (WER) across `33,732` reference words on `3,080` holdout call transcripts under narrowband telephony acoustic confusions.
   - Applies domain-lexicon shallow-fusion rescoring and morphological suffix recovery, cutting ASR WER from **`5.65%` to `0.30%`** (**`94.70%` relative WER reduction**) and recovering **`+6.04 percentage points`** of downstream 77-intent NLU accuracy (`82.44%` -> `88.47%`).
   - Models semantic End-of-Utterance (EOU) turn-taking endpointing versus fixed-silence VAD timeouts, reducing p95 turn-taking latency by **`45.07%`** (`848.3 ms` -> `466.0 ms`; p50 `780.7 ms` -> `434.6 ms`).
3. **Sublinear Word + Character N-Gram Linear Algebra Intent Classifier (`train_and_evaluate_nlu`):**
   - Implements L2-normalized sublinear TF-IDF over word `(1, 3)`-grams and character boundary `(3, 5)`-grams coupled with a sparse inverted-index multi-class margin classifier in pure Python (`math`, `collections`).
   - Achieves **`88.47%` Top-1 Accuracy** (`88.45%` Macro F1) and **`96.20%` Top-3 Accuracy** across all `77` regulated banking intents on the `3,080`-utterance holdout set.
4. **LLM Structured JSON Outputs, Tool Calling & Model Inventory Governance (`run_llm_structured_output_and_governance_audit`):**
   - Generates schema-constrained JSON call summaries, automatic PII digit redaction (`[REDACTED_NUM]`), and deterministic backend service tool calls across all `3,080` test calls with **`100.0%` (`3,080/3,080`) JSON schema & entity-grounding compliance**.
   - Persists model cards, validation metrics, and risk tiers in a normalized SQLite governance catalog (`PRAGMA integrity_check = ok`).

## Verified Benchmark Results (`results/validation_summary.json`)

| Subsystem | Metric | Verified Value |
| :--- | :--- | :--- |
| **Dataset Quality & Labeling** | Total Regulated Call Utterances / Intents | `13,083` (`10,003` train / `3,080` test) across `77` intents |
| **Dataset Quality & Labeling** | Normalized Label Entropy Balance | `0.9920` (`6.2169 bits`) |
| **ASR Rescoring** | Holdout Word Error Rate (Raw -> Rescored) | `5.65%` -> `0.30%` (`94.70%` relative WER reduction) |
| **Voice Turn-Taking** | p95 Endpointing Latency (Baseline -> Semantic EOU) | `848.3 ms` -> `466.0 ms` (`45.07%` latency reduction) |
| **77-Intent NLU** | Clean Holdout Top-1 / Top-3 Accuracy / Macro F1 | `88.47%` Top-1 / `96.20%` Top-3 / `88.45%` Macro F1 |
| **ASR-Robust NLU** | Top-1 Accuracy Recovery After Domain Rescoring | `+6.04 pp` (`82.44%` -> `88.47%`) |
| **LLM Governance** | Structured JSON Tool-Calling & Summary Grounding | `100.0%` (`3,080 / 3,080` holdout calls; `5/5` unit tests passed) |

## Quick Start

```bash
python3 src/voice_asr_nlu_pipeline.py
python3 -m unittest discover -s tests -v
```
