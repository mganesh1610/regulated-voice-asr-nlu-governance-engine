# Data Sources, Schema Mapping & Model Governance Documentation

## 1. Primary & Backup Public Datasets

- **Primary Executed Dataset:** PolyAI Banking77 Conversational Intent Dataset (`13,083` customer banking utterances across `77` intents).
  - Train split (`10,003` rows, SHA-256 `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`): `https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv`
  - Test split (`3,080` rows, SHA-256 `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`): `https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv`
- **Backup Public Dataset:** Amazon Science MASSIVE / SLURP Spoken Language Understanding Dataset (`16,521` utterances across `60` intents, `https://huggingface.co/datasets/AmazonScience/massive`) and OpenSLR LibriSpeech ASR Corpus (`https://www.openslr.org/12/`).

## 2. SQLite Governance Schema (`voice_governance_inventory.db`)

- `utterance_catalog(utterance_id TEXT PK, split TEXT, category TEXT, token_count INTEGER, char_length INTEGER, is_exact_duplicate INTEGER)`
- `model_inventory_governance(model_id TEXT PK, task_domain TEXT, dataset_scale INTEGER, primary_metric_name TEXT, primary_metric_value REAL, risk_tier TEXT, validation_status TEXT)`

## 3. Mathematical Formulations

- **Sublinear TF-IDF Weighting:** $w_{t,d} = (1 + \ln f_{t,d}) \cdot \left(\ln\frac{1 + N}{1 + \text{df}_t} + 1\right)$ normalized to unit $L_2$ norm $\hat{\mathbf{x}}_d = \mathbf{w}_d / \|\mathbf{w}_d\|_2$.
- **Word Error Rate (WER):** $\text{WER} = \frac{S + D + I}{N_{\text{ref}}} \times 100\%$ computed via dynamic-programming Levenshtein word alignment across `33,732` reference tokens.
- **Multi-Class Margin Update:** For document $\hat{\mathbf{x}}_i$ with ground-truth intent $y_i$ and highest-scoring competing intent $\hat{y}_i = \arg\max_{c \neq y_i} \mathbf{w}_c^\top \hat{\mathbf{x}}_i$, updates $\mathbf{w}_{y_i} \leftarrow \mathbf{w}_{y_i} + \eta \hat{\mathbf{x}}_i$ and $\mathbf{w}_{\hat{y}_i} \leftarrow \mathbf{w}_{\hat{y}_i} - \eta \hat{\mathbf{x}}_i$ whenever $\mathbf{w}_{y_i}^\top \hat{\mathbf{x}}_i - \mathbf{w}_{\hat{y}_i}^\top \hat{\mathbf{x}}_i < \gamma$.
